"""Integration tests for team-mode ACL: no leakage across interfaces (task 52).

The full stack runs with ``auth.mode="oidc"`` validated against a locally
generated RSA key (the JWKS client is stubbed; no network): a temporary source
root with ``teamA/`` and ``teamB/`` is ingested with a ``FakeEmbedder``
(dim 8), the KG is seeded through the real ``kg_extract`` →
``resolve_entities`` → ``kg_temporal_refresh`` handlers with a ``FakeChat``
returning fixed extraction JSON, and everything is synced into per-test
OpenSearch aliases. The secret "Projeto Omega budget 99k€" exists only in
``teamB/secret.md`` (with its KG relation), and ``teamA/idea_v2.md``
supersedes ``teamB/idea.md``. ``alice`` is granted ``teamA`` only; the tests
assert that search, answers, assembled context, version notes, files,
documents, entity cards and tag counts never leak team-B content, while an
admin sees everything.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from jwt.algorithms import RSAAlgorithm
from sqlalchemy import delete, func, select
from testcontainers.community.opensearch import OpenSearchContainer

from fakes import FakeChat, FakeEmbedder, FakeReranker
from mynotes.api.app import create_app
from mynotes.api.auth import principal_from_claims
from mynotes.config import Settings
from mynotes.core.acl import clear_cache
from mynotes.core.principal import LOCAL, Principal
from mynotes.db.models import (
    AclGrant,
    DocRelation,
    Document,
    DocumentTag,
    EmbeddingCache,
    File,
    IndexOutbox,
    Job,
    KgEntity,
    KgEntityAlias,
    KgEntityDistinct,
    KgExtractionCache,
    KgMention,
    KgRelation,
    KgRelationEvidence,
    ScanRun,
    SourceRoot,
    Tag,
    TagAlias,
    Workspace,
)
from mynotes.db.session import reset_engine, session_scope
from mynotes.graph.ontology import EXTRACTOR_VERSION
from mynotes.index.opensearch import ensure_indices, ensure_pipeline, get_client
from mynotes.index.sync import sync_once
from mynotes.jobs import pipeline
from mynotes.jobs.queue import complete
from mynotes.jobs.registry import JobContext
from mynotes.jobs.worker import Worker
from mynotes.retrieval import graph_expand, version_expand
from mynotes.retrieval.assemble import assemble_context
from mynotes.retrieval.pipeline import retrieve
from mynotes.retrieval.prompts import NOT_FOUND
from mynotes.retrieval.types import SearchFilters
from mynotes.scan.scanner import scan_root

pytestmark = pytest.mark.integration

ISSUER = "https://keycloak.example/realms/acl-test"
AUDIENCE = "mynotes"

_MTIME = 1_000_000_000_000_000_000
_JOB_TYPES = (
    "extract",
    "chunk",
    "embed",
    "apply_retention",
    "reindex_metadata",
    "delete_document",
    "version_detect",
    "tag_llm",
    "link_resolve",
    "kg_extract",
    "resolve_entities",
    "adjudicate_entities",
    "kg_temporal_refresh",
    "refresh_summaries",
)

_PATCH_TARGETS = (
    "mynotes.config.get_settings",
    "mynotes.db.session.get_settings",
    "mynotes.scan.scanner.get_settings",
    "mynotes.jobs.worker.get_settings",
    "mynotes.llm.limits.get_settings",
    "mynotes.index.sync.get_settings",
    "mynotes.index.opensearch.get_settings",
    "mynotes.index.documents.get_settings",
    "mynotes.graph.resolve.get_settings",
    "mynotes.graph.store.get_settings",
    "mynotes.embed.client.get_settings",
    "mynotes.retrieval.hybrid.get_settings",
    "mynotes.retrieval.analyze.get_settings",
    "mynotes.retrieval.pipeline.get_settings",
    "mynotes.retrieval.entity_link.get_settings",
    "mynotes.retrieval.graph_expand.get_settings",
    "mynotes.retrieval.version_expand.get_settings",
    "mynotes.retrieval.assemble.get_settings",
    "mynotes.core.setup.get_settings",
    "mynotes.core.service.get_settings",
    "mynotes.core.acl.get_settings",
    "mynotes.api.auth.get_settings",
    "mynotes.api.schemas.get_settings",
)

# The question shares no token with the teamA note, so with the FakeReranker
# its rerank scores are 0 and the retrieval gate answers "not found" before
# any model call - the restricted corpus simply has nothing relevant.
_SECRET_QUESTION = "budget do Projeto Omega"
_GROUNDED_QUESTION = "Qual e o orcamento da ideia revisada?"

_IDEA_OLD_PATH = "teamB/idea.md"
_IDEA_NEW_PATH = "teamA/idea_v2.md"
_SECRET_PATH = "teamB/secret.md"

# No token of the teamA note may appear in _SECRET_QUESTION, and neither note
# may contain the secret's tokens, so the gate and the ACL assertions stay
# attributable to the ACL alone.
_IDEA_NEW = """---
title: Proposta Revisada
date: 2026-03-01
supersedes: teamB/idea.md
---

# Ideia

Equipe aprovou nossa ideia revisada com orcamento atualizado e cronograma
novo. As entregas ficam divididas por fase, com uma meta final clara para
junho e responsaveis definidos em cada etapa planejada. A primeira etapa
cobre o levantamento das necessidades das areas, a segunda prepara os
materiais de apoio e a ultima consolida os resultados para aprovacao final
da coordenacao junto as areas interessadas.
"""

_IDEA_OLD = """---
title: Proposta Original
date: 2026-01-05
---

# Ideia

Primeira versao da ideia com escopo curto, orcamento inicial e cronograma
preliminar aprovado pela equipe. O documento original descreve as metas da
primeira entrega, os responsaveis envolvidos e os criterios de aceitacao
combinados na reuniao de abertura com as areas interessadas do laboratorio,
alem das premissas adotadas para a estimativa inicial de esforco.
"""

_SECRET = """---
title: Projeto Omega
date: 2026-02-10
---

# Orcamento

O Projeto Omega tem budget 99k€ aprovado pela diretoria para a fase piloto
do novo produto interno. A verba cobre licencas, servicos de consultoria e
uma reserva de contingencia. O fechamento contabil fica com a area
financeira e qualquer remanejamento precisa de autorizacao formal antes da
execucao da despesa prevista para este trimestre.
"""

_SECRET_MENTION = "99k€"


# ---------------------------------------------------------------------------
# OIDC test identity: locally generated RSA key, stubbed JWKS client
# ---------------------------------------------------------------------------


def _keypair() -> tuple[bytes, dict[str, Any]]:
    """One RSA key pair as (private PEM, JWK with kid/alg/use)."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk: dict[str, Any] = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    jwk["kid"] = "acl-test-key"
    jwk["alg"] = "RS256"
    jwk["use"] = "sig"
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pem, jwk


_PEM, _JWK = _keypair()


class StubJwksClient:
    """Duck-typed :class:`jwt.PyJWKClient` serving the static test key."""

    def get_signing_key_from_jwt(self, token: str) -> Any:
        return SimpleNamespace(key=jwt.PyJWK(_JWK))


def _claims(user: str, groups: Sequence[str] = ()) -> dict[str, Any]:
    """Valid OIDC claims for one test user."""
    import datetime as dt

    now = dt.datetime.now(dt.UTC)
    return {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": user,
        "preferred_username": user,
        "groups": list(groups),
        "iat": int(now.timestamp()),
        "exp": int((now + dt.timedelta(hours=1)).timestamp()),
    }


def _token(user: str, groups: Sequence[str] = ()) -> str:
    return jwt.encode(
        _claims(user, groups), _PEM, algorithm="RS256", headers={"kid": "acl-test-key"}
    )


def _bearer(user: str, groups: Sequence[str] = ()) -> dict[str, str]:
    return {"Authorization": f"Bearer {_token(user, groups)}"}


def _client() -> TestClient:
    """The real FastAPI app under test."""
    return TestClient(create_app())


def _principal(env: AclEnv, user: str, groups: Sequence[str] = ()) -> Principal:
    """The authenticated principal exactly as the API dependency builds it."""
    return principal_from_claims(_claims(user, groups), env.settings.auth)


# ---------------------------------------------------------------------------
# Corpus + KG seeding
# ---------------------------------------------------------------------------


def _omega_extraction() -> str:
    """The secret note's extraction: Projeto Omega approved by Diretoria."""
    return json.dumps(
        {
            "entities": [
                {
                    "name": "Projeto Omega",
                    "type": "Project",
                    "description": "projeto secreto de orcamento",
                    "aliases": [],
                },
                {
                    "name": "Diretoria",
                    "type": "Organization",
                    "description": "aprovou o budget",
                    "aliases": [],
                },
            ],
            "relations": [
                {
                    "source": "Diretoria",
                    "target": "Projeto Omega",
                    "type": "approves",
                    "raw_label": "aprovado pela",
                    "description": "",
                    "attribute": "budget",
                    "value": "99k EUR",
                    "quote": "budget 99k€ aprovado pela diretoria",
                    "confidence": 0.9,
                }
            ],
        }
    )


def _empty_extraction() -> str:
    """Notes without KG content."""
    return json.dumps({"entities": [], "relations": []})


def _kg_answer(messages: Sequence[Any]) -> str:
    """Extraction JSON keyed on the prompt text (order-independent)."""
    text = "\n".join(message.content for message in messages)
    return _omega_extraction() if "Projeto Omega" in text else _empty_extraction()


@dataclass
class AclEnv:
    """The registered test root, the fakes and the test's aliases."""

    settings: Settings
    fake: FakeEmbedder
    reranker: FakeReranker
    answer_chat: FakeChat
    root_id: int
    workspace_id: int
    root_path: Path
    chunks_alias: str


def write_file(path: Path, content: str, mtime_ns: int = _MTIME) -> None:
    """Create a file with a fixed mtime so scans are deterministic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))


def drain(*job_types: str) -> None:
    """Run the worker until no job of the given types is claimable."""
    worker = Worker(concurrency=1, poll_interval_s=0.05, types=list(job_types))
    while worker.run_once():
        pass


def run_queued(env: AclEnv, job_type: str) -> int:
    """Run every queued job of one type through its handler; return the count."""
    handled = 0
    while True:
        with session_scope() as session:
            job = (
                session.execute(
                    select(Job)
                    .where(Job.type == job_type, Job.status == "queued")
                    .order_by(Job.id)
                    .limit(1)
                )
                .scalars()
                .first()
            )
            if job is None:
                return handled
            job_id = job.id
            payload = dict(job.payload)
        context = JobContext(worker_id="test", settings=env.settings, embedder=env.fake)
        getattr(pipeline, f"handle_{job_type}")(payload, context)
        with session_scope() as session:
            complete(session, job_id)
        handled += 1


def latest_version_id(env: AclEnv, rel_path: str) -> int:
    """The latest version id of the document stored for ``rel_path``."""
    with session_scope() as session:
        file_row = session.execute(
            select(File).where(File.source_root_id == env.root_id, File.rel_path == rel_path)
        ).scalar_one()
        document = session.get(Document, file_row.document_id)
        assert document is not None and document.latest_version_id is not None
        return int(document.latest_version_id)


def document_id_of(env: AclEnv, rel_path: str) -> int:
    """The document id stored for ``rel_path``."""
    with session_scope() as session:
        file_row = session.execute(
            select(File).where(File.source_root_id == env.root_id, File.rel_path == rel_path)
        ).scalar_one()
        assert file_row.document_id is not None
        return int(file_row.document_id)


def refresh(env: AclEnv) -> None:
    """Make the synced chunks and entities visible to search."""
    client = get_client()
    client.indices.refresh(index=env.chunks_alias)
    client.indices.refresh(index=env.settings.opensearch.entities_alias)


@pytest.fixture
def acl_env(
    migrated_dsn: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    os_container: OpenSearchContainer,
) -> Iterator[AclEnv]:
    """OIDC settings, fakes, per-test aliases and the ingested teamA/teamB corpus."""
    settings = Settings(
        postgres={"dsn": migrated_dsn},
        auth={
            "mode": "oidc",
            "oidc": {"issuer": ISSUER, "audience": AUDIENCE},
            "admin_principals": ["admin"],
        },
    )
    config = os_container.get_config()
    suffix = uuid4().hex[:8]
    settings.opensearch.url = f"http://{config['host']}:{config['port']}"
    settings.opensearch.chunks_alias = f"t{suffix}_chunks"
    settings.opensearch.entities_alias = f"t{suffix}_entities"
    settings.models.embedding.dim = 8
    settings.scan.settle_seconds = 0
    settings.scan.delete_grace_seconds = 0
    settings.versions.min_interval_hours = 0.0
    settings.retrieval.min_rerank_score = 0.005
    settings.retrieval.graph_expansion = True
    for target in _PATCH_TARGETS:
        monkeypatch.setattr(target, lambda: settings)
    monkeypatch.setattr("mynotes.api.auth._jwks_client", lambda issuer, cache_s: StubJwksClient())
    fake = FakeEmbedder(dim=8)
    settings.models.embedding.model = fake.model
    monkeypatch.setattr("mynotes.embed.client.get_embedder", lambda: fake)
    monkeypatch.setattr("mynotes.retrieval.hybrid.get_embedder", lambda: fake)
    reranker = FakeReranker()
    monkeypatch.setattr("mynotes.retrieval.pipeline.get_reranker", lambda: reranker)
    answer_chat = FakeChat("Resposta generica do modelo de teste.")
    monkeypatch.setattr("mynotes.retrieval.answer.get_chat", lambda purpose="answer": answer_chat)
    monkeypatch.setattr(
        "mynotes.jobs.pipeline.get_chat", lambda purpose="extract": FakeChat(_kg_answer)
    )
    reset_engine()
    with session_scope() as session:
        workspace = Workspace(name=f"acl-{suffix}")
        session.add(workspace)
        session.flush()
        root = SourceRoot(workspace_id=workspace.id, name="root", abs_path=str(tmp_path))
        session.add(root)
        session.flush()
        settings.workspace = workspace.name
        env = AclEnv(
            settings=settings,
            fake=fake,
            reranker=reranker,
            answer_chat=answer_chat,
            root_id=root.id,
            workspace_id=int(workspace.id),
            root_path=tmp_path,
            chunks_alias=settings.opensearch.chunks_alias,
        )
    ensure_indices(get_client(), settings)
    ensure_pipeline(get_client(), settings)
    version_expand.register()
    graph_expand.register()
    clear_cache()
    yield env
    from mynotes.retrieval.assemble import unregister_extra_block
    from mynotes.retrieval.pipeline import unregister_expander

    unregister_expander("graph")
    unregister_extra_block("graph_facts")
    client = get_client()
    for alias in (settings.opensearch.chunks_alias, settings.opensearch.entities_alias):
        try:
            physical = list(client.indices.get(index=f"{alias}_v*"))
        except Exception:
            physical = []
        for name in physical:
            client.indices.delete(index=name)
    with session_scope() as session:
        session.execute(delete(AclGrant).where(AclGrant.workspace_id == env.workspace_id))
        session.execute(delete(KgEntityDistinct))
        session.execute(delete(KgRelationEvidence))
        session.execute(delete(KgRelation))
        session.execute(delete(KgMention))
        session.execute(delete(KgEntityAlias))
        session.execute(delete(KgEntity))
        session.execute(delete(Job).where(Job.type.in_(_JOB_TYPES)))
        session.execute(delete(IndexOutbox))
        session.execute(
            delete(KgExtractionCache).where(
                KgExtractionCache.extractor_version == EXTRACTOR_VERSION
            )
        )
        session.execute(delete(EmbeddingCache).where(EmbeddingCache.model == fake.model))
        session.execute(delete(TagAlias).where(TagAlias.workspace_id == env.workspace_id))
        session.execute(delete(Tag).where(Tag.workspace_id == env.workspace_id))
        session.execute(delete(File).where(File.source_root_id == env.root_id))
        session.execute(delete(Document).where(Document.source_root_id == env.root_id))
        session.execute(delete(ScanRun).where(ScanRun.source_root_id == env.root_id))
        session.execute(delete(SourceRoot).where(SourceRoot.id == env.root_id))
        session.execute(delete(Workspace).where(Workspace.id == env.workspace_id))
    clear_cache()
    reset_engine()


@pytest.fixture
def seeded_env(acl_env: AclEnv) -> AclEnv:
    """Ingest the corpus, extract the KG, sync it and grant alice/admin."""
    env = acl_env
    for rel_path, content in (
        (_IDEA_NEW_PATH, _IDEA_NEW),
        (_IDEA_OLD_PATH, _IDEA_OLD),
        (_SECRET_PATH, _SECRET),
    ):
        write_file(env.root_path / rel_path, content)
    scan_root(env.root_id)
    drain("extract", "chunk")
    for rel_path in (_IDEA_NEW_PATH, _IDEA_OLD_PATH, _SECRET_PATH):
        pipeline.handle_embed(
            {"version_id": latest_version_id(env, rel_path)},
            JobContext(worker_id="test", settings=env.settings, embedder=env.fake),
        )
    drain("apply_retention")
    drain("version_detect")
    drain("reindex_metadata")
    run_queued(env, "kg_extract")
    run_queued(env, "resolve_entities")
    run_queued(env, "kg_temporal_refresh")

    secret_doc = document_id_of(env, _SECRET_PATH)
    idea_new_doc = document_id_of(env, _IDEA_NEW_PATH)
    with session_scope() as session:
        relation = (
            session.execute(
                select(DocRelation).where(
                    DocRelation.src_document_id == idea_new_doc,
                    DocRelation.dst_document_id == document_id_of(env, _IDEA_OLD_PATH),
                    DocRelation.type == "supersedes",
                )
            )
            .scalars()
            .all()
        )
        assert relation, "the frontmatter supersedes relation must exist"
        omega = (
            session.execute(select(KgEntity).where(KgEntity.canonical_name == "Projeto Omega"))
            .scalars()
            .all()
        )
        assert len(omega) == 1, "the secret note's entity must be resolved"
        assert (
            session.execute(
                select(func.count())
                .select_from(KgMention)
                .where(KgMention.entity_id == omega[0].id)
            ).scalar_one()
            > 0
        ), "Projeto Omega must be mentioned in a chunk"

        orcamento = Tag(
            workspace_id=env.workspace_id, name="orcamento", display_name="orcamento", kind="topic"
        )
        confidencial = Tag(
            workspace_id=env.workspace_id,
            name="confidencial",
            display_name="confidencial",
            kind="topic",
        )
        session.add_all([orcamento, confidencial])
        session.flush()
        session.add(
            DocumentTag(
                document_id=idea_new_doc, tag_id=orcamento.id, source="manual", status="accepted"
            )
        )
        session.add(
            DocumentTag(
                document_id=secret_doc, tag_id=confidencial.id, source="manual", status="accepted"
            )
        )
        session.add_all(
            [
                AclGrant(workspace_id=env.workspace_id, principal="alice", path_prefix="teamA"),
                AclGrant(
                    workspace_id=env.workspace_id,
                    principal="group:equipe-alfa",
                    path_prefix="teamA",
                ),
                AclGrant(workspace_id=env.workspace_id, principal="admin", path_prefix=""),
            ]
        )
    sync_once()
    refresh(env)
    clear_cache()
    return env


# ---------------------------------------------------------------------------
# 1. Search
# ---------------------------------------------------------------------------


def test_search_hides_team_b_from_alice_and_admin_sees_it(seeded_env: AclEnv) -> None:
    """Search as alice never returns teamB chunks; the admin sees the secret."""
    with _client() as client:
        alice = client.post(
            "/v1/search",
            json={"query": _SECRET_QUESTION},
            headers=_bearer("alice", ["equipe-alfa"]),
        )
        admin = client.post(
            "/v1/search", json={"query": _SECRET_QUESTION}, headers=_bearer("admin")
        )

    assert alice.status_code == 200
    alice_paths = [hit["path"] for hit in alice.json()["hits"]]
    assert all(not path.startswith("teamB/") for path in alice_paths)
    assert all("99k" not in hit["text"] for hit in alice.json()["hits"])

    assert admin.status_code == 200
    admin_paths = [hit["path"] for hit in admin.json()["hits"]]
    assert _SECRET_PATH in admin_paths
    assert any("99k" in hit["text"] for hit in admin.json()["hits"])


# ---------------------------------------------------------------------------
# 2. Ask: gated (not found) and grounded paths
# ---------------------------------------------------------------------------


def test_ask_about_the_secret_answers_not_found(seeded_env: AclEnv) -> None:
    """The gated ask for a secret question returns not-found without an LLM call."""
    env = seeded_env
    with _client() as client:
        response = client.post(
            "/v1/ask", json={"question": _SECRET_QUESTION}, headers=_bearer("alice")
        )

    assert response.status_code == 200
    body = response.json()
    assert body["relevant"] is False
    assert body["answer"] in set(NOT_FOUND.values())
    assert body["citations"] == []
    assert env.answer_chat.calls == [], "the gated answer must not reach the model"


def test_grounded_ask_context_never_contains_team_b(seeded_env: AclEnv) -> None:
    """When the model is called, its context carries teamA content only."""
    env = seeded_env
    with _client() as client:
        response = client.post(
            "/v1/ask", json={"question": _GROUNDED_QUESTION}, headers=_bearer("alice")
        )

    assert response.status_code == 200
    assert env.answer_chat.calls, "the grounded question must reach the model"
    for messages in env.answer_chat.calls:
        text = "\n".join(message.content for message in messages)
        assert "99k" not in text
        assert "Omega" not in text
        assert "teamB" not in text
        assert "Proposta Original" not in text
    for citation in response.json()["citations"]:
        assert not citation["path"].startswith("teamB/")


# ---------------------------------------------------------------------------
# 3. Graph expansion
# ---------------------------------------------------------------------------


def test_graph_expansion_respects_the_acl(seeded_env: AclEnv) -> None:
    """With graph expansion on, alice's retrieval still has no teamB content."""
    env = seeded_env
    alice = _principal(env, "alice", ["equipe-alfa"])
    admin = _principal(env, "admin")

    result = retrieve(_SECRET_QUESTION, SearchFilters(acl_prefixes=alice.acl_prefixes or []))
    assert all(not hit.path.startswith("teamB/") for hit in result.hits)
    bundle = assemble_context(result)
    assert "99k" not in bundle.text
    assert "Omega" not in bundle.text
    assert "teamB" not in bundle.text

    admin_result = retrieve(_SECRET_QUESTION, SearchFilters(acl_prefixes=admin.acl_prefixes or []))
    admin_bundle = assemble_context(admin_result)
    assert "99k" in admin_bundle.text, "the admin must still see the secret in context"


# ---------------------------------------------------------------------------
# 4. Version notes
# ---------------------------------------------------------------------------


def test_version_notes_hide_the_superseded_team_b_document(seeded_env: AclEnv) -> None:
    """The version-notes block drops lines naming a restricted document."""
    env = seeded_env
    alice = _principal(env, "alice", ["equipe-alfa"])

    result = retrieve(_GROUNDED_QUESTION, SearchFilters(acl_prefixes=alice.acl_prefixes or []))
    bundle = assemble_context(result)
    assert "supersedes" not in bundle.text
    assert "Proposta Original" not in bundle.text
    assert "teamB" not in bundle.text

    unrestricted = retrieve(_GROUNDED_QUESTION)
    unrestricted_bundle = assemble_context(unrestricted)
    assert "supersedes" in unrestricted_bundle.text, "the relation line must exist unfiltered"
    assert "Proposta Original" in unrestricted_bundle.text


# ---------------------------------------------------------------------------
# 5. Files and documents
# ---------------------------------------------------------------------------


def test_secret_file_and_document_denied_for_alice(seeded_env: AclEnv) -> None:
    """Direct id access to a restricted document is 403 for alice, 200 for admin."""
    env = seeded_env
    secret_doc = document_id_of(env, _SECRET_PATH)
    with _client() as client:
        alice_file = client.get(f"/v1/files/{secret_doc}/raw", headers=_bearer("alice"))
        alice_doc = client.get(f"/v1/documents/{secret_doc}", headers=_bearer("alice"))
        admin_file = client.get(f"/v1/files/{secret_doc}/raw", headers=_bearer("admin"))
        admin_doc = client.get(f"/v1/documents/{secret_doc}", headers=_bearer("admin"))

    assert alice_file.status_code in (403, 404)
    assert alice_doc.status_code in (403, 404)
    assert admin_file.status_code == 200
    assert "99k€".encode() in admin_file.content
    assert admin_doc.status_code == 200


# ---------------------------------------------------------------------------
# 6. Entity cards
# ---------------------------------------------------------------------------


def test_entity_card_hides_the_restricted_summary(seeded_env: AclEnv) -> None:
    """Alice's Projeto Omega card has no summary, documents or relations."""
    with _client() as client:
        alice = client.get(
            "/v1/graph/entities", params={"q": "Projeto Omega"}, headers=_bearer("alice")
        )
        admin = client.get(
            "/v1/graph/entities", params={"q": "Projeto Omega"}, headers=_bearer("admin")
        )

    assert alice.status_code == 200
    alice_card = alice.json()
    assert alice_card["description"] == "", "the shared summary must stay hidden"
    assert alice_card["top_documents"] == []
    assert alice_card["mention_count_current"] == 0
    assert alice_card["relations_out"] == 0
    assert alice_card["relations_in"] == 0

    assert admin.status_code == 200
    admin_card = admin.json()
    assert admin_card["description"] != ""
    assert any(doc["rel_path"] == _SECRET_PATH for doc in admin_card["top_documents"])


# ---------------------------------------------------------------------------
# 7. Tag counts
# ---------------------------------------------------------------------------


def test_tag_counts_only_count_visible_documents(seeded_env: AclEnv) -> None:
    """Restricted tag counts ignore documents outside the allowed prefixes."""
    with _client() as client:
        alice = client.get("/v1/tags", headers=_bearer("alice"))
        admin = client.get("/v1/tags", headers=_bearer("admin"))

    assert alice.status_code == 200 and admin.status_code == 200
    alice_counts = {tag["name"]: tag["document_count"] for tag in alice.json()}
    admin_counts = {tag["name"]: tag["document_count"] for tag in admin.json()}
    assert alice_counts["orcamento"] == 1
    assert alice_counts["confidencial"] == 0
    assert admin_counts["orcamento"] == 1
    assert admin_counts["confidencial"] == 1


# ---------------------------------------------------------------------------
# 8. Mutations and status are admin-only
# ---------------------------------------------------------------------------


def test_mutations_and_status_require_admin(seeded_env: AclEnv) -> None:
    """Alice gets 403 on mutations and status; the admin may run them."""
    with session_scope() as session:
        relation_id = int(
            session.execute(select(DocRelation.id).where(DocRelation.type == "supersedes"))
            .scalars()
            .first()
        )
    with _client() as client:
        alice_merge = client.post(
            "/v1/tags/merge",
            json={"from": "orcamento", "into": "planejamento"},
            headers=_bearer("alice"),
        )
        alice_confirm = client.post(f"/v1/versions/{relation_id}/confirm", headers=_bearer("alice"))
        alice_status = client.get("/v1/status", headers=_bearer("alice"))
        admin_status = client.get("/v1/status", headers=_bearer("admin"))
        admin_merge = client.post(
            "/v1/tags/merge",
            json={"from": "confidencial", "into": "secreto"},
            headers=_bearer("admin"),
        )

    assert alice_merge.status_code == 403
    assert alice_confirm.status_code == 403
    assert alice_status.status_code == 403
    assert admin_status.status_code == 200
    assert admin_merge.status_code == 200


# ---------------------------------------------------------------------------
# 9. Authentication
# ---------------------------------------------------------------------------


def test_requests_without_a_valid_token_are_unauthorized(seeded_env: AclEnv) -> None:
    """Missing and invalid bearer tokens are rejected with 401."""
    with _client() as client:
        missing = client.post("/v1/search", json={"query": _SECRET_QUESTION})
        invalid = client.post(
            "/v1/search",
            json={"query": _SECRET_QUESTION},
            headers={"Authorization": "Bearer not-a-jwt"},
            follow_redirects=False,
        )

    assert missing.status_code == 401
    assert invalid.status_code == 401


def test_group_grant_resolves_through_the_real_table(seeded_env: AclEnv) -> None:
    """alice's prefixes come from the user and group grants in acl_grant."""
    env = seeded_env
    alice = _principal(env, "alice", ["equipe-alfa"])
    assert alice.acl_prefixes == ["teamA"]
    assert alice.can_open_files is False
    assert alice.is_admin is False

    admin = _principal(env, "admin")
    assert admin.acl_prefixes == [""]
    assert admin.is_admin is True


# LOCAL keeps meaning "no restriction" for the MCP/stdio path.
def test_local_principal_still_sees_everything(seeded_env: AclEnv) -> None:
    """The LOCAL principal is unaffected by team-mode grants."""
    env = seeded_env
    assert LOCAL.acl_prefixes is None
    result = retrieve(_SECRET_QUESTION)
    assert any(hit.path == _SECRET_PATH for hit in result.hits)
    del env
