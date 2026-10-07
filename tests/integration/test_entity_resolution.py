"""Integration tests for entity resolution, merges and the entities index (task 41).

The setup mirrors ``test_kg_extract_job.py``: a temporary source root is
ingested with a ``FakeEmbedder`` (dim 8), ``kg_extract`` runs with a ``FakeChat``
returning fixed extraction JSON, and ``resolve_entities``/``adjudicate_entities``
run as the worker would. Only the index test needs OpenSearch; resolution
degrades gracefully without it (the k-NN signal becomes None), which also keeps
the borderline case deterministic: it is decided by the trigram signal alone.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session
from testcontainers.community.opensearch import OpenSearchContainer
from typer.testing import CliRunner

from fakes import FakeChat, FakeEmbedder
from mynotes.cli import app
from mynotes.config import Settings
from mynotes.db.models import (
    Chunk,
    Document,
    DocumentVersion,
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
from mynotes.index.opensearch import ensure_indices
from mynotes.index.sync import sync_once
from mynotes.jobs import pipeline
from mynotes.jobs.registry import JobContext
from mynotes.scan.scanner import scan_root

pytestmark = pytest.mark.integration

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
)

# Targets that bind get_settings at import time and must see the test settings.
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
)

_ANA = {"name": "Ana", "type": "Person", "description": "gestora de dados", "aliases": []}


def _relation(source: str, target: str, quote: str) -> dict:
    return {
        "source": source,
        "target": target,
        "type": "works_on",
        "raw_label": "ficou responsavel",
        "description": "",
        "attribute": None,
        "value": None,
        "quote": quote,
        "confidence": 0.9,
    }


def _extraction(entities: list[dict], relations: list[dict]) -> str:
    return json.dumps({"entities": entities, "relations": relations})


_ALFA = {
    "name": "Projeto Alfa",
    "type": "Project",
    "description": "controle de acesso",
    "aliases": [],
}
_RELATORIOS = {
    "name": "Sistema Relatorios",
    "type": "Project",
    "description": "controle de metas",
    "aliases": [],
}
_RELATORIO = {
    "name": "Sistema Relatorio",
    "type": "Project",
    "description": "controle de metas",
    "aliases": [],
}

_NOTE_ALFA = """---
title: Plano Alfa
date: 2026-09-01
---

# Contexto

O Projeto Alfa moderniza o controle de acesso do laboratorio com crachas
digitais e auditoria automatica das portas. A Ana coordena as frentes e o
Bruno cuida da integracao com o leitor existente. A primeira fase cobre as
salas de reuniao e o cadastro dos visitantes externos antes do Congresso.
"""

_NOTE_ALFA_LOWER = """---
title: Reuniao Alfa
date: 2026-09-02
---

# Ata

Na reuniao de hoje o projeto alfa avancou para a fase de piloto e o comite
aprovou a compra dos leitores digitais. A Ana apresentou o cronograma revisado
e o Bruno demonstrou a leitura em serie dos crachas de teste. O piloto comeca
na proxima segunda-feira e os resultados serao revisados na reuniao seguinte.
"""

_NOTE_PA = """---
title: Piloto PA
date: 2026-09-03
---

# Status

O PA entrou em piloto nesta semana e os primeiros retornos chegaram das
equipes de limpeza e manutencao do predio sede. A Ana vai consolidar os
apontamentos em um documento unico e o Bruno ajusta a sensibilidade dos
leitores. A meta e fechar a avaliacao antes do fim do mes com uma decisao.
"""

_NOTE_SYSTEM = """---
title: Sistema Relatorios
date: 2026-09-01
---

# Contexto

O Sistema Relatorios substitui a planilha mensal da equipe de dados por um
servico proprio com filas, paineis diarios e exportacao para o time financeiro.
A Ana ficou responsavel pelo desenho dos indicadores e pelo treinamento das
pessoas que vao usar os paineis. O primeiro ciclo de entregas cobre o
inventario de relatorios e a revisao dos numeros com as areas de negocio.
"""

_NOTE_SYSTEM2 = """---
title: Sistema Relatorio
date: 2026-09-02
---

# Acompanhamento

O Sistema Relatorio entrou em testes com a area financeira e as primeiras
filas processaram sem erros criticos nesta semana. A Ana ajustou os indicadores
de fechamento e o time de dados acompanha a exportacao noturna. A revisao final
acontece com as areas antes da liberacao para todas as unidades do grupo.
"""

_ALFA_PATH = "projects/alpha/plano.md"
_ALFA_LOWER_PATH = "projects/alpha/reuniao.md"
_PA_PATH = "projects/alpha/piloto.md"
_SYSTEM_PATH = "projects/system/relatorios.md"
_SYSTEM2_PATH = "projects/system/acompanhamento.md"


@dataclass
class Env:
    """The registered test root, the fake embedder and the test's settings."""

    settings: Settings
    fake: FakeEmbedder
    root_id: int
    workspace_id: int
    root_path: Path


@pytest.fixture
def env(migrated_dsn: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Env]:
    """Container settings (dim 8, fast settle, grace 0) and a disposable workspace."""
    settings = Settings(postgres={"dsn": migrated_dsn})
    fake = FakeEmbedder(dim=8)
    settings.models.embedding.dim = 8
    settings.models.embedding.model = fake.model
    settings.scan.settle_seconds = 0
    settings.scan.delete_grace_seconds = 0
    suffix = uuid4().hex[:8]
    settings.opensearch.chunks_alias = f"t{suffix}_chunks"
    settings.opensearch.entities_alias = f"t{suffix}_entities"
    for target in _PATCH_TARGETS:
        monkeypatch.setattr(target, lambda: settings)
    reset_engine()
    with session_scope() as session:
        workspace = Workspace(name=f"kg-{suffix}")
        session.add(workspace)
        session.flush()
        root = SourceRoot(workspace_id=workspace.id, name="root", abs_path=str(tmp_path))
        session.add(root)
        session.flush()
        settings.workspace = workspace.name
        state = Env(
            settings=settings,
            fake=fake,
            root_id=root.id,
            workspace_id=workspace.id,
            root_path=tmp_path,
        )
    yield state
    with session_scope() as session:
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
        session.execute(delete(EmbeddingCache).where(EmbeddingCache.model == state.fake.model))
        session.execute(delete(TagAlias).where(TagAlias.workspace_id == state.workspace_id))
        session.execute(delete(Tag).where(Tag.workspace_id == state.workspace_id))
        session.execute(delete(File).where(File.source_root_id == state.root_id))
        session.execute(delete(Document).where(Document.source_root_id == state.root_id))
        session.execute(delete(ScanRun).where(ScanRun.source_root_id == state.root_id))
        session.execute(delete(SourceRoot).where(SourceRoot.id == state.root_id))
        session.execute(delete(Workspace).where(Workspace.id == state.workspace_id))
    reset_engine()


def write_file(path: Path, content: str, mtime_ns: int = _MTIME) -> None:
    """Create a file with a fixed mtime so scans are deterministic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))


def drain_extract_chunk() -> None:
    """Run the worker until no extract/chunk job is claimable."""
    from mynotes.jobs.worker import Worker

    worker = Worker(concurrency=1, poll_interval_s=0.05, types=["extract", "chunk"])
    while worker.run_once():
        pass


def document_and_version(env: Env, rel_path: str) -> tuple[int, int]:
    """The document and latest version ids currently stored for ``rel_path``."""
    with session_scope() as session:
        file_row = session.execute(
            select(File).where(File.source_root_id == env.root_id, File.rel_path == rel_path)
        ).scalar_one()
        assert file_row.document_id is not None
        document = session.get(Document, file_row.document_id)
        assert document is not None and document.latest_version_id is not None
        return document.id, document.latest_version_id


def ingest(env: Env, rel_path: str, content: str, mtime_ns: int = _MTIME) -> tuple[int, int]:
    """Write a file, scan, drain extract/chunk and embed its latest version."""
    write_file(env.root_path / rel_path, content, mtime_ns)
    scan_root(env.root_id)
    drain_extract_chunk()
    document_id, version_id = document_and_version(env, rel_path)
    pipeline.handle_embed(
        {"version_id": version_id},
        JobContext(worker_id="test", settings=env.settings, embedder=env.fake),
    )
    return document_id, version_id


def patch_chat(monkeypatch: pytest.MonkeyPatch, chat: FakeChat) -> None:
    """Point the pipeline's ``get_chat`` at the fake."""
    monkeypatch.setattr("mynotes.jobs.pipeline.get_chat", lambda purpose="extract": chat)


def run_handler(env: Env, job_type: str, payload: dict) -> None:
    """Run one pipeline handler directly, like the worker would."""
    context = JobContext(worker_id="test", settings=env.settings, embedder=env.fake)
    getattr(pipeline, f"handle_{job_type}")(payload, context)


def queued_payloads(job_type: str) -> list[dict]:
    """The payloads of the queued jobs of one type, in id order."""
    with session_scope() as session:
        jobs = (
            session.execute(select(Job).where(Job.type == job_type).order_by(Job.id))
            .scalars()
            .all()
        )
        return [dict(job.payload) for job in jobs]


def resolve_job_for(version_id: int) -> dict:
    """The queued resolve_entities payload of one version."""
    for payload in queued_payloads("resolve_entities"):
        if payload["version_id"] == version_id:
            return payload
    raise AssertionError(f"no resolve_entities job for version {version_id}")


def entity_by_name(name: str) -> KgEntity | None:
    """The entity whose canonical name matches exactly, or None."""
    with session_scope() as session:
        entity = session.execute(
            select(KgEntity).where(KgEntity.canonical_name == name)
        ).scalar_one_or_none()
        if entity is None:
            return None
        session.expunge(entity)
        return entity


def mention_documents(entity_id: int) -> set[str]:
    """The rel_paths of the documents currently mentioning one entity."""
    with session_scope() as session:
        return set(
            session.execute(
                select(File.rel_path)
                .select_from(KgMention)
                .join(Chunk, Chunk.id == KgMention.chunk_id)
                .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
                .join(Document, Document.id == DocumentVersion.document_id)
                .join(File, File.id == Document.current_file_id)
                .where(KgMention.entity_id == entity_id)
            ).scalars()
        )


def ingest_and_resolve(
    env: Env,
    monkeypatch: pytest.MonkeyPatch,
    notes: list[tuple[str, str]],
    answers: list[str],
) -> None:
    """Ingest notes, run kg_extract per version and drain the resolve jobs."""
    patch_chat(monkeypatch, FakeChat(answers))
    for index, (rel_path, content) in enumerate(notes):
        _document_id, version_id = ingest(env, rel_path, content, _MTIME + index)
        run_handler(env, "kg_extract", {"version_id": version_id})
    for payload in queued_payloads("resolve_entities"):
        run_handler(env, "resolve_entities", payload)


def ingest_and_merge_borderline(env: Env, monkeypatch: pytest.MonkeyPatch) -> tuple[int, int]:
    """The borderline setup: two near-identical names, adjudicated to one entity.

    Note B's name differs by one letter from note A's: the trigram signal
    passes (0.85 >= 0.6) while the embedding signal is absent or below the
    cosine threshold, so note B's entity is created and adjudicated. The
    scripted verdict merges the newer entity into the older one.

    Returns:
        ``(keep_id, drop_id)``.
    """
    ingest_and_resolve(
        env,
        monkeypatch,
        [(_SYSTEM_PATH, _NOTE_SYSTEM), (_SYSTEM2_PATH, _NOTE_SYSTEM2)],
        [
            _extraction(
                [_RELATORIOS, _ANA],
                [_relation("Ana", "Sistema Relatorios", "A Ana ficou responsavel")],
            ),
            _extraction(
                [_RELATORIO, _ANA],
                [_relation("Ana", "Sistema Relatorio", "O Sistema Relatorio entrou em testes")],
            ),
        ],
    )
    keep = entity_by_name("Sistema Relatorios")
    drop = entity_by_name("Sistema Relatorio")
    assert keep is not None and drop is not None
    assert keep.id < drop.id, "the adjudication keeps the older entity"
    assert queued_payloads("adjudicate_entities") == [{"pairs": [[drop.id, keep.id]]}]
    patch_chat(
        monkeypatch,
        FakeChat(['{"decisions": [{"pair": 0, "same": true, "confidence": 0.95}]}']),
    )
    for payload in queued_payloads("adjudicate_entities"):
        run_handler(env, "adjudicate_entities", payload)
    return keep.id, drop.id


def test_two_notes_merge_into_one_entity_with_mentions_in_both(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test 1: 'Projeto Alfa' and 'projeto alfa' resolve to one entity."""
    ingest_and_resolve(
        env,
        monkeypatch,
        [(_ALFA_PATH, _NOTE_ALFA), (_ALFA_LOWER_PATH, _NOTE_ALFA_LOWER)],
        [
            _extraction(
                [_ALFA, _ANA], [_relation("Ana", "Projeto Alfa", "A Ana coordena as frentes")]
            ),
            _extraction(
                [_ALFA, _ANA],
                [_relation("Ana", "Projeto Alfa", "o projeto alfa avancou para a fase de piloto")],
            ),
        ],
    )
    entity = entity_by_name("Projeto Alfa")
    assert entity is not None
    assert entity.norm_name == "projeto alfa"
    assert entity.status == "active"
    assert mention_documents(entity.id) == {_ALFA_PATH, _ALFA_LOWER_PATH}


def test_alias_row_is_created_and_later_mention_links_by_alias(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test 2: 'PA' given as an alias; a later 'PA' mention links by exact alias."""
    ingest_and_resolve(
        env,
        monkeypatch,
        [(_ALFA_PATH, _NOTE_ALFA), (_PA_PATH, _NOTE_PA)],
        [
            _extraction([{**_ALFA, "aliases": ["PA"]}], []),
            _extraction(
                [
                    {
                        "name": "PA",
                        "type": "Project",
                        "description": "controle de acesso",
                        "aliases": [],
                    }
                ],
                [],
            ),
        ],
    )
    entity = entity_by_name("Projeto Alfa")
    assert entity is not None
    with session_scope() as session:
        aliases = (
            session.execute(select(KgEntityAlias).where(KgEntityAlias.entity_id == entity.id))
            .scalars()
            .all()
        )
        assert [(alias.alias, alias.alias_norm) for alias in aliases] == [("PA", "pa")]
        pa_mention = session.execute(
            select(KgMention)
            .join(DocumentVersion, DocumentVersion.id == KgMention.document_version_id)
            .join(Document, Document.id == DocumentVersion.document_id)
            .join(File, File.id == Document.current_file_id)
            .where(File.rel_path == _PA_PATH)
        ).scalar_one()
        assert pa_mention.entity_id == entity.id
        assert pa_mention.surface_form == "PA"


def test_same_relation_from_two_documents_keeps_two_evidence_rows(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test 3: one (src, dst, type) edge, two evidence rows with their doc_dates."""
    ingest_and_resolve(
        env,
        monkeypatch,
        [(_ALFA_PATH, _NOTE_ALFA), (_ALFA_LOWER_PATH, _NOTE_ALFA_LOWER)],
        [
            _extraction(
                [_ALFA, _ANA], [_relation("Ana", "Projeto Alfa", "A Ana coordena as frentes")]
            ),
            _extraction(
                [_ALFA, _ANA],
                [_relation("Ana", "Projeto Alfa", "o projeto alfa avancou para a fase de piloto")],
            ),
        ],
    )
    with session_scope() as session:
        relations = session.execute(select(KgRelation)).scalars().all()
        assert len(relations) == 1
        relation = relations[0]
        assert relation.type == "works_on"
        assert relation.status == "active"
        evidence = (
            session.execute(
                select(KgRelationEvidence)
                .where(KgRelationEvidence.relation_id == relation.id)
                .order_by(KgRelationEvidence.id)
            )
            .scalars()
            .all()
        )
        assert len(evidence) == 2
        dates = sorted(evidence_item.doc_date for evidence_item in evidence)
        assert [date.date().isoformat() for date in dates] == ["2026-09-01", "2026-09-02"]
        assert all(evidence_item.is_current for evidence_item in evidence)


def test_rerunning_resolve_entities_is_idempotent(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test 4: identical row counts after re-running the same payload."""
    ingest_and_resolve(
        env,
        monkeypatch,
        [(_ALFA_PATH, _NOTE_ALFA)],
        [
            _extraction(
                [{**_ALFA, "aliases": ["PA"]}, _ANA],
                [_relation("Ana", "Projeto Alfa", "A Ana coordena as frentes")],
            )
        ],
    )
    _document_id, version_id = document_and_version(env, _ALFA_PATH)
    payload = resolve_job_for(version_id)

    def counts() -> dict[str, int]:
        with session_scope() as session:
            return {
                "entities": _total(session, KgEntity),
                "aliases": _total(session, KgEntityAlias),
                "mentions": _total(session, KgMention),
                "relations": _total(session, KgRelation),
                "evidence": _total(session, KgRelationEvidence),
                "entity_outbox": int(
                    session.execute(
                        select(func.count())
                        .select_from(IndexOutbox)
                        .where(IndexOutbox.doc_kind == "entity")
                    ).scalar_one()
                ),
            }

    before = counts()
    run_handler(
        env,
        "resolve_entities",
        {"version_id": payload["version_id"], "groups": payload["groups"]},
    )
    assert before == counts()


def _total(session: Session, model: type) -> int:
    return int(session.execute(select(func.count()).select_from(model)).scalar_one())


def test_borderline_pair_adjudicates_and_merges(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 5: borderline pair creates an entity + adjudication; same verdict merges."""
    keep_id, drop_id = ingest_and_merge_borderline(env, monkeypatch)
    with session_scope() as session:
        merged = session.get(KgEntity, drop_id)
        assert merged is not None and merged.status == "merged"
        assert merged.merged_into_id == keep_id
        keep_alias_norms = set(
            session.execute(
                select(KgEntityAlias.alias_norm).where(KgEntityAlias.entity_id == keep_id)
            ).scalars()
        )
        assert "sistema relatorio" in keep_alias_norms
        relations = (
            session.execute(select(KgRelation).where(KgRelation.type == "works_on")).scalars().all()
        )
        assert len(relations) == 1, "the colliding edge was folded into the surviving one"
        assert relations[0].dst_entity_id == keep_id
        evidence = (
            session.execute(
                select(KgRelationEvidence).where(KgRelationEvidence.relation_id == relations[0].id)
            )
            .scalars()
            .all()
        )
        assert len(evidence) == 2, "the evidence moved to the kept entity's relation"
    assert mention_documents(keep_id) == {_SYSTEM_PATH, _SYSTEM2_PATH}


def test_unmerge_restores_both_and_records_distinct_pair(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test 6: unmerge re-splits mentions and re-resolution does not re-merge."""
    from mynotes.graph.store import unmerge_entity

    keep_id, drop_id = ingest_and_merge_borderline(env, monkeypatch)

    with session_scope() as session:
        assert unmerge_entity(session, drop_id) == keep_id

    with session_scope() as session:
        restored = session.get(KgEntity, drop_id)
        assert restored is not None and restored.status == "active"
        assert restored.merged_into_id is None
        pairs = session.execute(select(KgEntityDistinct)).scalars().all()
        assert len(pairs) == 1 and (pairs[0].a_id, pairs[0].b_id) == (keep_id, drop_id)
        keep_alias_norms = set(
            session.execute(
                select(KgEntityAlias.alias_norm).where(KgEntityAlias.entity_id == keep_id)
            ).scalars()
        )
        assert "sistema relatorio" not in keep_alias_norms

    adjudication_jobs = len(queued_payloads("adjudicate_entities"))
    for payload in queued_payloads("resolve_entities"):
        run_handler(env, "resolve_entities", payload)

    assert session_get_status(drop_id) == "active"
    assert session_get_status(keep_id) == "active"
    assert mention_documents(keep_id) == {_SYSTEM_PATH}
    assert mention_documents(drop_id) == {_SYSTEM2_PATH}
    assert len(queued_payloads("adjudicate_entities")) == adjudication_jobs, (
        "the distinct pair must not re-trigger adjudication"
    )


def session_get_status(entity_id: int) -> str:
    """The current status of one entity."""
    with session_scope() as session:
        entity = session.get(KgEntity, entity_id)
        assert entity is not None
        return str(entity.status)


def test_graph_cli_lists_and_merges_entities(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """The graph entity/entities/merge/unmerge commands drive the store."""
    from mynotes.graph.store import unmerge_entity

    ingest_and_resolve(
        env,
        monkeypatch,
        [(_SYSTEM_PATH, _NOTE_SYSTEM), (_SYSTEM2_PATH, _NOTE_SYSTEM2)],
        [
            _extraction([_RELATORIOS, _ANA], []),
            _extraction([_RELATORIO, _ANA], []),
        ],
    )
    keep = entity_by_name("Sistema Relatorios")
    drop = entity_by_name("Sistema Relatorio")
    assert keep is not None and drop is not None
    runner = CliRunner()

    listed = runner.invoke(app, ["graph", "entities"])
    assert listed.exit_code == 0
    assert "Sistema Relatorios" in listed.output and "Ana" in listed.output

    card = runner.invoke(app, ["graph", "entity", "Sistema Relatorios"])
    assert card.exit_code == 0
    assert "Sistema Relatorios" in card.output

    typed = runner.invoke(app, ["graph", "entities", "--type", "Person"])
    assert typed.exit_code == 0
    assert "Ana" in typed.output and "Sistema Relatorios" not in typed.output

    merged = runner.invoke(app, ["graph", "merge", "Sistema Relatorios", "Sistema Relatorio"])
    assert merged.exit_code == 0, merged.output
    assert session_get_status(drop.id) == "merged"

    with session_scope() as session:
        assert unmerge_entity(session, drop.id) == keep.id
    unmerged = runner.invoke(app, ["graph", "unmerge", "Sistema Relatorio"])
    assert unmerged.exit_code == 1, "an active entity cannot be unmerged again"

    ambiguous = runner.invoke(app, ["graph", "entity", "Sistema Relatorio"])
    assert ambiguous.exit_code == 0, "the restored entity resolves again"


def test_entities_index_tracks_active_entities_and_drops_merged(
    env: Env, monkeypatch: pytest.MonkeyPatch, os_container: OpenSearchContainer
) -> None:
    """Test 7: after sync_once the index holds the active entities; merges remove."""
    from opensearchpy import OpenSearch

    config = os_container.get_config()
    env.settings.opensearch.url = f"http://{config['host']}:{config['port']}"
    client = OpenSearch(hosts=[env.settings.opensearch.url], use_ssl=False)
    ensure_indices(client, env.settings)

    ingest_and_resolve(
        env,
        monkeypatch,
        [(_SYSTEM_PATH, _NOTE_SYSTEM)],
        [_extraction([{**_RELATORIOS, "aliases": ["SR"]}, _ANA], [])],
    )
    stats = sync_once()
    assert stats.upserts >= 1
    alias = env.settings.opensearch.entities_alias
    client.indices.refresh(index=alias)
    body = {"query": {"match_all": {}}, "size": 20, "_source": {"excludes": ["embedding"]}}
    by_name = {
        hit["_source"]["canonical_name"]: hit["_source"]
        for hit in client.search(index=alias, body=body)["hits"]["hits"]
    }
    assert set(by_name) == {"Sistema Relatorios", "Ana"}
    project = by_name["Sistema Relatorios"]
    assert project["aliases"] == ["SR"]
    assert project["type"] == "Project"
    assert project["status"] == "active"
    assert project["description"] == "controle de metas"
    assert project["workspace_id"] == str(env.workspace_id)

    keep = entity_by_name("Sistema Relatorios")
    ana = entity_by_name("Ana")
    assert keep is not None and ana is not None
    from mynotes.graph.store import merge_entities

    with session_scope() as session:
        merge_entities(session, keep_id=keep.id, drop_id=ana.id)
    sync_once()
    client.indices.refresh(index=alias)
    names = {
        hit["_source"]["canonical_name"]
        for hit in client.search(index=alias, body=body)["hits"]["hits"]
    }
    assert names == {"Sistema Relatorios"}


def test_concurrent_creation_of_the_same_entity_links_to_the_winner(migrated_dsn: str) -> None:
    """Two resolution jobs creating the same new entity at once end up with one row.

    Session A inserts and stays uncommitted; session B's insert waits on the
    partial unique index (migration 0004) and, once A commits, links to A's
    row instead of raising UniqueViolation. Before the fix both inserted.
    """
    from sqlalchemy import create_engine as sa_create_engine
    from sqlalchemy import delete, select
    from sqlalchemy.orm import Session as SASession

    from mynotes.db.models import KgEntity, Workspace
    from mynotes.graph.extract import ExtractedEntity
    from mynotes.graph.resolve import create_entity, entity_norm

    extracted = ExtractedEntity(name="Lattice", type="Technology", description="a tool")
    norm = entity_norm(extracted.name, extracted.type)
    engine = sa_create_engine(migrated_dsn)
    try:
        with SASession(bind=engine.connect()) as bootstrap:
            workspace = Workspace(name="entity-race")
            bootstrap.add(workspace)
            bootstrap.commit()
            workspace_id = int(workspace.id)

        session_a = SASession(bind=engine.connect())
        id_a, created_a = create_entity(session_a, workspace_id, extracted, norm)
        assert created_a

        result: list[tuple[int, bool]] = []

        def worker_b() -> None:
            session_b = SASession(bind=engine.connect())
            try:
                result.append(create_entity(session_b, workspace_id, extracted, norm))
                session_b.commit()
            finally:
                session_b.close()

        thread = threading.Thread(target=worker_b, name="worker-b")
        thread.start()
        time.sleep(0.2)
        session_a.commit()
        thread.join(timeout=10)
        assert not thread.is_alive(), "worker B stayed blocked on worker A's insert"
        assert result == [(id_a, False)]

        with SASession(bind=engine.connect()) as verify:
            rows = (
                verify.execute(
                    select(KgEntity.id).where(
                        KgEntity.workspace_id == workspace_id, KgEntity.norm_name == norm
                    )
                )
                .scalars()
                .all()
            )
            assert rows == [id_a]
            # A merged twin is allowed (the index is partial on status = 'active').
            verify.add(
                KgEntity(
                    workspace_id=workspace_id,
                    type="Technology",
                    canonical_name="Lattice",
                    norm_name=norm,
                    status="merged",
                    merged_into_id=id_a,
                    summary_stale=True,
                )
            )
            verify.commit()
        session_a.close()
        with SASession(bind=engine.connect()) as cleanup:
            cleanup.execute(delete(KgEntity).where(KgEntity.workspace_id == workspace_id))
            cleanup.execute(delete(Workspace).where(Workspace.id == workspace_id))
            cleanup.commit()
    finally:
        engine.dispose()


_STANZA = {
    "name": "Stanza",
    "type": "Technology",
    "description": "biblioteca de PLN",
    "aliases": [],
}
_TUA = {"name": "Projeto Tua", "type": "Project", "description": "pipeline de texto", "aliases": []}
_FERRAMENTAS = {
    "name": "ferramentas externas",
    "type": "Concept",
    "description": "dependencias",
    "aliases": [],
}

_NOTE_STANZA = """---
title: Stanza
date: 2026-09-01
---

# Stanza

A Stanza e uma biblioteca de processamento de linguagem natural mantida pelo
grupo de PLN de Stanford, publicada sob a licenca Apache 2.0, com modelos para
mais de sessenta linguas e uma interface simples em Python para anotacao.
"""

_NOTE_TUA = """---
title: Projeto Tua
date: 2026-09-02
---

# Projeto Tua

O Projeto Tua assenta na ferramenta Stanza para a anotacao morfologica dos
textos e usa ferramentas externas para a conversao dos ficheiros de entrada.
A equipa revê as etiquetas por amostragem antes de cada entrega mensal.
"""

_STANZA_PATH = "tools/stanza.md"
_TUA_PATH = "projects/tua.md"


def _mentions_in(rel_path: str, entity_id: int) -> list[float]:
    """The confidences of one entity's mentions in one document."""
    with session_scope() as session:
        return list(
            session.execute(
                select(KgMention.confidence)
                .join(Chunk, Chunk.id == KgMention.chunk_id)
                .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
                .join(Document, Document.id == DocumentVersion.document_id)
                .join(File, File.id == Document.current_file_id)
                .where(File.rel_path == rel_path, KgMention.entity_id == entity_id)
            ).scalars()
        )


def test_lexical_pass_links_a_known_entity_the_extractor_missed(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tool note is resolved first; the project note names the tool verbatim but
    its extraction yields only "ferramentas externas": the lexical pass adds the mention."""
    from mynotes.graph.lexical import LEXICAL_CONFIDENCE

    ingest_and_resolve(
        env,
        monkeypatch,
        [(_STANZA_PATH, _NOTE_STANZA), (_TUA_PATH, _NOTE_TUA)],
        [_extraction([_STANZA], []), _extraction([_TUA, _FERRAMENTAS], [])],
    )
    stanza = entity_by_name("Stanza")
    assert stanza is not None
    assert mention_documents(stanza.id) == {_STANZA_PATH, _TUA_PATH}
    assert _mentions_in(_STANZA_PATH, stanza.id) == [1.0]
    assert _mentions_in(_TUA_PATH, stanza.id) == [pytest.approx(LEXICAL_CONFIDENCE)]
    # Concepts are never linked lexically, and the pass creates no entity for "ferramenta".
    with session_scope() as session:
        names = set(
            session.execute(
                select(KgEntity.canonical_name).where(KgEntity.workspace_id == env.workspace_id)
            ).scalars()
        )
    assert names == {"Stanza", "Projeto Tua", "ferramentas externas"}
    with session_scope() as session:
        count = session.execute(
            select(KgEntity.mention_count_current).where(KgEntity.id == stanza.id)
        ).scalar_one()
    assert count == 1, "lexical mentions are evidence, not salience"


def test_link_mentions_cli_covers_the_other_direction_and_is_idempotent(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The project note is resolved before the tool entity exists; ``graph
    link-mentions`` adds the mention afterwards, and a second run changes nothing."""
    from mynotes.graph.lexical import LEXICAL_CONFIDENCE

    ingest_and_resolve(
        env,
        monkeypatch,
        [(_TUA_PATH, _NOTE_TUA), (_STANZA_PATH, _NOTE_STANZA)],
        [_extraction([_TUA, _FERRAMENTAS], []), _extraction([_STANZA], [])],
    )
    stanza = entity_by_name("Stanza")
    assert stanza is not None
    assert mention_documents(stanza.id) == {_STANZA_PATH}
    monkeypatch.setattr("mynotes.cli.get_settings", lambda: env.settings, raising=False)
    runner = CliRunner()
    dry = runner.invoke(app, ["graph", "link-mentions", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "would add 1 lexical mention" in dry.output
    assert mention_documents(stanza.id) == {_STANZA_PATH}
    first = runner.invoke(app, ["graph", "link-mentions"])
    assert first.exit_code == 0, first.output
    assert "removed 0 and added 1 lexical mention" in first.output
    assert _mentions_in(_TUA_PATH, stanza.id) == [pytest.approx(LEXICAL_CONFIDENCE)]
    second = runner.invoke(app, ["graph", "link-mentions"])
    assert second.exit_code == 0, second.output
    assert "removed 1 and added 1 lexical mention" in second.output
    assert _mentions_in(_TUA_PATH, stanza.id) == [pytest.approx(LEXICAL_CONFIDENCE)]
    with session_scope() as session:
        count = session.execute(
            select(KgEntity.mention_count_current).where(KgEntity.id == stanza.id)
        ).scalar_one()
    assert count == 1, "lexical mentions are evidence, not salience"


_UPDATE_OS = {"name": "Update OS", "type": "Project", "description": "manutencao", "aliases": []}
_DATE_ENTITY = {"name": "2024-10-15", "type": "Milestone", "description": "", "aliases": []}
_NOTE_OS = """---
title: Manutencao
date: 2026-09-05
---

# Plano

O Update OS esta agendado para 2024-10-15, depois da janela de manutencao
acordada com a equipa de operacoes e com os responsaveis pelos servicos.
A equipa prepara os backups na semana anterior, valida o plano de reversao
com os donos das aplicacoes e comunica a paragem aos utilizadores com uma
semana de antecedencia, como ficou combinado na ultima reuniao de planeamento.
"""
_OS_PATH = "ops/update-os.md"


def test_resolution_folds_bare_dates_into_attribute_values(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A date entity in the extraction becomes a date attribute of its source."""
    ingest_and_resolve(
        env,
        monkeypatch,
        [(_OS_PATH, _NOTE_OS)],
        [
            _extraction(
                [_UPDATE_OS, _DATE_ENTITY],
                [
                    {
                        "source": "Update OS",
                        "target": "2024-10-15",
                        "type": "scheduled_for",
                        "raw_label": "agendado para",
                        "description": "",
                        "attribute": None,
                        "value": None,
                        "quote": "agendado para 2024-10-15",
                        "confidence": 0.9,
                    }
                ],
            )
        ],
    )
    assert entity_by_name("2024-10-15") is None
    update_os = entity_by_name("Update OS")
    assert update_os is not None
    with session_scope() as session:
        rows = session.execute(
            select(
                KgRelation.src_entity_id,
                KgRelation.dst_entity_id,
                KgRelation.type,
                KgRelation.attribute,
                KgRelation.value,
            ).where(KgRelation.workspace_id == env.workspace_id)
        ).all()
    assert [tuple(row) for row in rows] == [
        (update_os.id, update_os.id, "scheduled_for", "date", "2024-10-15")
    ]


def test_fold_dates_cli_cleans_existing_date_entities(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An older graph with a date entity: ``graph fold-dates`` moves its relation
    evidence to the source's attribute and deletes the entity; hubs lists degrees."""
    ingest_and_resolve(env, monkeypatch, [(_OS_PATH, _NOTE_OS)], [_extraction([_UPDATE_OS], [])])
    update_os = entity_by_name("Update OS")
    assert update_os is not None
    _document_id, version_id = document_and_version(env, _OS_PATH)
    with session_scope() as session:
        date_entity = KgEntity(
            workspace_id=env.workspace_id,
            type="Milestone",
            canonical_name="2024-10-15",
            norm_name="2024-10-15",
            status="active",
        )
        session.add(date_entity)
        session.flush()
        chunk_id = (
            session.execute(
                select(Chunk.id)
                .where(Chunk.document_version_id == version_id)
                .order_by(Chunk.ordinal)
            )
            .scalars()
            .first()
        )
        session.add(
            KgMention(
                entity_id=date_entity.id,
                chunk_id=chunk_id,
                document_version_id=version_id,
                surface_form="2024-10-15",
            )
        )
        relation = KgRelation(
            workspace_id=env.workspace_id,
            src_entity_id=update_os.id,
            dst_entity_id=date_entity.id,
            type="scheduled_for",
        )
        session.add(relation)
        session.flush()
        session.add(
            KgRelationEvidence(
                relation_id=relation.id,
                chunk_id=chunk_id,
                document_version_id=version_id,
                quote="agendado para 2024-10-15",
                confidence=0.9,
                is_current=True,
            )
        )
        date_id = int(date_entity.id)
        ghost = KgEntity(
            workspace_id=env.workspace_id,
            type="Milestone",
            canonical_name="15/10/2024",
            norm_name="15/10/2024",
            status="merged",
            merged_into_id=date_id,
        )
        session.add(ghost)
    monkeypatch.setattr("mynotes.cli.get_settings", lambda: env.settings, raising=False)
    runner = CliRunner()
    hubs = runner.invoke(app, ["graph", "hubs", "--limit", "5"])
    assert hubs.exit_code == 0, hubs.output
    assert "2024-10-15" in hubs.output
    dry = runner.invoke(app, ["graph", "fold-dates", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "would fold 1 date entity: 1 relation(s)" in dry.output
    assert entity_by_name("2024-10-15") is not None
    real = runner.invoke(app, ["graph", "fold-dates"])
    assert real.exit_code == 0, real.output
    assert "folded 1 date entity" in real.output
    assert entity_by_name("2024-10-15") is None
    assert entity_by_name("15/10/2024") is None, "merged ghosts of the date go with it"
    with session_scope() as session:
        rows = session.execute(
            select(
                KgRelation.src_entity_id,
                KgRelation.dst_entity_id,
                KgRelation.attribute,
                KgRelation.value,
            ).where(KgRelation.workspace_id == env.workspace_id)
        ).all()
        evidence = session.execute(
            select(func.count())
            .select_from(KgRelationEvidence)
            .join(KgRelation, KgRelation.id == KgRelationEvidence.relation_id)
            .where(KgRelation.workspace_id == env.workspace_id)
        ).scalar_one()
        deletes = session.execute(
            select(func.count())
            .select_from(IndexOutbox)
            .where(IndexOutbox.op == "delete", IndexOutbox.doc_id == str(date_id))
        ).scalar_one()
    assert [tuple(r) for r in rows] == [(update_os.id, update_os.id, "date", "2024-10-15")]
    assert evidence == 1
    assert deletes >= 1
    again = runner.invoke(app, ["graph", "fold-dates"])
    assert again.exit_code == 0 and "folded 0 date entities" in again.output
