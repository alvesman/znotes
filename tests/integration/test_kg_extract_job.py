"""Integration tests for the ``kg_extract`` job and the graph CLI (task 40).

The setup mirrors ``test_autotag_job.py`` but needs only Postgres: a temporary
source root is ingested with a ``FakeEmbedder`` (dim 8) and the ``kg_extract``
handler runs with a ``FakeChat`` returning fixed JSON, so no model endpoint and
no OpenSearch is touched. The notes have one section per heading with ~50 words
each (>= 40 tokens under the test whitespace counter), so every section becomes
one extractable group.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from typer.testing import CliRunner

from fakes import FakeChat, FakeEmbedder
from mynotes.cli import app
from mynotes.config import Settings
from mynotes.db.models import (
    Document,
    EmbeddingCache,
    File,
    IndexOutbox,
    Job,
    KgExtractionCache,
    ScanRun,
    SourceRoot,
    Tag,
    TagAlias,
    Workspace,
)
from mynotes.db.session import reset_engine, session_scope
from mynotes.graph.extract import (
    cached_group_hashes,
    clear_error_cache,
    extractable_groups,
    split_group,
    versions_missing_extraction,
)
from mynotes.graph.ontology import EXTRACTOR_VERSION
from mynotes.jobs import pipeline
from mynotes.jobs.registry import JobContext, Priority
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
)

_NOTE_3 = """---
title: Plano Alfa
date: 2026-09-01
---

# Contexto

O projeto Alfa começou em janeiro de 2026 com o objetivo de substituir a
planilha de controle por um sistema próprio. A Ana ficou responsável pela parte
de dados e o Bruno pela integração com o sistema legado. O orçamento inicial
aprovado foi de vinte mil euros até o fim de dezembro.

# Decisões

Em março a equipe decidiu usar PostgreSQL como banco principal e adiar a
migração do relatório fiscal para o segundo semestre. A decisão foi registrada
na ata da reunião e o Bruno ficou de preparar o plano de migração detalhado
antes do fim de abril deste ano.

# Riscos

O principal risco do projeto Alfa é a dependência do fornecedor externo, que já
atrasou duas entregas seguidas. Se o contrato não for renovado até junho, o
marco de lançamento de setembro precisa ser revisto e a equipe deve apresentar
um plano alternativo ao comitê.
"""

_NOTE_3_EDITED = _NOTE_3.replace(
    "Em março a equipe decidiu usar PostgreSQL como banco principal",
    "Em abril a equipe decidiu manter PostgreSQL e o orçamento revisado subiu",
)

_NOTE_1 = """---
title: Ideia Listas
date: 2026-09-02
---

# Notas

Esta nota registra a ideia de um aplicativo de listas compartilhadas para a
família. A ideia é simples: cada pessoa marca os itens que faltam e o
aplicativo avisa quem está no supermercado naquele momento. A primeira versão
deve sair antes das férias de julho.
"""

_PLAN_PATH = "projects/alpha/plano.md"


@dataclass
class KgEnv:
    """The registered test root, the fake embedder and the test's settings."""

    settings: Settings
    fake: FakeEmbedder
    root_id: int
    workspace_id: int
    root_path: Path


@pytest.fixture
def env(migrated_dsn: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[KgEnv]:
    """Container settings (dim 8, fast settle, grace 0) and a disposable workspace."""
    settings = Settings(postgres={"dsn": migrated_dsn})
    fake = FakeEmbedder(dim=8)
    settings.models.embedding.dim = 8
    settings.models.embedding.model = fake.model
    settings.scan.settle_seconds = 0
    settings.scan.delete_grace_seconds = 0
    monkeypatch.setattr("mynotes.config.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.db.session.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.scan.scanner.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.jobs.worker.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.llm.limits.get_settings", lambda: settings)
    reset_engine()
    with session_scope() as session:
        workspace = Workspace(name=f"kg-{uuid4().hex[:8]}")
        session.add(workspace)
        session.flush()
        root = SourceRoot(workspace_id=workspace.id, name="root", abs_path=str(tmp_path))
        session.add(root)
        session.flush()
        settings.workspace = workspace.name
        state = KgEnv(
            settings=settings,
            fake=fake,
            root_id=root.id,
            workspace_id=workspace.id,
            root_path=tmp_path,
        )
    yield state
    with session_scope() as session:
        session.execute(delete(TagAlias).where(TagAlias.workspace_id == state.workspace_id))
        session.execute(delete(Tag).where(Tag.workspace_id == state.workspace_id))
        session.execute(delete(Job).where(Job.type.in_(_JOB_TYPES)))
        session.execute(delete(IndexOutbox).where(IndexOutbox.doc_kind == "chunk"))
        session.execute(
            delete(KgExtractionCache).where(
                KgExtractionCache.extractor_version == EXTRACTOR_VERSION
            )
        )
        session.execute(delete(EmbeddingCache).where(EmbeddingCache.model == state.fake.model))
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


def document_and_version(env: KgEnv, rel_path: str) -> tuple[int, int]:
    """The document and latest version ids currently stored for ``rel_path``."""
    with session_scope() as session:
        file_row = session.execute(
            select(File).where(File.source_root_id == env.root_id, File.rel_path == rel_path)
        ).scalar_one()
        assert file_row.document_id is not None
        document = session.get(Document, file_row.document_id)
        assert document is not None and document.latest_version_id is not None
        return document.id, document.latest_version_id


def run_embed(env: KgEnv, version_id: int) -> None:
    """Run the embed handler with the fake embedder, like the worker would."""
    pipeline.handle_embed(
        {"version_id": version_id},
        JobContext(worker_id="test", settings=env.settings, embedder=env.fake),
    )


def ingest(env: KgEnv, rel_path: str, content: str, mtime_ns: int = _MTIME) -> tuple[int, int]:
    """Write a file, scan, drain extract/chunk and embed its latest version."""
    write_file(env.root_path / rel_path, content, mtime_ns)
    scan_root(env.root_id)
    drain_extract_chunk()
    document_id, version_id = document_and_version(env, rel_path)
    run_embed(env, version_id)
    return document_id, version_id


def run_kg_extract(env: KgEnv, version_id: int) -> None:
    """Run the kg_extract handler directly, like the worker would."""
    pipeline.handle_kg_extract(
        {"version_id": version_id},
        JobContext(worker_id="test", settings=env.settings, embedder=env.fake),
    )


def patch_chat(monkeypatch: pytest.MonkeyPatch, chat: FakeChat) -> None:
    """Point the pipeline's ``get_chat`` at the fake."""
    monkeypatch.setattr("mynotes.jobs.pipeline.get_chat", lambda purpose="extract": chat)


def extraction_json(name: str, entity_type: str = "Project") -> str:
    """A fixed extraction answer naming one entity and one self-relation."""
    return json.dumps(
        {
            "entities": [{"name": name, "type": entity_type, "description": "d", "aliases": []}],
            "relations": [
                {
                    "source": name,
                    "target": name,
                    "type": "related_to",
                    "raw_label": "",
                    "description": "",
                    "attribute": None,
                    "value": None,
                    "quote": "q",
                    "confidence": 0.9,
                }
            ],
        }
    )


def test_three_sections_call_llm_per_group_and_enqueue_resolve(
    env: KgEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One call per section group, one cache row each, one resolve job."""
    _document_id, version_id = ingest(env, _PLAN_PATH, _NOTE_3)
    chat = FakeChat(
        [
            extraction_json("Entidade Contexto"),
            extraction_json("Entidade Decisões"),
            extraction_json("Entidade Riscos"),
        ]
    )
    patch_chat(monkeypatch, chat)

    run_kg_extract(env, version_id)

    assert len(chat.calls) == 3, "one LLM call per section group"
    second_user = chat.calls[1][-1].content
    assert "Section: Decisões" in second_user
    assert "Entidade Contexto (Project)" in second_user, (
        "entities of earlier groups are carried forward"
    )
    with session_scope() as session:
        rows = (
            session.execute(
                select(KgExtractionCache).where(
                    KgExtractionCache.extractor_version == EXTRACTOR_VERSION
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 3
        assert all(row.model == "unknown" and "entities" in row.result for row in rows)
        hashes = {row.content_hash for row in rows}
        resolve_jobs = (
            session.execute(
                select(Job).where(
                    Job.type == "resolve_entities",
                    Job.dedupe_key == f"resolve_entities:{version_id}",
                )
            )
            .scalars()
            .all()
        )
        assert len(resolve_jobs) == 1
        job = resolve_jobs[0]
        assert job.status == "queued"
        assert job.priority == int(Priority.RESOLVE_ENTITIES)
        assert job.payload["version_id"] == version_id
        groups = job.payload["groups"]
    assert len(groups) == 3
    assert {entry["content_hash"] for entry in groups} == hashes
    for entry in groups:
        assert entry["chunk_ids"] and all(
            isinstance(chunk_id, str) for chunk_id in entry["chunk_ids"]
        )


def test_rerun_serves_every_group_from_cache(env: KgEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second run of the same version makes zero new LLM calls."""
    _document_id, version_id = ingest(env, _PLAN_PATH, _NOTE_3)
    chat = FakeChat(
        [
            extraction_json("Entidade Contexto"),
            extraction_json("Entidade Decisões"),
            extraction_json("Entidade Riscos"),
        ]
    )
    patch_chat(monkeypatch, chat)
    run_kg_extract(env, version_id)
    assert len(chat.calls) == 3

    run_kg_extract(env, version_id)

    assert len(chat.calls) == 3, "the cache must answer every group"
    with session_scope() as session:
        resolve_jobs = (
            session.execute(
                select(Job).where(
                    Job.type == "resolve_entities",
                    Job.dedupe_key == f"resolve_entities:{version_id}",
                )
            )
            .scalars()
            .all()
        )
    assert len(resolve_jobs) == 1, "the resolve job is not duplicated"


def test_edited_section_only_calls_llm_for_the_changed_group(
    env: KgEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new version re-extracts just the group whose text changed."""
    _document_id, first_version = ingest(env, _PLAN_PATH, _NOTE_3)
    chat = FakeChat(
        [
            extraction_json("Entidade Contexto"),
            extraction_json("Entidade Decisões"),
            extraction_json("Entidade Riscos"),
            extraction_json("Entidade Decisões Nova"),
        ]
    )
    patch_chat(monkeypatch, chat)
    run_kg_extract(env, first_version)
    assert len(chat.calls) == 3

    _document_id2, second_version = ingest(env, _PLAN_PATH, _NOTE_3_EDITED, _MTIME + 1)
    assert second_version != first_version
    run_kg_extract(env, second_version)

    assert len(chat.calls) == 4, "only the edited section group calls the LLM"
    fourth_user = chat.calls[3][-1].content
    assert "orçamento revisado subiu" in fourth_user
    assert "Section: Decisões" in fourth_user


def test_invalid_json_is_cached_as_error_and_never_retried(
    env: KgEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid answers for the group and all its parts end the job with cached errors."""
    _document_id, version_id = ingest(env, "notes/ideia.md", _NOTE_1)
    chat = FakeChat(lambda messages: "definitely not json")
    patch_chat(monkeypatch, chat)

    run_kg_extract(env, version_id)

    calls = len(chat.calls)
    assert calls >= 2, "initial call plus the single repair retry, then the parts"
    with session_scope() as session:
        rows = (
            session.execute(
                select(KgExtractionCache).where(
                    KgExtractionCache.extractor_version == EXTRACTOR_VERSION
                )
            )
            .scalars()
            .all()
        )
        assert rows, "the group and its parts are cached"
        assert all(list(row.result) == ["error"] and row.result["error"] for row in rows)
    run_kg_extract(env, version_id)
    assert len(chat.calls) == calls, "the cached error extraction prevents any retry loop"

    with session_scope() as session:
        assert version_id not in versions_missing_extraction(session)
        session.add(
            KgExtractionCache(
                content_hash=f"partial-{uuid4().hex}",
                extractor_version=EXTRACTOR_VERSION,
                model="fake",
                result={"entities": [], "relations": [], "partial": True},
            )
        )
        session.flush()
        assert clear_error_cache(session) == len(rows) + 1, "errors and partial results"
        assert version_id in versions_missing_extraction(session), "backfill queues it again"


_NOTE_SPLIT = """# Partes

A Alpha é o projeto principal desta nota e o texto desta parte fala apenas dele, com
palavras suficientes para que o grupo passe do mínimo de tokens que o extrator exige.

A Beta é a ferramenta que faz o modelo falhar nesta parte, de novo com palavras
suficientes para encher a linha e dar ao grupo um tamanho razoável para o teste.

A Gama também faz o modelo falhar, numa só linha para que esta parte não se divida mais.
"""


def test_retry_errors_forgets_groups_split_before_the_marker(
    env: KgEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial group without the marker is found through its parts and forgotten."""
    _document_id, version_id = ingest(env, "notes/partes.md", _NOTE_SPLIT)

    def answer(messages: list) -> str:
        text = messages[1].content.split("Text:\n", 1)[1]
        if "Beta" in text or "Gama" in text:
            return "not json"
        return extraction_json("Alpha")

    patch_chat(monkeypatch, FakeChat(answer))
    run_kg_extract(env, version_id)

    with session_scope() as session:
        (top,) = extractable_groups(session, version_id)
        row = session.get(KgExtractionCache, (top.content_hash, EXTRACTOR_VERSION))
        assert row is not None and row.result.get("partial") is True
        assert [entity["name"] for entity in row.result["entities"]] == ["Alpha"]
        # Strip the marker, as rows written before it existed look.
        row.result = {key: value for key, value in row.result.items() if key != "partial"}
        session.flush()
        assert version_id not in versions_missing_extraction(session)

        deleted = clear_error_cache(session)

        assert deleted >= 2, "the error rows of the failed parts plus the group itself"
        assert session.get(KgExtractionCache, (top.content_hash, EXTRACTOR_VERSION)) is None
        assert version_id in versions_missing_extraction(session)
        halves = split_group(top)
        parts = [*halves, *(part for half in halves for part in split_group(half))]
        assert cached_group_hashes(session, [part.content_hash for part in parts]), (
            "the part that worked stays cached"
        )


def test_extract_status_reports_counts(env: KgEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    """``graph extract-status --json`` totals the extractable groups and queue."""
    _document_id, version_id = ingest(env, _PLAN_PATH, _NOTE_3)
    chat = FakeChat(
        [
            extraction_json("Entidade Contexto"),
            extraction_json("Entidade Decisões"),
            extraction_json("Entidade Riscos"),
        ]
    )
    patch_chat(monkeypatch, chat)
    run_kg_extract(env, version_id)
    ingest(env, "notes/ideia.md", _NOTE_1)

    result = CliRunner().invoke(app, ["graph", "extract-status", "--json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["total"] == 4, "three plan groups plus one idea group"
    assert data["cached"] == 3
    assert data["pending"] == 1
    assert data["queued_kg_extract_jobs"] == 2, "the follow-up rows of both versions"
    assert data["llm_p50_ms"] is None, "no real LLM call was observed in this process"
