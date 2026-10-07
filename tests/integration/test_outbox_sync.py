"""Integration tests for the outbox syncer and its job handlers (task 22).

Each test registers a temporary directory as a ``source_root``, scans it and
drains ``extract``/``chunk`` with a ``Worker``; the ``embed`` handler runs with
a ``FakeEmbedder`` (dim 8), so no model endpoint is contacted. ``sync_once``
then pushes the outbox into per-test OpenSearch aliases on the shared
container. Handlers and the syncer commit through ``session_scope`` and
``sync_once`` itself, so setup and assertions go through ``session_scope`` too
and every test cleans up its rows and indices afterwards.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from testcontainers.community.opensearch import OpenSearchContainer

from fakes import FakeEmbedder
from mynotes.config import Settings
from mynotes.db.models import (
    Chunk,
    Document,
    EmbeddingCache,
    File,
    IndexOutbox,
    Job,
    ScanRun,
    SourceRoot,
    Tag,
    TagAlias,
    Workspace,
)
from mynotes.db.session import reset_engine, session_scope
from mynotes.index.documents import indexable_chunk_ids
from mynotes.index.opensearch import ensure_indices, get_client
from mynotes.index.outbox import enqueue_chunk_deletes
from mynotes.index.sync import SyncStats, sync_once
from mynotes.jobs import pipeline
from mynotes.jobs.registry import JobContext
from mynotes.jobs.worker import Worker
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
)
_HEADINGS = ("Introdução", "Resultados", "Riscos", "Próximos passos")

_NOTE = """---
title: Relatório Trimestral
tags: [Trabalho, Área-Técnica]
date: 2026-09-01
---

# Introdução
O trimestre trouxe crescimento moderado e uma demanda estável dos clientes corporativos.

# Resultados
A receita cresceu doze por cento e a margem bruta manteve-se estável apesar do câmbio.

# Riscos
O principal risco é a dependência de um único fornecedor estrangeiro de componentes.

# Próximos passos
A próxima etapa é automatizar a conciliação contábil e contratar dois analistas novos.
"""

_EDITED_NOTE = _NOTE.replace(
    "A receita cresceu doze por cento e a margem bruta manteve-se estável apesar do câmbio.",
    "A receita encolheu três por cento no período e a margem bruta caiu com o custo dos insumos.",
)


@dataclass
class SyncEnv:
    """The registered test root, the fake embedder and the test's aliases."""

    settings: Settings
    fake: FakeEmbedder
    root_id: int
    workspace_id: int
    root_path: Path
    chunks_alias: str


@pytest.fixture
def sync_env(
    migrated_dsn: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    os_container: OpenSearchContainer,
) -> Iterator[SyncEnv]:
    """Container settings (dim 8, fast settle, grace 0) and per-test aliases."""
    settings = Settings(postgres={"dsn": migrated_dsn})
    config = os_container.get_config()
    suffix = uuid4().hex[:8]
    settings.opensearch.url = f"http://{config['host']}:{config['port']}"
    settings.opensearch.chunks_alias = f"t{suffix}_chunks"
    settings.opensearch.entities_alias = f"t{suffix}_entities"
    settings.models.embedding.dim = 8
    settings.scan.settle_seconds = 0
    settings.scan.delete_grace_seconds = 0
    monkeypatch.setattr("mynotes.config.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.db.session.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.scan.scanner.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.jobs.worker.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.index.sync.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.index.opensearch.get_settings", lambda: settings)
    reset_engine()
    fake = FakeEmbedder(dim=8)
    settings.models.embedding.model = fake.model
    with session_scope() as session:
        workspace = Workspace(name=f"sync-{suffix}")
        session.add(workspace)
        session.flush()
        root = SourceRoot(workspace_id=workspace.id, name="root", abs_path=str(tmp_path))
        session.add(root)
        session.flush()
        env = SyncEnv(
            settings=settings,
            fake=fake,
            root_id=root.id,
            workspace_id=workspace.id,
            root_path=tmp_path,
            chunks_alias=settings.opensearch.chunks_alias,
        )
    ensure_indices(get_client(), settings)
    client = get_client()
    yield env
    for alias in (settings.opensearch.chunks_alias, settings.opensearch.entities_alias):
        if client.indices.exists(index=f"{alias}_v1"):
            client.indices.delete(index=f"{alias}_v1")
    with session_scope() as session:
        session.execute(delete(Job).where(Job.type.in_(_JOB_TYPES)))
        session.execute(delete(IndexOutbox).where(IndexOutbox.doc_kind == "chunk"))
        session.execute(delete(EmbeddingCache).where(EmbeddingCache.model == fake.model))
        session.execute(delete(File).where(File.source_root_id == env.root_id))
        session.execute(delete(Document).where(Document.source_root_id == env.root_id))
        session.execute(delete(ScanRun).where(ScanRun.source_root_id == env.root_id))
        session.execute(delete(SourceRoot).where(SourceRoot.id == env.root_id))
        session.execute(delete(TagAlias).where(TagAlias.workspace_id == env.workspace_id))
        session.execute(delete(Tag).where(Tag.workspace_id == env.workspace_id))
        session.execute(delete(Workspace).where(Workspace.id == env.workspace_id))
    reset_engine()


def write_file(path: Path, content: str, mtime_ns: int = _MTIME) -> None:
    """Create a file with a fixed mtime so scans are deterministic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))


def drain_extract_chunk() -> None:
    """Run the worker until no extract/chunk job is claimable."""
    worker = Worker(concurrency=1, poll_interval_s=0.05, types=["extract", "chunk"])
    while worker.run_once():
        pass


def run_embed(env: SyncEnv, version_id: int) -> None:
    """Run the embed handler with the fake embedder, like the worker would."""
    pipeline.handle_embed(
        {"version_id": version_id},
        JobContext(worker_id="test", settings=env.settings, embedder=env.fake),
    )


def run_scan_handler(env: SyncEnv, job_type: str, document_id: int) -> None:
    """Run the scanner-enqueued handler of one type for a document."""
    handlers = {
        "reindex_metadata": pipeline.handle_reindex_metadata,
        "delete_document": pipeline.handle_delete_document,
    }
    with session_scope() as session:
        jobs = [
            job
            for job in session.execute(select(Job).where(Job.type == job_type)).scalars()
            if job.payload == {"document_id": document_id}
        ]
        payload = dict(jobs[0].payload) if jobs else None
    assert payload is not None, f"no {job_type} job for document {document_id}"
    handlers[job_type](
        payload, JobContext(worker_id="test", settings=env.settings, embedder=env.fake)
    )


def ingest(env: SyncEnv, rel_path: str, content: str, mtime_ns: int = _MTIME) -> None:
    """Write a file, scan, drain extract/chunk and embed its latest version."""
    write_file(env.root_path / rel_path, content, mtime_ns)
    scan_root(env.root_id)
    drain_extract_chunk()
    with session_scope() as session:
        document = session.execute(
            select(Document)
            .join(File, File.id == Document.current_file_id)
            .where(File.rel_path == rel_path)
        ).scalar_one()
        version_id = document.latest_version_id
    assert version_id is not None
    run_embed(env, version_id)


def document_id_for(env: SyncEnv, rel_path: str) -> int:
    """The document currently stored for ``rel_path``."""
    with session_scope() as session:
        file_row = session.execute(
            select(File).where(File.source_root_id == env.root_id, File.rel_path == rel_path)
        ).scalar_one()
        assert file_row.document_id is not None
        return file_row.document_id


def chunk_ids_of(env: SyncEnv, document_id: int) -> list[str]:
    """The Postgres chunk ids of a document (as strings, ordered by ordinal)."""
    with session_scope() as session:
        return list(
            session.execute(
                select(Chunk.id).where(Chunk.document_id == document_id).order_by(Chunk.ordinal)
            ).scalars()
        )


def refresh_count(alias: str) -> int:
    """Refresh the alias and return its document count."""
    client = get_client()
    client.indices.refresh(index=alias)
    return int(client.count(index=alias)["count"])


def indexed_hits(alias: str) -> list[dict]:
    """Every indexed document's ``_source`` (plus ``_id``) after a refresh."""
    client = get_client()
    client.indices.refresh(index=alias)
    result = client.search(index=alias, body={"query": {"match_all": {}}, "size": 100})
    hits = result["hits"]["hits"]
    return [{"_id": hit["_id"], **hit["_source"]} for hit in hits]


def expected_count(env: SyncEnv) -> int:
    """The number of indexable chunks across the test root's documents."""
    with session_scope() as session:
        document_ids = list(
            session.execute(
                select(Document.id).where(Document.source_root_id == env.root_id)
            ).scalars()
        )
        return sum(
            len(indexable_chunk_ids(session, document_id, env.settings))
            for document_id in document_ids
        )


def pending_outbox_rows(session: Session) -> list[IndexOutbox]:
    """The unprocessed chunk outbox rows, ordered by id."""
    return list(
        session.execute(
            select(IndexOutbox)
            .where(IndexOutbox.doc_kind == "chunk", IndexOutbox.processed_at.is_(None))
            .order_by(IndexOutbox.id)
        ).scalars()
    )


def test_sync_drains_full_pipeline_into_opensearch(sync_env: SyncEnv) -> None:
    """Two documents, embedded: the index holds exactly the indexable chunks."""
    ingest(sync_env, "relatorio.md", _NOTE)
    ingest(sync_env, "sub/segundo.md", _NOTE.replace("Relatório", "Segundo"))

    stats = sync_once()

    assert isinstance(stats, SyncStats)
    assert stats.processed > 0
    assert stats.upserts > 0
    assert stats.deletes == 0
    assert stats.failures == 0
    assert refresh_count(sync_env.chunks_alias) == expected_count(sync_env) > 0
    with session_scope() as session:
        assert pending_outbox_rows(session) == []


def test_document_source_has_expected_fields(sync_env: SyncEnv) -> None:
    """A synced chunk carries every mapped field, a dim-8 vector and paths."""
    ingest(sync_env, "Projetos/Ação/notas.md", _NOTE)
    sync_once()

    hits = indexed_hits(sync_env.chunks_alias)
    assert len(hits) == expected_count(sync_env)
    with session_scope() as session:
        document = session.execute(
            select(Document).where(Document.source_root_id == sync_env.root_id)
        ).scalar_one()
        assert document.language is not None
        for hit in hits:
            source = {key: value for key, value in hit.items() if key != "_id"}
            assert hit["_id"] == source["chunk_id"]
            assert source["document_id"] == str(document.id)
            assert source["version_id"] == str(document.latest_version_id)
            assert source["source_root_id"] == str(document.source_root_id)
            assert source["workspace_id"] == str(document.workspace_id)
            assert source["format"] == "md"
            assert source["doc_status"] == "current"
            assert source["language"] == document.language
            assert source["title"] == "Relatório Trimestral"
            assert source["tags"] == ["folder:acao", "folder:projetos", "area-tecnica", "trabalho"]
            assert source["doc_date"] == "2026-09-01T00:00:00+00:00"
            assert source["is_latest"] is True
            assert source["is_table"] is False
            assert source["path"] == "Projetos/Ação/notas.md"
            assert source["heading_path_text"] in _HEADINGS
            assert isinstance(source["ordinal"], int)
            assert len(source["embedding"]) == 8
            assert source["path_ancestors"] == ["Projetos", "Projetos/Ação"]
            assert source["path_ancestors_folded"] == ["projetos", "projetos/acao"]


def test_edit_replaces_old_version_chunks(sync_env: SyncEnv) -> None:
    """After an edit the old chunks leave the index and the new ones enter."""
    ingest(sync_env, "relatorio.md", _NOTE)
    sync_once()
    old_ids = {hit["_id"] for hit in indexed_hits(sync_env.chunks_alias)}
    assert old_ids

    ingest(sync_env, "relatorio.md", _EDITED_NOTE, _MTIME + 5_000_000_000)
    sync_once()

    new_ids = {hit["_id"] for hit in indexed_hits(sync_env.chunks_alias)}
    assert new_ids
    assert new_ids.isdisjoint(old_ids)
    assert len(new_ids) == expected_count(sync_env)


def test_deleted_file_removes_chunks_from_index(sync_env: SyncEnv) -> None:
    """Delete → scanner → delete_document → sync: the chunks are gone."""
    ingest(sync_env, "relatorio.md", _NOTE)
    sync_once()
    document_id = document_id_for(sync_env, "relatorio.md")
    stored_ids = chunk_ids_of(sync_env, document_id)
    assert refresh_count(sync_env.chunks_alias) > 0
    (sync_env.root_path / "relatorio.md").unlink()
    scan_root(sync_env.root_id)
    scan_root(sync_env.root_id)

    with session_scope() as session:
        document = session.get(Document, document_id)
        assert document is not None and document.status == "deleted"

    run_scan_handler(sync_env, "delete_document", document_id)
    sync_once()

    assert refresh_count(sync_env.chunks_alias) == 0
    assert chunk_ids_of(sync_env, document_id) == stored_ids


def test_rename_updates_path_without_reembedding(sync_env: SyncEnv) -> None:
    """A move → reindex_metadata → sync updates `path` and calls no embedder."""
    ingest(sync_env, "relatorio.md", _NOTE)
    sync_once()
    document_id = document_id_for(sync_env, "relatorio.md")
    calls_before = [list(batch) for batch in sync_env.fake.calls]
    (sync_env.root_path / "Projetos").mkdir()
    (sync_env.root_path / "relatorio.md").rename(sync_env.root_path / "Projetos" / "relatorio.md")
    scan_root(sync_env.root_id)

    run_scan_handler(sync_env, "reindex_metadata", document_id)
    sync_once()

    hits = indexed_hits(sync_env.chunks_alias)
    assert hits
    assert {hit["path"] for hit in hits} == {"Projetos/relatorio.md"}
    assert [hit["path_ancestors_folded"] for hit in hits] == [["projetos"]] * len(hits)
    assert refresh_count(sync_env.chunks_alias) == expected_count(sync_env)
    assert sync_env.fake.calls == calls_before


def test_collapse_upsert_then_delete_leaves_chunk_absent(sync_env: SyncEnv) -> None:
    """An upsert followed by a delete for the same chunk ends in a delete."""
    ingest(sync_env, "relatorio.md", _NOTE)
    with session_scope() as session:
        chunk_ids = list(
            session.execute(
                select(Chunk.id)
                .join(Document, Chunk.document_id == Document.id)
                .where(Document.source_root_id == sync_env.root_id)
            ).scalars()
        )
        assert chunk_ids
        enqueue_chunk_deletes(session, chunk_ids)

    sync_once()

    assert refresh_count(sync_env.chunks_alias) == 0


def test_unreachable_opensearch_fails_rows_without_processing(
    sync_env: SyncEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With OpenSearch down, rows gain an attempt and stay unprocessed."""
    ingest(sync_env, "relatorio.md", _NOTE)
    with session_scope() as session:
        row_ids = [row.id for row in pending_outbox_rows(session)]
        assert row_ids
    broken = sync_env.settings.model_copy(deep=True)
    broken.opensearch.url = "http://127.0.0.1:1"
    monkeypatch.setattr("mynotes.index.sync.get_settings", lambda: broken)
    monkeypatch.setattr("mynotes.index.opensearch.get_settings", lambda: broken)

    stats = sync_once()

    assert stats.processed == 0
    assert stats.failures == len(row_ids)
    with session_scope() as session:
        stored = {
            row.id: row
            for row in session.execute(
                select(IndexOutbox).where(IndexOutbox.id.in_(row_ids))
            ).scalars()
        }
        assert set(stored) == set(row_ids)
        for row in stored.values():
            assert row.processed_at is None
            assert row.attempts == 1
            assert row.last_error


def test_outbox_bulk_insert_writes_complete_rows(sync_env: SyncEnv) -> None:
    """The executemany outbox insert (task 51) writes one complete row per id.

    Outbox rows are inserted with a Core ``insert()`` executemany; this pins
    the row shape (string doc ids, ops) and the server-side defaults.
    """
    from sqlalchemy import select

    from mynotes.db.models import IndexOutbox
    from mynotes.index.outbox import enqueue_chunk_deletes, enqueue_chunk_upserts

    with session_scope() as session:
        upserts = enqueue_chunk_upserts(session, [f"{uuid4()}" for _ in range(3)])
        deletes = enqueue_chunk_deletes(session, [f"{uuid4()}" for _ in range(2)])
    assert upserts == 3 and deletes == 2
    with session_scope() as session:
        rows_ = list(session.execute(select(IndexOutbox).order_by(IndexOutbox.id)).scalars())
        assert len(rows_) == 5
        assert [row.op for row in rows_] == ["upsert"] * 3 + ["delete"] * 2
        assert all(row.index_name == "chunks" and row.doc_kind == "chunk" for row in rows_)
        assert all(
            row.doc_id and row.created_at is not None and row.processed_at is None for row in rows_
        )
