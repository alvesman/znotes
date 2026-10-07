"""Integration tests for drift reconciliation and the index rebuild (task 23).

The setup mirrors ``test_outbox_sync.py``: a temporary source root is scanned,
``extract``/``chunk`` are drained with a ``Worker`` and the ``embed`` handler
runs with a ``FakeEmbedder`` (dim 8); ``sync_once`` pushes the outbox into
per-test OpenSearch aliases. Direct OpenSearch edits and direct Postgres edits
inject drift; the rebuild tests patch the embedder with the same fake so
nothing touches a model endpoint. The drift report file is redirected into
``tmp_path`` so tests never touch the real user state dir.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from testcontainers.community.opensearch import OpenSearchContainer

from fakes import FakeEmbedder
from mynotes.config import Settings
from mynotes.core.setup import _set_root_enabled
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
from mynotes.index import rebuild as rebuild_module
from mynotes.index.documents import indexable_chunk_ids
from mynotes.index.opensearch import current_physical_index, ensure_indices, get_client
from mynotes.index.rebuild import rebuild_chunks_index
from mynotes.index.reconcile import reconcile
from mynotes.index.sync import sync_once
from mynotes.jobs import pipeline
from mynotes.jobs.registry import JobContext
from mynotes.jobs.worker import Worker
from mynotes.scan.scanner import scan_root
from mynotes.text.normalize import normalize_key

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

_NOTE = """---
title: Relatório Trimestral
tags: [Trabalho]
date: 2026-09-01
---

# Introdução
O trimestre trouxe crescimento moderado e uma demanda estável dos clientes.

# Resultados
A receita cresceu doze por cento e a margem bruta manteve-se estável.
"""

_SECOND = _NOTE.replace("Relatório", "Segundo")


@dataclass
class Env:
    """The registered test root, the fake embedder and the test's aliases."""

    settings: Settings
    fake: FakeEmbedder
    root_id: int
    workspace_id: int
    root_path: Path
    chunks_alias: str


@pytest.fixture
def env(
    migrated_dsn: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    os_container: OpenSearchContainer,
) -> Iterator[Env]:
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
    monkeypatch.setattr("mynotes.index.reconcile.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.index.rebuild.get_settings", lambda: settings)
    fake = FakeEmbedder(dim=8)
    settings.models.embedding.model = fake.model
    monkeypatch.setattr("mynotes.index.rebuild.get_embedder", lambda: fake)
    monkeypatch.setattr(
        "mynotes.index.reconcile.report_path", lambda: tmp_path / "drift_report.json"
    )
    reset_engine()
    with session_scope() as session:
        workspace = Workspace(name=f"reconcile-{suffix}")
        session.add(workspace)
        session.flush()
        root = SourceRoot(workspace_id=workspace.id, name="root", abs_path=str(tmp_path))
        session.add(root)
        session.flush()
        state = Env(
            settings=settings,
            fake=fake,
            root_id=root.id,
            workspace_id=workspace.id,
            root_path=tmp_path,
            chunks_alias=settings.opensearch.chunks_alias,
        )
    ensure_indices(get_client(), settings)
    client = get_client()
    yield state
    for alias in (settings.opensearch.chunks_alias, settings.opensearch.entities_alias):
        try:
            physical = list(client.indices.get(index=f"{alias}_v*"))
        except Exception:
            physical = []
        for name in physical:
            client.indices.delete(index=name)
    with session_scope() as session:
        session.execute(delete(Job).where(Job.type.in_(_JOB_TYPES)))
        session.execute(delete(IndexOutbox).where(IndexOutbox.doc_kind == "chunk"))
        session.execute(delete(EmbeddingCache).where(EmbeddingCache.model == fake.model))
        session.execute(delete(File).where(File.source_root_id == state.root_id))
        session.execute(delete(Document).where(Document.source_root_id == state.root_id))
        session.execute(delete(ScanRun).where(ScanRun.source_root_id == state.root_id))
        session.execute(delete(SourceRoot).where(SourceRoot.id == state.root_id))
        session.execute(delete(TagAlias).where(TagAlias.workspace_id == state.workspace_id))
        session.execute(delete(Tag).where(Tag.workspace_id == state.workspace_id))
        session.execute(delete(Workspace).where(Workspace.id == state.workspace_id))
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


def ingest(env: Env, rel_path: str, content: str, mtime_ns: int = _MTIME) -> None:
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
    pipeline.handle_embed(
        {"version_id": version_id},
        JobContext(worker_id="test", settings=env.settings, embedder=env.fake),
    )


def expected_count(env: Env) -> int:
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


def refresh_count(index: str) -> int:
    """Refresh an index (or alias) and return its document count."""
    client = get_client()
    client.indices.refresh(index=index)
    return int(client.count(index=index)["count"])


def indexed_chunk_ids(alias: str) -> list[str]:
    """Every indexed chunk id (``_id``) after a refresh, sorted."""
    client = get_client()
    client.indices.refresh(index=alias)
    result = client.search(index=alias, body={"query": {"match_all": {}}, "size": 100})
    return sorted(hit["_id"] for hit in result["hits"]["hits"])


def reconcile_after_refresh(env: Env, fix: bool = False):
    """Refresh the chunks alias, then reconcile (searches only see refreshes)."""
    get_client().indices.refresh(index=env.chunks_alias)
    return reconcile(fix=fix)


def document_id_for(env: Env, rel_path: str) -> int:
    """The document currently stored for ``rel_path``."""
    with session_scope() as session:
        file_row = session.execute(
            select(File).where(File.source_root_id == env.root_id, File.rel_path == rel_path)
        ).scalar_one()
        assert file_row.document_id is not None
        return file_row.document_id


def chunk_ids_of(env: Env, document_id: int) -> list[str]:
    """The Postgres chunk ids of a document (as strings, ordered by ordinal)."""
    with session_scope() as session:
        return [
            str(chunk_id)
            for chunk_id in session.execute(
                select(Chunk.id).where(Chunk.document_id == document_id).order_by(Chunk.ordinal)
            ).scalars()
        ]


def test_reconcile_reports_zero_drift_after_sync(env: Env) -> None:
    """After the normal pipeline plus a sync, nothing is missing/extra/stale."""
    ingest(env, "relatorio.md", _NOTE)
    ingest(env, "sub/segundo.md", _SECOND)
    sync_once()

    report = reconcile_after_refresh(env)

    assert report.expected == expected_count(env) > 0
    assert report.indexed == report.expected
    assert report.missing == []
    assert report.extra == []
    assert report.stale == []


def test_disabled_root_leaves_the_index_and_returns_without_reembedding(env: Env) -> None:
    """Disabling a root removes its chunks from the index; re-enabling restores them."""
    ingest(env, "relatorio.md", _NOTE)
    sync_once()
    chunk_count = expected_count(env)
    assert chunk_count > 0 and refresh_count(env.chunks_alias) == chunk_count
    embed_calls = len(env.fake.calls)

    with session_scope() as session:
        root = session.get(SourceRoot, env.root_id)
        assert root is not None
        assert _set_root_enabled(session, root, False, env.settings) == chunk_count
    sync_once()

    assert refresh_count(env.chunks_alias) == 0
    assert expected_count(env) == 0
    report = reconcile_after_refresh(env)
    assert report.missing == [] and report.extra == []

    with session_scope() as session:
        root = session.get(SourceRoot, env.root_id)
        assert root is not None
        assert _set_root_enabled(session, root, True, env.settings) == chunk_count
    sync_once()

    assert refresh_count(env.chunks_alias) == chunk_count
    assert len(env.fake.calls) == embed_calls


def test_missing_chunk_is_reported_and_repaired(env: Env) -> None:
    """A chunk deleted directly in OpenSearch is missing; the handler fixes it."""
    ingest(env, "relatorio.md", _NOTE)
    sync_once()
    victim = indexed_chunk_ids(env.chunks_alias)[0]
    get_client().delete(index=env.chunks_alias, id=victim)
    get_client().indices.refresh(index=env.chunks_alias)

    report = reconcile_after_refresh(env)

    assert report.missing == [victim]
    assert report.extra == [] and report.stale == []

    pipeline.handle_reconcile(
        {}, JobContext(worker_id="test", settings=env.settings, embedder=env.fake)
    )
    sync_once()

    assert refresh_count(env.chunks_alias) == expected_count(env)
    assert reconcile_after_refresh(env).missing == []


def test_bogus_doc_is_extra_and_removed_by_fix(env: Env) -> None:
    """A document indexed directly behind the alias is extra and gets deleted."""
    ingest(env, "relatorio.md", _NOTE)
    sync_once()
    client = get_client()
    client.index(index=env.chunks_alias, id="bogus-doc", body={"chunk_id": "bogus-doc"})
    client.indices.refresh(index=env.chunks_alias)

    report = reconcile_after_refresh(env)

    assert report.extra == ["bogus-doc"]
    assert report.missing == [] and report.stale == []

    reconcile(fix=True)
    sync_once()

    assert refresh_count(env.chunks_alias) == expected_count(env)
    clean = reconcile_after_refresh(env)
    assert clean.extra == [] and clean.indexed == clean.expected


def test_title_change_without_outbox_is_stale(env: Env) -> None:
    """A Postgres-only title change changes the meta hash of all its chunks."""
    ingest(env, "relatorio.md", _NOTE)
    ingest(env, "sub/segundo.md", _SECOND)
    sync_once()
    document_id = document_id_for(env, "relatorio.md")
    doc_chunks = chunk_ids_of(env, document_id)
    with session_scope() as session:
        document = session.get(Document, document_id)
        assert document is not None
        document.title = "Título Alterado"
        document.title_norm = normalize_key("Título Alterado")

    report = reconcile_after_refresh(env)

    assert sorted(report.stale) == doc_chunks
    assert report.missing == [] and report.extra == []

    reconcile(fix=True)
    sync_once()

    clean = reconcile_after_refresh(env)
    assert clean.stale == [] and clean.missing == [] and clean.extra == []


def test_rebuild_swaps_alias_to_v2_without_reembedding(env: Env) -> None:
    """The rebuild fills _v2 from Postgres + cache, swaps the alias, keeps _v1."""
    ingest(env, "relatorio.md", _NOTE)
    ingest(env, "sub/segundo.md", _SECOND)
    sync_once()
    calls_before = [list(batch) for batch in env.fake.calls]
    client = get_client()

    result = rebuild_chunks_index()

    assert result == {
        "old": f"{env.chunks_alias}_v1",
        "new": f"{env.chunks_alias}_v2",
        "count": expected_count(env),
        "reembedded": False,
    }
    assert current_physical_index(client, env.chunks_alias) == f"{env.chunks_alias}_v2"
    assert client.indices.exists(index=f"{env.chunks_alias}_v1")
    assert refresh_count(env.chunks_alias) == expected_count(env)
    assert sorted(indexed_chunk_ids(env.chunks_alias)) == sorted(
        indexed_chunk_ids(f"{env.chunks_alias}_v1")
    )
    assert env.fake.calls == calls_before


def test_rebuild_count_mismatch_leaves_alias_on_old_index(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verification shortfall raises and the alias stays on the old index."""
    ingest(env, "relatorio.md", _NOTE)
    sync_once()
    client = get_client()
    real_count = rebuild_module._count_after_refresh
    monkeypatch.setattr(
        rebuild_module,
        "_count_after_refresh",
        lambda os_client, index: real_count(os_client, index) - 1,
    )

    with pytest.raises(RuntimeError, match="incomplete"):
        rebuild_chunks_index()

    assert current_physical_index(client, env.chunks_alias) == f"{env.chunks_alias}_v1"
    assert client.indices.exists(index=f"{env.chunks_alias}_v2")
