"""Integration tests for version coalescing, retention and history (task 33).

Each test registers a temporary directory as a ``source_root``, scans it and
drains ``extract``/``chunk`` with a ``Worker``; the ``embed`` and
``apply_retention`` handlers run through the queue with a ``FakeEmbedder``
(dim 8), so no model endpoint is contacted. Index assertions go against
per-test OpenSearch aliases on the shared container. Handlers commit through
``session_scope``, so setup and assertions do too and every test cleans up its
rows and indices afterwards.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
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
    DocumentVersion,
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
from mynotes.index.sync import sync_once
from mynotes.jobs import pipeline
from mynotes.jobs.registry import JobContext, Priority
from mynotes.jobs.worker import Worker
from mynotes.scan.scanner import scan_root
from mynotes.versioning.history import document_history, history_timeline_text

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

_V1 = """---
title: Plano Alfa
date: 2026-03-15
---

# Objetivos

conteúdo original dos objetivos do plano estratégico.

# Equipa

lista inicial da equipa responsável pelo plano.
"""

_V2 = """---
title: Plano Alfa
date: 2026-06-02
---

# Objetivos

conteúdo original dos objetivos do plano estratégico.

# Equipa

lista renovada da equipa responsável pelo plano.

# Riscos

novos riscos identificados na revisão trimestral.
"""


@dataclass
class VersionsEnv:
    """The registered test root, the fake embedder and the test's aliases."""

    settings: Settings
    fake: FakeEmbedder
    root_id: int
    workspace_id: int
    root_path: Path
    chunks_alias: str


@pytest.fixture
def versions_env(
    migrated_dsn: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    os_container: OpenSearchContainer,
) -> Iterator[VersionsEnv]:
    """Container settings (dim 8, fast settle) and per-test OpenSearch aliases."""
    settings = Settings(postgres={"dsn": migrated_dsn})
    config = os_container.get_config()
    suffix = uuid4().hex[:8]
    settings.opensearch.url = f"http://{config['host']}:{config['port']}"
    settings.opensearch.chunks_alias = f"t{suffix}_chunks"
    settings.opensearch.entities_alias = f"t{suffix}_entities"
    settings.models.embedding.dim = 8
    settings.scan.settle_seconds = 0
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
        workspace = Workspace(name=f"vers-{suffix}")
        session.add(workspace)
        session.flush()
        root = SourceRoot(workspace_id=workspace.id, name="root", abs_path=str(tmp_path))
        session.add(root)
        session.flush()
        env = VersionsEnv(
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


def drain_retention() -> None:
    """Run the worker until no apply_retention job is claimable."""
    worker = Worker(concurrency=1, poll_interval_s=0.05, types=["apply_retention"])
    while worker.run_once():
        pass


def document_for(env: VersionsEnv, rel_path: str) -> Document:
    """The document stored for ``rel_path`` (detached; access columns inside)."""
    with session_scope() as session:
        file_row = session.execute(
            select(File).where(File.source_root_id == env.root_id, File.rel_path == rel_path)
        ).scalar_one()
        assert file_row.document_id is not None
        document = session.get(Document, file_row.document_id)
        assert document is not None
        session.expunge(document)
        return document


def versions_of(env: VersionsEnv, document_id: int) -> list[DocumentVersion]:
    """The document's versions ordered by version_no (detached)."""
    with session_scope() as session:
        versions = list(
            session.execute(
                select(DocumentVersion)
                .where(DocumentVersion.document_id == document_id)
                .order_by(DocumentVersion.version_no)
            ).scalars()
        )
        for version in versions:
            session.expunge(version)
        return versions


def ingest_cycle(env: VersionsEnv, rel_path: str, content: str, mtime_ns: int) -> int:
    """Write, scan, drain extract/chunk, embed the latest version, apply retention.

    Returns:
        The id of the version the cycle produced.
    """
    write_file(env.root_path / rel_path, content, mtime_ns)
    scan_root(env.root_id)
    drain_extract_chunk()
    document = document_for(env, rel_path)
    with session_scope() as session:
        version_id = session.get(Document, document.id).latest_version_id
    assert version_id is not None
    pipeline.handle_embed(
        {"version_id": version_id},
        JobContext(worker_id="test", settings=env.settings, embedder=env.fake),
    )
    drain_retention()
    return version_id


def rows(session: Session, model: type, **filters: object) -> list:
    """Fetch all rows of ``model`` matching the filters, ordered by id."""
    stmt = select(model).filter_by(**filters).order_by(model.id)
    return list(session.execute(stmt).scalars().all())


def indexed_hits(alias: str) -> list[dict]:
    """Every indexed document's ``_source`` (plus ``_id``) after a refresh."""
    client = get_client()
    client.indices.refresh(index=alias)
    result = client.search(index=alias, body={"query": {"match_all": {}}, "size": 100})
    return [{"_id": hit["_id"], **hit["_source"]} for hit in result["hits"]["hits"]]


def search_ids(alias: str, filters: list[dict]) -> set[str]:
    """The chunk ids matching a filtered bool query, after a refresh."""
    client = get_client()
    client.indices.refresh(index=alias)
    body = {"query": {"bool": {"filter": filters}}, "_source": ["chunk_id"], "size": 100}
    result = client.search(index=alias, body=body)
    return {hit["_source"]["chunk_id"] for hit in result["hits"]["hits"]}


def expected_count(env: VersionsEnv, document_id: int) -> int:
    """The number of indexable chunks of one document under current settings."""
    with session_scope() as session:
        return len(indexable_chunk_ids(session, document_id, env.settings))


def test_two_edits_within_interval_coalesce_into_one_version(versions_env: VersionsEnv) -> None:
    """v1 + two rapid edits: one coalesced v2 row is reused, not a third version."""
    env = versions_env
    env.settings.versions.min_interval_hours = 24.0
    second = _V1.replace("conteúdo original", "segunda versão do conteúdo")
    third = _V1.replace("conteúdo original", "terceira versão reescrita")

    write_file(env.root_path / "plano.md", _V1)
    scan_root(env.root_id)
    drain_extract_chunk()
    write_file(env.root_path / "plano.md", second, _MTIME + 5_000_000_000)
    scan_root(env.root_id)
    drain_extract_chunk()
    document = document_for(env, "plano.md")
    versions = versions_of(env, document.id)
    assert [item.version_no for item in versions] == [1, 2]
    coalesced = versions[1]
    assert coalesced.is_latest and coalesced.status == "chunked"
    coalesced_id = coalesced.id
    coalesced_created_at = coalesced.created_at
    coalesced_chunk_ids = _chunk_ids(env, coalesced_id)

    write_file(env.root_path / "plano.md", third, _MTIME + 10_000_000_000)
    scan_root(env.root_id)
    drain_extract_chunk()

    versions = versions_of(env, document.id)
    assert [item.version_no for item in versions] == [1, 2]
    reused = versions[1]
    assert reused.id == coalesced_id
    assert reused.created_at == coalesced_created_at
    assert reused.is_latest and reused.status == "chunked"
    assert reused.mtime_ns == _MTIME + 10_000_000_000
    assert reused.ir_json is not None
    file_hash = _file_content_hash(env, "plano.md")
    assert reused.content_hash == file_hash
    with session_scope() as session:
        assert document.latest_version_id == reused.id
        replaced = {str(row.doc_id) for row in rows(session, IndexOutbox, op="delete")}
        assert set(coalesced_chunk_ids) <= replaced
    new_texts = " ".join(_chunk_texts(env, reused.id))
    assert "terceira versão reescrita" in new_texts
    assert "segunda versão do conteúdo" not in new_texts
    v1_texts = " ".join(_chunk_texts(env, versions[0].id))
    assert "conteúdo original dos objetivos" in v1_texts


def test_max_per_document_two_purges_the_oldest_version(versions_env: VersionsEnv) -> None:
    """min_interval 0 and max_per_document=2: after three edits v1 is gone."""
    env = versions_env
    env.settings.versions.min_interval_hours = 0.0
    env.settings.versions.max_per_document = 2
    second = _V1.replace("conteúdo original", "segunda versão do conteúdo")
    third = _V1.replace("conteúdo original", "terceira versão reescrita")

    v1_id = ingest_cycle(env, "plano.md", _V1, _MTIME)
    with session_scope() as session:
        v1_cache_keys = list(
            session.execute(
                select(EmbeddingCache.content_hash).where(EmbeddingCache.model == env.fake.model)
            ).scalars()
        )
    assert v1_cache_keys
    v2_id = ingest_cycle(env, "plano.md", second, _MTIME + 5_000_000_000)
    v3_id = ingest_cycle(env, "plano.md", third, _MTIME + 10_000_000_000)

    document = document_for(env, "plano.md")
    versions = versions_of(env, document.id)
    assert [item.version_no for item in versions] == [2, 3]
    assert versions[0].id == v2_id and versions[1].id == v3_id
    assert [item.is_latest for item in versions] == [False, True]
    with session_scope() as session:
        assert session.get(DocumentVersion, v1_id) is None
        assert rows(session, Chunk, document_version_id=v1_id) == []
        retention_jobs = rows(session, Job, type="apply_retention")
        assert len(retention_jobs) == 3
        assert all(job.status == "done" for job in retention_jobs)
        assert all(job.priority == Priority.MAINTENANCE for job in retention_jobs)
        assert {job.dedupe_key for job in retention_jobs} == {f"apply_retention:{document.id}"}
        assert all(job.payload == {"document_id": document.id} for job in retention_jobs)
        remaining_cache_keys = list(
            session.execute(
                select(EmbeddingCache.content_hash).where(EmbeddingCache.model == env.fake.model)
            ).scalars()
        )
    assert set(v1_cache_keys) <= set(remaining_cache_keys)


def test_index_old_versions_off_indexes_only_the_latest(versions_env: VersionsEnv) -> None:
    """Default setting: after a sync only the latest version's chunks remain."""
    env = versions_env
    env.settings.versions.min_interval_hours = 0.0
    second = _V1.replace("conteúdo original", "segunda versão do conteúdo")

    v1_id = ingest_cycle(env, "plano.md", _V1, _MTIME)
    sync_once()
    assert {hit["_id"] for hit in indexed_hits(env.chunks_alias)} == set(_chunk_ids(env, v1_id))

    v2_id = ingest_cycle(env, "plano.md", second, _MTIME + 5_000_000_000)
    sync_once()
    hits = indexed_hits(env.chunks_alias)
    assert hits
    assert {hit["_id"] for hit in hits} == set(_chunk_ids(env, v2_id))
    assert {hit["version_id"] for hit in hits} == {str(v2_id)}
    assert all(hit["is_latest"] for hit in hits)
    assert len(hits) == expected_count(env, document_for(env, "plano.md").id)


def test_index_old_versions_on_indexes_retained_snapshots(versions_env: VersionsEnv) -> None:
    """With the setting on, old retained snapshots are indexed as is_latest=false."""
    env = versions_env
    env.settings.versions.min_interval_hours = 0.0
    env.settings.versions.index_old_versions = True
    second = _V1.replace("conteúdo original", "segunda versão do conteúdo")

    v1_id = ingest_cycle(env, "plano.md", _V1, _MTIME)
    v2_id = ingest_cycle(env, "plano.md", second, _MTIME + 5_000_000_000)
    sync_once()

    hits = indexed_hits(env.chunks_alias)
    assert hits
    assert len(hits) == expected_count(env, document_for(env, "plano.md").id)
    by_version = {hit["version_id"]: hit["is_latest"] for hit in hits}
    assert set(by_version) == {str(v1_id), str(v2_id)}
    assert by_version[str(v1_id)] is False
    assert by_version[str(v2_id)] is True

    latest_ids = search_ids(env.chunks_alias, [{"term": {"is_latest": True}}])
    old_ids = search_ids(env.chunks_alias, [{"term": {"is_latest": False}}])
    assert latest_ids == set(_chunk_ids(env, v2_id))
    assert old_ids == set(_chunk_ids(env, v1_id))
    with session_scope() as session:
        versions = list(
            session.execute(
                select(DocumentVersion)
                .where(DocumentVersion.document_id == document_for(env, "plano.md").id)
                .order_by(DocumentVersion.version_no)
            ).scalars()
        )
        assert [item.version_no for item in versions] == [1, 2]


def test_document_history_lists_versions_newest_first(versions_env: VersionsEnv) -> None:
    """document_history: newest first, changed_sections against the previous version."""
    env = versions_env
    env.settings.versions.min_interval_hours = 0.0
    second = _V2

    ingest_cycle(env, "plano.md", _V1, _MTIME)
    ingest_cycle(env, "plano.md", second, _MTIME + 5_000_000_000)
    document = document_for(env, "plano.md")

    with session_scope() as session:
        infos = document_history(session, document.id)
        timeline = history_timeline_text(infos, "Plano Alfa")

    assert [info.version_no for info in infos] == [2, 1]
    assert [info.is_latest for info in infos] == [True, False]
    assert [info.status for info in infos] == ["embedded", "embedded"]
    assert [info.chunk_count for info in infos] == [3, 2]
    assert [info.rel_path for info in infos] == ["plano.md", "plano.md"]
    assert infos[0].doc_date == datetime(2026, 6, 2, tzinfo=UTC)
    assert infos[0].changed_sections == ["Equipa", "Riscos"]
    assert infos[1].changed_sections == []
    assert timeline.startswith('Version history of "Plano Alfa": v2 ')
    assert "(latest; changed: Equipa, Riscos)" in timeline
    assert "v1 2026-03-15" in timeline


def _chunk_ids(env: VersionsEnv, version_id: int) -> list[str]:
    """The Postgres chunk ids of a version as strings, ordered by ordinal."""
    with session_scope() as session:
        return [
            str(chunk_id)
            for chunk_id in session.execute(
                select(Chunk.id)
                .where(Chunk.document_version_id == version_id)
                .order_by(Chunk.ordinal)
            ).scalars()
        ]


def _chunk_texts(env: VersionsEnv, version_id: int) -> list[str]:
    """The chunk texts of a version, ordered by ordinal."""
    with session_scope() as session:
        return list(
            session.execute(
                select(Chunk.text)
                .where(Chunk.document_version_id == version_id)
                .order_by(Chunk.ordinal)
            ).scalars()
        )


def _file_content_hash(env: VersionsEnv, rel_path: str) -> str:
    """The current content hash stored on the file row."""
    with session_scope() as session:
        file_row = session.execute(
            select(File).where(File.source_root_id == env.root_id, File.rel_path == rel_path)
        ).scalar_one()
        return file_row.content_hash
