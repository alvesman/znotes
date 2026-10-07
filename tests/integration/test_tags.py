"""Integration tests for tags: folder rules, persistence, aliases, filters (task 37).

The setup mirrors ``test_hybrid_search.py``: a temporary source root is scanned
and drained through ``extract``/``chunk`` with a ``Worker``; the ``embed``
handler runs with a ``FakeEmbedder`` (dim 8) and ``sync_once`` pushes the
chunks into per-test OpenSearch aliases. The ``tags`` settings carry the PLAN
§7.4 folder rules, so ``projects/alpha/note.md`` is tagged ``project:alpha``.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from testcontainers.community.opensearch import OpenSearchContainer
from typer.testing import CliRunner

from fakes import FakeEmbedder, FakeReranker
from mynotes.cli import app
from mynotes.config import FolderRule, Settings
from mynotes.db.models import (
    Chunk,
    Document,
    DocumentTag,
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
from mynotes.index.opensearch import ensure_indices, ensure_pipeline, get_client
from mynotes.index.sync import sync_once
from mynotes.jobs import pipeline
from mynotes.jobs.registry import JobContext
from mynotes.jobs.worker import Worker
from mynotes.retrieval.analyze import analyze_query
from mynotes.retrieval.hybrid import hybrid_search
from mynotes.scan.scanner import scan_root
from mynotes.tags.store import accepted_tags, ensure_tag, set_document_tags, tag_summaries

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

_FOLDER_RULES = [
    FolderRule(pattern="projects/{project}/**", tags=["project:{project}"]),
    FolderRule(pattern="**/archive/**", tags=["status:archived"]),
]

_NOTE = """---
title: Nota Alfa
tags: [ML]
date: 2026-09-01
---

# Ideia

#ideia Anotações sobre machine learning e busca híbrida no projeto.
"""

_OTHER_NOTE = """---
title: Outra Nota
date: 2026-09-02
---

# Outra

Texto sem relação com o tema principal deste repositório de notas.
"""


@dataclass
class TagsEnv:
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
) -> Iterator[TagsEnv]:
    """Container settings (dim 8, fast settle, grace 0, folder rules) and aliases."""
    settings = Settings(postgres={"dsn": migrated_dsn})
    config = os_container.get_config()
    suffix = uuid4().hex[:8]
    settings.opensearch.url = f"http://{config['host']}:{config['port']}"
    settings.opensearch.chunks_alias = f"t{suffix}_chunks"
    settings.opensearch.entities_alias = f"t{suffix}_entities"
    settings.models.embedding.dim = 8
    settings.scan.settle_seconds = 0
    settings.scan.delete_grace_seconds = 0
    settings.tags.folder_rules = list(_FOLDER_RULES)
    fake = FakeEmbedder(dim=8)
    settings.models.embedding.model = fake.model
    monkeypatch.setattr("mynotes.config.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.db.session.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.scan.scanner.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.jobs.worker.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.index.sync.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.index.opensearch.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.retrieval.hybrid.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.retrieval.hybrid.get_embedder", lambda: fake)
    monkeypatch.setattr("mynotes.retrieval.pipeline.get_reranker", lambda: FakeReranker())
    reset_engine()
    with session_scope() as session:
        workspace = Workspace(name=f"tags-{suffix}")
        session.add(workspace)
        session.flush()
        root = SourceRoot(workspace_id=workspace.id, name="root", abs_path=str(tmp_path))
        session.add(root)
        session.flush()
        settings.workspace = workspace.name
        state = TagsEnv(
            settings=settings,
            fake=fake,
            root_id=root.id,
            workspace_id=workspace.id,
            root_path=tmp_path,
            chunks_alias=settings.opensearch.chunks_alias,
        )
    ensure_indices(get_client(), settings)
    ensure_pipeline(get_client(), settings)
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
        session.execute(delete(TagAlias).where(TagAlias.workspace_id == state.workspace_id))
        session.execute(delete(Tag).where(Tag.workspace_id == state.workspace_id))
        session.execute(delete(Job).where(Job.type.in_(_JOB_TYPES)))
        session.execute(delete(IndexOutbox).where(IndexOutbox.doc_kind == "chunk"))
        session.execute(delete(EmbeddingCache).where(EmbeddingCache.model == fake.model))
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
    worker = Worker(concurrency=1, poll_interval_s=0.05, types=["extract", "chunk"])
    while worker.run_once():
        pass


def drain_chunk() -> None:
    """Run the worker until no chunk job is claimable."""
    worker = Worker(concurrency=1, poll_interval_s=0.05, types=["chunk"])
    while worker.run_once():
        pass


def run_embed(env: TagsEnv, version_id: int) -> None:
    """Run the embed handler with the fake embedder, like the worker would."""
    pipeline.handle_embed(
        {"version_id": version_id},
        JobContext(worker_id="test", settings=env.settings, embedder=env.fake),
    )


def document_and_version(env: TagsEnv, rel_path: str) -> tuple[int, int]:
    """The document and latest version ids currently stored for ``rel_path``."""
    with session_scope() as session:
        file_row = session.execute(
            select(File).where(File.source_root_id == env.root_id, File.rel_path == rel_path)
        ).scalar_one()
        assert file_row.document_id is not None
        document = session.get(Document, file_row.document_id)
        assert document is not None and document.latest_version_id is not None
        return document.id, document.latest_version_id


def run_scan_handler(env: TagsEnv, job_type: str, document_id: int) -> None:
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


def ingest(env: TagsEnv, rel_path: str, content: str, mtime_ns: int = _MTIME) -> tuple[int, int]:
    """Write a file, scan, drain extract/chunk and embed its latest version."""
    write_file(env.root_path / rel_path, content, mtime_ns)
    scan_root(env.root_id)
    drain_extract_chunk()
    document_id, version_id = document_and_version(env, rel_path)
    run_embed(env, version_id)
    return document_id, version_id


def indexed_tags(env: TagsEnv) -> list[list[str]]:
    """The ``tags`` of every indexed chunk document after a refresh."""
    client = get_client()
    client.indices.refresh(index=env.chunks_alias)
    result = client.search(index=env.chunks_alias, body={"query": {"match_all": {}}, "size": 100})
    return [
        list(tags)
        for tags in sorted({tuple(hit["_source"]["tags"]) for hit in result["hits"]["hits"]})
    ]


def expected_count(env: TagsEnv) -> int:
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


def test_ingestion_persists_and_indexes_tag_sources(env: TagsEnv) -> None:
    """Folder rules, frontmatter and hashtags become accepted rows and index tags."""
    document_id, _version_id = ingest(env, "projects/alpha/note.md", _NOTE)
    sync_once()

    with session_scope() as session:
        rows = session.execute(
            select(Tag.name, DocumentTag.source)
            .join(DocumentTag, DocumentTag.tag_id == Tag.id)
            .where(DocumentTag.document_id == document_id)
            .order_by(Tag.name)
        ).all()
    assert sorted(rows) == [
        ("ideia", "hashtag"),
        ("ml", "frontmatter"),
        ("project:alpha", "folder"),
    ]

    tags = indexed_tags(env)
    assert tags and len(tags) == 1
    assert tags[0] == ["project:alpha", "ml", "ideia"]
    assert get_client().count(index=env.chunks_alias)["count"] == expected_count(env)


def test_tag_summaries_source_filter_lists_only_that_source(env: TagsEnv) -> None:
    """``--source`` keeps only tags applied by that source, with filtered counts."""
    ingest(env, "projects/alpha/note.md", _NOTE)
    with session_scope() as session:
        ensure_tag(session, env.workspace_id, "unused")

    with session_scope() as session:
        everything = {s.name: s for s in tag_summaries(session, env.workspace_id)}
        frontmatter = tag_summaries(session, env.workspace_id, source="frontmatter")
        folder = tag_summaries(session, env.workspace_id, source="folder")
        manual = tag_summaries(session, env.workspace_id, source="manual")

    assert everything["unused"].document_count == 0
    assert set(everything) >= {"ideia", "ml", "project:alpha", "unused"}
    assert [(s.name, s.document_count, s.sources) for s in frontmatter] == [
        ("ml", 1, ["frontmatter"])
    ]
    assert [s.name for s in folder] == ["project:alpha"]
    assert manual == []


def test_inline_tag_filter_matches_across_case(env: TagsEnv) -> None:
    """``#ML`` (uppercase) matches the indexed ``ml`` tag and only that document."""
    document_id, _version_id = ingest(env, "projects/alpha/note.md", _NOTE)
    ingest(env, "misc/outra.md", _OTHER_NOTE)
    sync_once()
    get_client().indices.refresh(index=env.chunks_alias)

    analysis = analyze_query("busca híbrida #ML")
    assert analysis.filters.tags == ["ml"]
    hits = hybrid_search(analysis.text, analysis.filters, size=10)

    assert hits
    assert {hit.document_id for hit in hits} == {document_id}


def test_merge_moves_rows_creates_alias_and_resolves_later_tags(env: TagsEnv) -> None:
    """``tags merge ml machine-learning`` rewrites rows, aliases and the index."""
    document_id, _version_id = ingest(env, "projects/alpha/note.md", _NOTE)
    sync_once()

    result = CliRunner().invoke(app, ["tags", "merge", "ml", "machine-learning"])
    assert result.exit_code == 0, result.output

    with session_scope() as session:
        names = {
            name
            for (name,) in session.execute(
                select(Tag.name).where(Tag.workspace_id == env.workspace_id)
            )
        }
        assert "ml" not in names and "machine-learning" in names
        alias_target = session.execute(
            select(Tag.name)
            .join(TagAlias, TagAlias.tag_id == Tag.id)
            .where(TagAlias.workspace_id == env.workspace_id, TagAlias.alias_norm == "ml")
        ).scalar_one()
        assert alias_target == "machine-learning"
        assert "machine-learning" in accepted_tags(session, document_id)
        assert "ml" not in accepted_tags(session, document_id)

    with session_scope() as session:
        set_document_tags(session, document_id, None, {"frontmatter": ["ml"]})
        tags = accepted_tags(session, document_id)
        names = {
            name
            for (name,) in session.execute(
                select(Tag.name).where(Tag.workspace_id == env.workspace_id)
            )
        }
    assert "machine-learning" in tags and "ml" not in tags
    assert "ml" not in names

    run_scan_handler(env, "reindex_metadata", document_id)
    sync_once()
    tags = indexed_tags(env)
    assert tags and len(tags) == 1
    assert tags[0] == ["project:alpha", "machine-learning", "ideia"]

    analysis = analyze_query("qualquer coisa #ML")
    hits = hybrid_search(analysis.text, analysis.filters, size=10)
    assert hits, "the ml alias must resolve to machine-learning at query time"


def test_move_replaces_folder_tags_and_reindexes(env: TagsEnv) -> None:
    """A move recomputes the folder tags, re-chunks and updates the index."""
    document_id, version_id = ingest(env, "projects/alpha/note.md", _NOTE)
    sync_once()
    with session_scope() as session:
        old_chunk_ids = {
            str(chunk_id)
            for chunk_id in session.execute(
                select(Chunk.id).where(Chunk.document_id == document_id)
            ).scalars()
        }
    assert old_chunk_ids

    (env.root_path / "projects" / "beta").mkdir()
    (env.root_path / "projects" / "alpha" / "note.md").rename(
        env.root_path / "projects" / "beta" / "note.md"
    )
    scan_root(env.root_id)
    run_scan_handler(env, "reindex_metadata", document_id)

    with session_scope() as session:
        assert accepted_tags(session, document_id) == ["project:beta", "ml", "ideia"]
        chunk_jobs = [
            job
            for job in session.execute(select(Job).where(Job.type == "chunk")).scalars()
            if job.payload == {"version_id": version_id} and job.status == "queued"
        ]
        assert chunk_jobs, "the re-tagging must enqueue a chunk job for the latest version"

    drain_chunk()
    with session_scope() as session:
        version = session.get(DocumentVersion, version_id)
        assert version is not None
        new_texts = [
            row[0]
            for row in session.execute(
                select(Chunk.embed_text).where(Chunk.document_id == document_id)
            ).all()
        ]
    assert new_texts and all("project:beta" in text for text in new_texts)
    run_embed(env, version_id)
    sync_once()

    tags = indexed_tags(env)
    assert tags and len(tags) == 1
    assert tags[0] == ["project:beta", "ml", "ideia"]
    assert get_client().count(index=env.chunks_alias)["count"] == expected_count(env)
    client = get_client()
    client.indices.refresh(index=env.chunks_alias)
    indexed_ids = {
        hit["_id"]
        for hit in client.search(index=env.chunks_alias, body={"query": {"match_all": {}}})["hits"][
            "hits"
        ]
    }
    assert indexed_ids.isdisjoint(old_chunk_ids)


def test_ensure_tag_survives_concurrent_workers(migrated_dsn: str) -> None:
    """Two workers creating the same new tag concurrently both succeed (task 51).

    Worker A inserts the tag and stays uncommitted; worker B's insert hits the
    unique constraint and, once A commits, must return A's row id instead of
    failing the chunk job with a ``UniqueViolation``. Before the task-51 fix
    (plain ORM insert after a select) worker B failed exactly that way.
    """
    from sqlalchemy import create_engine as sa_create_engine
    from sqlalchemy.orm import Session as SASession

    engine = sa_create_engine(migrated_dsn)
    try:
        with SASession(bind=engine.connect()) as bootstrap:
            workspace = Workspace(name="tags-race-51")
            bootstrap.add(workspace)
            bootstrap.commit()
            workspace_id = workspace.id

        session_a = SASession(bind=engine.connect())
        tag_id_a = ensure_tag(session_a, workspace_id, "project:raca")
        # session_a holds the tag insert uncommitted at this point.

        result: list[int] = []

        def worker_b() -> None:
            session_b = SASession(bind=engine.connect())
            try:
                result.append(ensure_tag(session_b, workspace_id, "project:raca"))
                session_b.commit()
            finally:
                session_b.close()

        thread = threading.Thread(target=worker_b, name="worker-b")
        thread.start()
        time.sleep(0.2)
        session_a.commit()
        thread.join(timeout=10)
        assert not thread.is_alive(), "worker B stayed blocked on worker A's insert"
        assert result == [tag_id_a]

        with SASession(bind=engine.connect()) as verify:
            rows = (
                verify.execute(
                    select(Tag).where(Tag.workspace_id == workspace_id, Tag.name == "project:raca")
                )
                .scalars()
                .all()
            )
        assert len(rows) == 1 and rows[0].id == tag_id_a

        session_a.close()
        with SASession(bind=engine.connect()) as cleanup:
            cleanup.execute(delete(Tag).where(Tag.workspace_id == workspace_id))
            workspace_row = cleanup.get(Workspace, workspace_id)
            assert workspace_row is not None
            cleanup.delete(workspace_row)
            cleanup.commit()
    finally:
        engine.dispose()
