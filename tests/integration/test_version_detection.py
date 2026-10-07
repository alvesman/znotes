"""Integration tests for successor detection (task 34).

Each test registers a temporary directory as a ``source_root``, installs files
from the ``tests/fixtures/trees/versions/`` fixture tree and runs the real
pipeline: scan, extract/chunk through the worker, ``embed`` with a
``FakeEmbedder`` (dim 8), OpenSearch sync through the outbox and the
``version_detect`` handler. Handlers commit through ``session_scope``, so
setup and assertions do too, and every test cleans up its rows and indices.
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

from fakes import FakeEmbedder
from mynotes.config import Settings
from mynotes.db.models import (
    DocRelation,
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
from mynotes.index.opensearch import ensure_indices, get_client
from mynotes.index.sync import sync_once
from mynotes.jobs import pipeline
from mynotes.jobs.registry import JobContext
from mynotes.jobs.worker import Worker
from mynotes.scan.scanner import scan_root

pytestmark = pytest.mark.integration

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "trees" / "versions"
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
    "kg_temporal_refresh",
)


@dataclass
class DetectionEnv:
    """The registered test root, the fake embedder and the test's aliases."""

    settings: Settings
    fake: FakeEmbedder
    root_id: int
    workspace_id: int
    root_path: Path
    chunks_alias: str


@pytest.fixture
def detection_env(
    migrated_dsn: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    os_container: OpenSearchContainer,
) -> Iterator[DetectionEnv]:
    """Container settings (dim 8, fast settle) and per-test OpenSearch aliases."""
    settings = Settings(postgres={"dsn": migrated_dsn})
    config = os_container.get_config()
    suffix = uuid4().hex[:8]
    settings.opensearch.url = f"http://{config['host']}:{config['port']}"
    settings.opensearch.chunks_alias = f"t{suffix}_chunks"
    settings.opensearch.entities_alias = f"t{suffix}_entities"
    settings.models.embedding.dim = 8
    settings.scan.settle_seconds = 0
    settings.versions.min_interval_hours = 0.0
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
        workspace = Workspace(name=f"det-{suffix}")
        session.add(workspace)
        session.flush()
        root = SourceRoot(workspace_id=workspace.id, name="root", abs_path=str(tmp_path))
        session.add(root)
        session.flush()
        env = DetectionEnv(
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


def fixture(name: str) -> str:
    """The content of one fixture-tree file."""
    return (_FIXTURES / name).read_text(encoding="utf-8")


def install(env: DetectionEnv, rel_path: str, source: str, mtime_ns: int = _MTIME) -> None:
    """Write a fixture file into the root with a fixed mtime."""
    path = env.root_path / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fixture(source), encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))


def drain(*job_types: str) -> None:
    """Run the worker until no job of the given types is claimable."""
    worker = Worker(concurrency=1, poll_interval_s=0.05, types=list(job_types))
    while worker.run_once():
        pass


def ingest(env: DetectionEnv, rel_path: str, source: str, mtime_ns: int = _MTIME) -> int:
    """Install a fixture file and run it through scan/extract/chunk/embed.

    Returns:
        The id of the new document's latest version.
    """
    install(env, rel_path, source, mtime_ns)
    scan_root(env.root_id)
    drain("extract", "chunk")
    version_id = latest_version_id(env, rel_path)
    pipeline.handle_embed(
        {"version_id": version_id},
        JobContext(worker_id="test", settings=env.settings, embedder=env.fake),
    )
    drain("apply_retention")
    return version_id


def latest_version_id(env: DetectionEnv, rel_path: str) -> int:
    """The latest version id of the document stored for ``rel_path``."""
    with session_scope() as session:
        file_row = session.execute(
            select(File).where(File.source_root_id == env.root_id, File.rel_path == rel_path)
        ).scalar_one()
        assert file_row.document_id is not None
        document = session.get(Document, file_row.document_id)
        assert document is not None and document.latest_version_id is not None
        return int(document.latest_version_id)


def document_id_of(env: DetectionEnv, rel_path: str) -> int:
    """The document id stored for ``rel_path`` (0 when there is none)."""
    with session_scope() as session:
        file_row = session.execute(
            select(File).where(File.source_root_id == env.root_id, File.rel_path == rel_path)
        ).scalar_one_or_none()
        if file_row is None or file_row.document_id is None:
            return 0
        return int(file_row.document_id)


def document_row(env: DetectionEnv, document_id: int) -> Document:
    """A detached document row for assertions."""
    with session_scope() as session:
        document = session.get(Document, document_id)
        assert document is not None
        session.expunge(document)
        return document


def supersedes_relations(env: DetectionEnv) -> list[DocRelation]:
    """Every supersedes relation in the workspace, detached, ordered by id."""
    with session_scope() as session:
        relations = list(
            session.execute(select(DocRelation).where(DocRelation.type == "supersedes")).scalars()
        )
        for relation in relations:
            session.expunge(relation)
        return relations


def indexed_doc_statuses(env: DetectionEnv, document_id: int) -> set[str]:
    """The ``doc_status`` values indexed for one document, after a refresh."""
    client = get_client()
    client.indices.refresh(index=env.chunks_alias)
    body = {
        "query": {"term": {"document_id": str(document_id)}},
        "_source": ["doc_status"],
        "size": 100,
    }
    result = client.search(index=env.chunks_alias, body=body)
    return {hit["_source"]["doc_status"] for hit in result["hits"]["hits"]}


def test_new_version_file_supersedes_and_reindexes_status(
    detection_env: DetectionEnv,
) -> None:
    """idea_v2.md (80% same text) auto-supersedes idea.md end to end."""
    env = detection_env
    old_version = ingest(env, "idea.md", "idea.md")
    new_version = ingest(env, "idea_v2.md", "idea_v2.md", _MTIME + 1_000_000_000)
    sync_once()
    old_doc = document_id_of(env, "idea.md")
    new_doc = document_id_of(env, "idea_v2.md")
    assert old_doc and new_doc and old_doc != new_doc
    assert indexed_doc_statuses(env, old_doc) == {"current"}

    drain("version_detect")
    drain("reindex_metadata")
    sync_once()

    relations = supersedes_relations(env)
    assert len(relations) == 1
    relation = relations[0]
    assert relation.src_document_id == new_doc
    assert relation.dst_document_id == old_doc
    assert relation.status == "auto"
    assert relation.method == "name_content"
    assert relation.confidence >= env.settings.versions.auto_threshold
    assert relation.evidence["order_reason"] == "version_number"
    assert relation.evidence["name_signal"] is True

    old_row = document_row(env, old_doc)
    assert old_row.status == "superseded"
    assert old_row.superseded_by_document_id == new_doc
    new_row = document_row(env, new_doc)
    assert new_row.status == "current"
    with session_scope() as session:
        assert session.get(DocumentVersion, old_version).is_latest is True
        assert session.get(DocumentVersion, new_version).is_latest is True
    assert indexed_doc_statuses(env, old_doc) == {"superseded"}
    assert indexed_doc_statuses(env, new_doc) == {"current"}


def test_dated_copies_supersede_by_doc_date(detection_env: DetectionEnv) -> None:
    """plan-2026-09.md (no version marker) wins over plan-2026-03.md by date."""
    env = detection_env
    ingest(env, "plan-2026-03.md", "plan-2026-03.md")
    ingest(env, "plan-2026-09.md", "plan-2026-09.md", _MTIME + 1_000_000_000)
    march_doc = document_id_of(env, "plan-2026-03.md")
    september_doc = document_id_of(env, "plan-2026-09.md")

    drain("version_detect")
    drain("reindex_metadata")

    relations = supersedes_relations(env)
    assert len(relations) == 1
    assert relations[0].src_document_id == september_doc
    assert relations[0].dst_document_id == march_doc
    assert relations[0].status == "auto"
    assert relations[0].evidence["order_reason"] == "doc_date"
    assert document_row(env, march_doc).status == "superseded"


def test_archive_move_supersedes_with_bonus(detection_env: DetectionEnv) -> None:
    """The old file moved into archive/ supersedes with the archive bonus."""
    env = detection_env
    ingest(env, "plano.md", "plano.md")
    ingest(env, "plano_v2.md", "plano_v2.md", _MTIME + 1_000_000_000)
    old_doc = document_id_of(env, "plano.md")
    new_doc = document_id_of(env, "plano_v2.md")
    (env.root_path / "archive").mkdir()
    os.rename(env.root_path / "plano.md", env.root_path / "archive" / "plano.md")
    scan_root(env.root_id)
    drain("extract", "chunk")
    drain("version_detect")
    drain("reindex_metadata")

    relations = supersedes_relations(env)
    assert len(relations) == 1
    assert relations[0].src_document_id == new_doc
    assert relations[0].dst_document_id == old_doc
    assert relations[0].status == "auto"
    assert relations[0].evidence["archive_move"] is True
    assert relations[0].evidence["order_reason"] == "version_number"
    assert document_row(env, old_doc).status == "superseded"


def test_identical_copy_is_duplicate_without_supersedes(
    detection_env: DetectionEnv,
) -> None:
    """notes (1).md is a scanner duplicate; identical hashes never supersede."""
    env = detection_env
    ingest(env, "notes.md", "notes.md")
    install(env, "notes (1).md", "notes.md", _MTIME + 2_000_000_000)
    scan_root(env.root_id)
    drain("extract", "chunk")
    drain("version_detect")

    relations = supersedes_relations(env)
    assert relations == []
    first_doc = document_id_of(env, "notes.md")
    copy_doc = document_id_of(env, "notes (1).md")
    assert first_doc and copy_doc and first_doc != copy_doc
    assert document_row(env, first_doc).status == "current"
    assert document_row(env, copy_doc).status == "current"
    with session_scope() as session:
        duplicates = list(
            session.execute(
                select(DocRelation).where(
                    DocRelation.src_document_id == copy_doc,
                    DocRelation.type == "duplicate_of",
                )
            ).scalars()
        )
        assert len(duplicates) == 1


def test_unrelated_versions_get_no_relation(detection_env: DetectionEnv) -> None:
    """report_v2.md about another topic stays below the propose threshold."""
    env = detection_env
    ingest(env, "report.md", "report.md")
    ingest(env, "report_v2.md", "report_v2.md", _MTIME + 1_000_000_000)
    drain("version_detect")

    assert supersedes_relations(env) == []
    assert document_row(env, document_id_of(env, "report.md")).status == "current"
    assert document_row(env, document_id_of(env, "report_v2.md")).status == "current"


def test_frontmatter_supersedes_creates_confirmed_relation(
    detection_env: DetectionEnv,
) -> None:
    """better-idea.md wins with confidence 1.0 despite low similarity."""
    env = detection_env
    ingest(env, "idea.md", "idea.md")
    ingest(env, "better-idea.md", "better-idea.md", _MTIME + 1_000_000_000)
    drain("reindex_metadata")
    sync_once()
    old_doc = document_id_of(env, "idea.md")
    new_doc = document_id_of(env, "better-idea.md")
    assert indexed_doc_statuses(env, old_doc) == {"current"}

    drain("version_detect")
    drain("reindex_metadata")
    sync_once()

    relations = supersedes_relations(env)
    assert len(relations) == 1
    assert relations[0].src_document_id == new_doc
    assert relations[0].dst_document_id == old_doc
    assert relations[0].status == "confirmed"
    assert relations[0].method == "frontmatter"
    assert relations[0].confidence == 1.0
    assert document_row(env, old_doc).status == "superseded"
    assert document_row(env, old_doc).superseded_by_document_id == new_doc
    assert indexed_doc_statuses(env, old_doc) == {"superseded"}


def test_rejected_relation_is_not_recreated(detection_env: DetectionEnv) -> None:
    """A user rejection survives later detection runs untouched."""
    env = detection_env
    ingest(env, "idea.md", "idea.md")
    ingest(env, "idea_v2.md", "idea_v2.md", _MTIME + 1_000_000_000)
    old_doc = document_id_of(env, "idea.md")
    new_doc = document_id_of(env, "idea_v2.md")
    with session_scope() as session:
        session.add(
            DocRelation(
                src_document_id=new_doc,
                dst_document_id=old_doc,
                type="supersedes",
                method="name_content",
                confidence=0.9,
                status="rejected",
            )
        )

    drain("version_detect")
    drain("reindex_metadata")

    relations = supersedes_relations(env)
    assert len(relations) == 1
    assert relations[0].status == "rejected"
    assert relations[0].confidence == 0.9
    assert document_row(env, old_doc).status == "current"


def test_move_plus_edit_merges_identity(detection_env: DetectionEnv) -> None:
    """Renaming a.md to a-renamed.md with an edit keeps one document."""
    env = detection_env
    env.settings.scan.delete_grace_seconds = 0
    first_version = ingest(env, "a.md", "a.md")
    old_doc = document_id_of(env, "a.md")

    install(env, "a-renamed.md", "a-renamed.md", _MTIME + 2_000_000_000)
    (env.root_path / "a.md").unlink()
    scan_root(env.root_id)
    drain("extract", "chunk")
    new_version = latest_version_id(env, "a-renamed.md")
    pipeline.handle_embed(
        {"version_id": new_version},
        JobContext(worker_id="test", settings=env.settings, embedder=env.fake),
    )
    new_doc_id = document_id_of(env, "a-renamed.md")

    pipeline.handle_version_detect(
        {"document_id": new_doc_id},
        JobContext(worker_id="test", settings=env.settings, embedder=env.fake),
    )

    assert supersedes_relations(env) == []
    merged_doc = document_id_of(env, "a-renamed.md")
    assert merged_doc == old_doc
    assert new_doc_id != old_doc
    with session_scope() as session:
        assert session.get(Document, new_doc_id) is None
        versions = list(
            session.execute(
                select(DocumentVersion)
                .where(DocumentVersion.document_id == merged_doc)
                .order_by(DocumentVersion.version_no)
            ).scalars()
        )
        assert [version.version_no for version in versions] == [1, 2]
        assert [version.is_latest for version in versions] == [False, True]
        assert versions[0].id == first_version
        assert versions[1].id == new_version
        assert versions[1].rel_path_at_version == "a-renamed.md"
        moved_file = session.execute(
            select(File).where(File.source_root_id == env.root_id, File.rel_path == "a-renamed.md")
        ).scalar_one()
        assert moved_file.document_id == merged_doc
        assert moved_file.status == "active"
        old_file = session.execute(
            select(File).where(File.source_root_id == env.root_id, File.rel_path == "a.md")
        ).scalar_one()
        assert old_file.status == "deleted"
        document = session.get(Document, merged_doc)
        assert document is not None
        assert document.status == "current"
        assert document.current_file_id == moved_file.id
        assert document.latest_version_id == new_version
    drain("reindex_metadata")
    drain("delete_document")
    sync_once()
    client = get_client()
    client.indices.refresh(index=env.chunks_alias)
    body = {"query": {"match_all": {}}, "_source": ["document_id", "version_id"], "size": 100}
    hits = client.search(index=env.chunks_alias, body=body)["hits"]["hits"]
    assert {hit["_source"]["document_id"] for hit in hits} == {str(merged_doc)}
    assert {hit["_source"]["version_id"] for hit in hits} == {str(new_version)}


def test_concurrent_upsert_of_the_same_supersedes_relation(
    detection_env: DetectionEnv, migrated_dsn: str
) -> None:
    """Two version_detect jobs writing the same relation at once both succeed.

    The two documents of a pair are detected by two jobs that can run on two
    worker threads at the same time. Session A inserts the relation and stays
    uncommitted; session B's insert waits on the unique index and, once A
    commits, must update A's row instead of failing with a UniqueViolation.
    """
    from sqlalchemy import create_engine as sa_create_engine
    from sqlalchemy.orm import Session as SASession

    from mynotes.versioning.detector import _upsert_supersedes

    note = "---\ntitle: Plano Raca\ndate: {date}\n---\n# Plano Raca\n\n{body}\n"
    for rel_path, date, body, mtime_ns in (
        ("raca/plano.md", "2025-01-01", "Texto um.", _MTIME),
        ("raca/plano-v2.md", "2026-01-01", "Texto dois.", _MTIME + 10**9),
    ):
        path = detection_env.root_path / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(note.format(date=date, body=body), encoding="utf-8")
        os.utime(path, ns=(mtime_ns, mtime_ns))
        scan_root(detection_env.root_id)
        drain("extract", "chunk")
        pipeline.handle_embed(
            {"version_id": latest_version_id(detection_env, rel_path)},
            JobContext(
                worker_id="test", settings=detection_env.settings, embedder=detection_env.fake
            ),
        )
        drain("apply_retention")
    older = document_id_of(detection_env, "raca/plano.md")
    newer = document_id_of(detection_env, "raca/plano-v2.md")
    assert older and newer

    engine = sa_create_engine(migrated_dsn)
    try:
        session_a = SASession(bind=engine.connect())
        id_a = _upsert_supersedes(
            session_a,
            src=newer,
            dst=older,
            method="name_content",
            confidence=0.9,
            status="auto",
            evidence={"by": "a"},
        )
        assert id_a is not None  # inserted, not yet committed

        result: list[int | None] = []

        def worker_b() -> None:
            session_b = SASession(bind=engine.connect())
            try:
                result.append(
                    _upsert_supersedes(
                        session_b,
                        src=newer,
                        dst=older,
                        method="name_content",
                        confidence=0.95,
                        status="auto",
                        evidence={"by": "b"},
                    )
                )
                session_b.commit()
            finally:
                session_b.close()

        thread = threading.Thread(target=worker_b, name="worker-b")
        thread.start()
        time.sleep(0.2)
        session_a.commit()
        thread.join(timeout=10)
        assert not thread.is_alive(), "worker B stayed blocked on worker A's insert"
        assert result == [id_a]

        with SASession(bind=engine.connect()) as verify:
            rows = (
                verify.execute(
                    select(DocRelation).where(
                        DocRelation.src_document_id == newer, DocRelation.dst_document_id == older
                    )
                )
                .scalars()
                .all()
            )
        assert len(rows) == 1 and rows[0].id == id_a
        assert rows[0].confidence == pytest.approx(0.95) and rows[0].evidence == {"by": "b"}
        session_a.close()
        with SASession(bind=engine.connect()) as cleanup:
            cleanup.execute(delete(DocRelation).where(DocRelation.id == id_a))
            cleanup.commit()
    finally:
        engine.dispose()
