"""Integration tests for the scanner's classification persistence (task 16).

Each test registers a temporary directory as a ``source_root`` in the
session-scoped Postgres container, runs ``scan_root`` against real files and
inspects the committed rows. ``scan_root`` manages its own transactions, so
setup and assertions go through ``session_scope`` (committed data) and every
test cleans up its rows afterwards.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select, text
from sqlalchemy.orm import Session

from mynotes.config import Settings
from mynotes.db.models import (
    Document,
    DocumentVersion,
    File,
    Job,
    ScanRun,
    SourceRoot,
    Tag,
    TagAlias,
    Workspace,
)
from mynotes.db.session import get_engine, session_scope
from mynotes.jobs.registry import Priority
from mynotes.scan.scanner import (
    ADVISORY_LOCK_SQL,
    ADVISORY_UNLOCK_SQL,
    ScanSummary,
    scan_root,
)
from mynotes.text.normalize import normalize_path_key

pytestmark = pytest.mark.integration

_MTIME = 1_000_000_000_000_000_000
_JOB_TYPES = (
    "extract",
    "reindex_metadata",
    "delete_document",
    "version_detect",
    "tag_llm",
    "link_resolve",
    "kg_extract",
)


@dataclass
class ScanEnv:
    """The registered test root plus the settings the scanner will read."""

    settings: Settings
    root_id: int
    workspace_id: int
    root_path: Path


@pytest.fixture
def scan_env(
    settings_with_db: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[ScanEnv]:
    """Container settings with a fast settle time and a registered temp root."""
    settings_with_db.scan.settle_seconds = 0
    monkeypatch.setattr("mynotes.scan.scanner.get_settings", lambda: settings_with_db)
    with session_scope() as session:
        workspace = Workspace(name=f"scan-{uuid4().hex[:8]}")
        session.add(workspace)
        session.flush()
        root = SourceRoot(workspace_id=workspace.id, name="root", abs_path=str(tmp_path))
        session.add(root)
        session.flush()
        env = ScanEnv(
            settings=settings_with_db,
            root_id=root.id,
            workspace_id=workspace.id,
            root_path=tmp_path,
        )
    yield env
    with session_scope() as session:
        session.execute(delete(Job).where(Job.type.in_(_JOB_TYPES)))
        session.execute(delete(File).where(File.source_root_id == env.root_id))
        session.execute(delete(Document).where(Document.source_root_id == env.root_id))
        session.execute(delete(ScanRun).where(ScanRun.source_root_id == env.root_id))
        session.execute(delete(SourceRoot).where(SourceRoot.id == env.root_id))
        session.execute(delete(TagAlias).where(TagAlias.workspace_id == env.workspace_id))
        session.execute(delete(Tag).where(Tag.workspace_id == env.workspace_id))
        session.execute(delete(Workspace).where(Workspace.id == env.workspace_id))


def write_file(path: Path, content: str, mtime_ns: int = _MTIME) -> None:
    """Create a file with a fixed mtime so scans are deterministic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))


def rows(session: Session, model: Any, **filters: Any) -> list[Any]:
    """Fetch all rows of ``model`` matching the filters, ordered by id."""
    stmt = select(model).filter_by(**filters).order_by(model.id)
    return list(session.execute(stmt).scalars().all())


def all_jobs(session: Session) -> list[Job]:
    """The pipeline jobs created by the scanner."""
    return [job for job in rows(session, Job) if job.type in _JOB_TYPES]


def test_first_scan_creates_files_documents_versions_and_jobs(scan_env: ScanEnv) -> None:
    """Scanning a fresh root registers every file as NEW with one version each."""
    write_file(scan_env.root_path / "a.md", "# A\n")
    write_file(scan_env.root_path / "b.md", "# B\n longer content\n")
    write_file(scan_env.root_path / "sub" / "c.md", "C\n")

    summary = scan_root(scan_env.root_id)

    assert summary.errors == []
    assert summary.counts == {"new": 3}
    with session_scope() as session:
        files = rows(session, File, source_root_id=scan_env.root_id)
        assert [item.rel_path for item in files] == ["a.md", "b.md", "sub/c.md"]
        assert all(item.status == "active" and item.document_id for item in files)
        assert all(item.rel_path_norm == normalize_path_key(item.rel_path) for item in files)
        documents = rows(session, Document, source_root_id=scan_env.root_id)
        assert len(documents) == 3
        assert all(item.status == "current" and item.title is None for item in documents)
        assert all(item.current_file_id and item.latest_version_id for item in documents)
        versions = rows(session, DocumentVersion)
        assert len(versions) == 3
        assert all(item.version_no == 1 and item.is_latest for item in versions)
        assert all(item.status == "pending" for item in versions)
        jobs = all_jobs(session)
        assert len(jobs) == 3
        assert all(job.type == "extract" and job.status == "queued" for job in jobs)
        assert all(job.priority == Priority.EXTRACT for job in jobs)
        assert {job.dedupe_key for job in jobs} == {f"extract:{item.id}" for item in versions}
        run = rows(session, ScanRun, source_root_id=scan_env.root_id)[0]
        assert run.status == "done" and run.kind == "incremental" and run.trigger == "manual"
        assert run.counts == {"new": 3} and run.finished_at is not None


def test_second_scan_without_changes_is_all_unchanged(scan_env: ScanEnv) -> None:
    """A repeated scan takes the stat fast path and writes nothing new."""
    write_file(scan_env.root_path / "a.md", "# A\n")
    write_file(scan_env.root_path / "b.md", "# B\n")
    scan_root(scan_env.root_id)

    summary = scan_root(scan_env.root_id)

    assert summary.counts == {"unchanged": 2}
    with session_scope() as session:
        assert len(rows(session, Document, source_root_id=scan_env.root_id)) == 2
        assert len(rows(session, DocumentVersion)) == 2
        assert len(all_jobs(session)) == 2


def test_targeted_scan_reports_unsettled_paths(scan_env: ScanEnv) -> None:
    """A file modified within ``settle_seconds`` is skipped and listed for the watcher."""
    write_file(scan_env.root_path / "old.md", "# Old\n")
    fresh = scan_env.root_path / "fresh.md"
    fresh.write_text("# Fresh\n", encoding="utf-8")  # mtime = now
    scan_env.settings.scan.settle_seconds = 60

    summary = scan_root(scan_env.root_id, kind="targeted", rel_paths=["fresh.md", "old.md"])

    assert summary.counts == {"new": 1, "unsettled": 1}
    assert summary.unsettled == ["fresh.md"]
    with session_scope() as session:
        files = rows(session, File, source_root_id=scan_env.root_id)
        assert [item.rel_path for item in files] == ["old.md"]


def test_edited_file_creates_version_two(scan_env: ScanEnv) -> None:
    """Editing content → MODIFIED: version 2, v1 not latest, one extra extract."""
    write_file(scan_env.root_path / "a.md", "# A\n")
    write_file(scan_env.root_path / "b.md", "# B\n")
    scan_root(scan_env.root_id)
    write_file(
        scan_env.root_path / "a.md",
        "# A\n edited with a different length\n",
        _MTIME + 5_000_000_000,
    )

    summary = scan_root(scan_env.root_id)

    assert summary.counts == {"modified": 1, "unchanged": 1}
    with session_scope() as session:
        file_row = session.execute(select(File).where(File.rel_path == "a.md")).scalar_one()
        document = session.get(Document, file_row.document_id)
        assert document is not None
        versions = rows(session, DocumentVersion, document_id=document.id)
        assert len(versions) == 2
        assert [item.version_no for item in versions] == [1, 2]
        assert [item.is_latest for item in versions] == [False, True]
        assert versions[1].rel_path_at_version == "a.md"
        assert document.latest_version_id == versions[1].id
        file_row = session.execute(select(File).where(File.rel_path == "a.md")).scalar_one()
        assert file_row.content_hash == versions[1].content_hash
        jobs = all_jobs(session)
        assert len(jobs) == 3
        assert {job.dedupe_key for job in jobs} == {
            f"extract:{item.id}" for item in rows(session, DocumentVersion)
        }


def test_rename_moves_the_document(scan_env: ScanEnv) -> None:
    """A rename → MOVED: same document, metadata reindex, no re-extraction."""
    write_file(scan_env.root_path / "a.md", "# A\n")
    write_file(scan_env.root_path / "b.md", "# B\n")
    scan_root(scan_env.root_id)
    with session_scope() as session:
        file_row = session.execute(select(File).where(File.rel_path == "b.md")).scalar_one()
        document_id, file_id = file_row.document_id, file_row.id
    (scan_env.root_path / "b.md").rename(scan_env.root_path / "renamed.md")

    summary = scan_root(scan_env.root_id)

    assert summary.counts == {"moved": 1, "unchanged": 1}
    with session_scope() as session:
        moved = session.get(File, file_id)
        assert moved is not None
        assert moved.rel_path == "renamed.md"
        assert moved.rel_path_norm == normalize_path_key("renamed.md")
        assert moved.status == "active" and moved.missing_since is None
        assert moved.document_id == document_id
        assert (
            session.execute(
                select(func.count()).select_from(File).where(File.rel_path == "b.md")
            ).scalar_one()
            == 0
        )
        jobs = all_jobs(session)
        assert [job.type for job in jobs] == ["extract", "extract", "reindex_metadata"]
        reindex = jobs[-1]
        assert reindex.priority == Priority.EMBED
        assert reindex.payload == {"document_id": document_id}
        assert reindex.dedupe_key == f"reindex_metadata:{document_id}"


def test_deleted_file_after_grace(scan_env: ScanEnv) -> None:
    """With grace 0: one scan marks MISSING, the next DELETED with a job."""
    scan_env.settings.scan.delete_grace_seconds = 0
    write_file(scan_env.root_path / "a.md", "# A\n")
    write_file(scan_env.root_path / "b.md", "# B\n")
    scan_root(scan_env.root_id)
    with session_scope() as session:
        file_row = session.execute(select(File).where(File.rel_path == "b.md")).scalar_one()
        document_id, file_id = file_row.document_id, file_row.id
    (scan_env.root_path / "b.md").unlink()

    missing_scan = scan_root(scan_env.root_id)
    assert missing_scan.counts == {"missing": 1, "unchanged": 1}
    with session_scope() as session:
        file_row = session.get(File, file_id)
        assert file_row is not None and file_row.status == "missing"
        assert file_row.missing_since is not None
        assert session.get(Document, document_id).status == "current"

    deleted_scan = scan_root(scan_env.root_id)
    assert deleted_scan.counts == {"deleted": 1, "unchanged": 1}
    with session_scope() as session:
        file_row = session.get(File, file_id)
        assert file_row is not None
        assert file_row.status == "deleted" and file_row.deleted_at is not None
        assert session.get(Document, document_id).status == "deleted"
        deletes = [job for job in all_jobs(session) if job.type == "delete_document"]
        assert len(deletes) == 1
        assert deletes[0].payload == {"document_id": document_id}
        assert deletes[0].dedupe_key == f"delete_document:{document_id}"
        assert deletes[0].priority == Priority.EMBED


def test_concurrent_scans_only_one_runs(scan_env: ScanEnv) -> None:
    """A scan that cannot take the advisory lock reports skipped_locked."""
    write_file(scan_env.root_path / "a.md", "# A\n")
    write_file(scan_env.root_path / "b.md", "# B\n")
    holder = get_engine().connect()
    try:
        assert holder.execute(text(ADVISORY_LOCK_SQL), {"rid": scan_env.root_id}).scalar_one()

        results: dict[str, ScanSummary] = {}

        def worker() -> None:
            results["thread"] = scan_root(scan_env.root_id)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        skipped = results["thread"]
        assert skipped.counts == {"skipped_locked": 1}
        assert skipped.scan_run_id == 0
        with session_scope() as session:
            assert (
                session.execute(
                    select(func.count())
                    .select_from(ScanRun)
                    .where(ScanRun.source_root_id == scan_env.root_id)
                ).scalar_one()
                == 0
            )

        holder.execute(text(ADVISORY_UNLOCK_SQL), {"rid": scan_env.root_id})
        holder.commit()
    finally:
        holder.close()

    summary = scan_root(scan_env.root_id)
    assert summary.counts == {"new": 2}
    with session_scope() as session:
        runs = rows(session, ScanRun, source_root_id=scan_env.root_id)
        assert len(runs) == 1 and runs[0].status == "done"
