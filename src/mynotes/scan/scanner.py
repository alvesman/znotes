"""Scan orchestration: classify changes against the database and persist them.

Implements PLAN.md §7.1 steps 4-5 on top of the pure classification in
:mod:`mynotes.scan.classify`. Each scan of a source root takes a Postgres
session-level advisory lock (PLAN.md §7.8), records a ``scan_run`` row, walks
or stats the root, classifies every file and applies the decisions
transactionally in batches, creating documents and versions and enqueueing
follow-up jobs. Handlers for the enqueued job types arrive in task 22.
"""

from __future__ import annotations

import logging
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from sqlalchemy import func, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session

from mynotes.config import Settings, get_settings
from mynotes.db.models import DocRelation, Document, DocumentVersion, File, ScanRun, SourceRoot
from mynotes.db.session import get_engine
from mynotes.jobs.queue import enqueue
from mynotes.jobs.registry import Priority
from mynotes.scan.classify import Action, Decision, KnownFile, classify
from mynotes.scan.hashing import hash_file
from mynotes.scan.ignore import IgnoreRules
from mynotes.scan.walker import FileStat, stat_paths, walk_root
from mynotes.text.normalize import normalize_path_key
from mynotes.versioning.retention import should_coalesce

logger = logging.getLogger(__name__)

_BATCH_SIZE = 200
_MAX_ERROR_LENGTH = 2_000
_NS_PER_MS = 1_000_000

ADVISORY_LOCK_SQL = "SELECT pg_try_advisory_lock(hashtext('scan:' || CAST(:rid AS bigint)))"
ADVISORY_UNLOCK_SQL = "SELECT pg_advisory_unlock(hashtext('scan:' || CAST(:rid AS bigint)))"
"""Session-level lock taken per source root so only one scan runs at a time."""

_KINDS = ("incremental", "full", "targeted")
_TRIGGERS = ("schedule", "manual", "watch", "startup")


@dataclass
class ScanSummary:
    """Outcome of one :func:`scan_root` call.

    Attributes:
        scan_run_id: The ``scan_run.id``, or 0 when no run was created
            (another scan of the same root held the advisory lock).
        counts: Number of decisions per action value; the skipped case
            reports ``{"skipped_locked": 1}``.
        errors: Messages of decision batches that failed and were skipped;
            the affected files are picked up by the next scan.
        unsettled: Rel paths skipped because they were modified less than
            ``settle_seconds`` ago; the watcher re-scans them once settled.
    """

    scan_run_id: int
    counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    unsettled: list[str] = field(default_factory=list)


def scan_root(
    source_root_id: int,
    *,
    kind: Literal["incremental", "full", "targeted"] = "incremental",
    trigger: Literal["schedule", "manual", "watch", "startup"] = "manual",
    rel_paths: Sequence[str] | None = None,
) -> ScanSummary:
    """Scan one source root and persist every change it finds.

    Args:
        source_root_id: The ``source_root.id`` to scan.
        kind: ``incremental`` walks the root and hashes changed files,
            ``full`` re-hashes everything, ``targeted`` stats only the given
            paths (plus all missing rows, so delayed renames are caught).
        trigger: What started the scan, recorded on the ``scan_run`` row.
        rel_paths: Relative paths for targeted scans; required for
            ``kind="targeted"``.

    Returns:
        A :class:`ScanSummary` with per-action counts. When another scan of
        the same root holds the advisory lock, the summary reports
        ``skipped_locked`` and nothing is written.

    Raises:
        ValueError: For an unknown kind/trigger or a targeted scan without
            paths, or when the source root does not exist.
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown scan kind: {kind!r}")
    if trigger not in _TRIGGERS:
        raise ValueError(f"unknown scan trigger: {trigger!r}")
    settings = get_settings()
    engine = get_engine()
    lock_connection = engine.connect()
    try:
        locked = lock_connection.execute(
            text(ADVISORY_LOCK_SQL), {"rid": source_root_id}
        ).scalar_one()
        if not locked:
            logger.info("scan of source root %d skipped: lock is held elsewhere", source_root_id)
            return ScanSummary(scan_run_id=0, counts={"skipped_locked": 1})
        lock_connection.commit()
        run_id: int | None = None
        try:
            summary, run_id = _run_scan(engine, source_root_id, settings, kind, trigger, rel_paths)
            return summary
        except BaseException as exc:
            _mark_run_failed(engine, run_id, exc)
            raise
        finally:
            _release_lock(lock_connection, source_root_id)
    finally:
        lock_connection.close()


def _run_scan(
    engine: Engine,
    source_root_id: int,
    settings: Settings,
    kind: str,
    trigger: str,
    rel_paths: Sequence[str] | None,
) -> tuple[ScanSummary, int]:
    """Run one scan: record the run, classify and apply decisions in batches."""
    now = datetime.now(UTC)
    scan_config = settings.scan
    session = Session(bind=engine)
    try:
        root_row = session.get(SourceRoot, source_root_id)
        if root_row is None:
            raise ValueError(f"source root {source_root_id} does not exist")
        root_id = root_row.id
        workspace_id = root_row.workspace_id
        root_name = root_row.name
        scan_run = ScanRun(source_root_id=root_id, kind=kind, trigger=trigger, counts={})
        session.add(scan_run)
        session.commit()
        run_id = scan_run.id

        on_disk, known = _collect(
            session, root_id, root_name, Path(root_row.abs_path), settings, kind, rel_paths
        )
        decisions = classify(
            on_disk,
            known,
            hash_fn=hash_file,
            mtime_tolerance_ns=scan_config.mtime_tolerance_ms * _NS_PER_MS,
            hash_always=_hash_always_for(settings, root_name),
            delete_grace=timedelta(seconds=scan_config.delete_grace_seconds),
            now=now,
            full_scan=(kind == "full"),
        )

        counts: dict[str, int] = {}
        errors: list[str] = []
        for batch in _chunks(decisions, _BATCH_SIZE):
            try:
                batch_counts = _apply_batch(session, batch, root_id, workspace_id, now, settings)
                session.commit()
            except Exception as exc:
                session.rollback()
                message = f"{type(exc).__name__}: {exc}"
                errors.append(message)
                logger.exception("scan batch failed for source root %d", root_id)
                continue
            for action, count in batch_counts.items():
                counts[action] = counts.get(action, 0) + count

        session.execute(
            update(ScanRun)
            .where(ScanRun.id == run_id)
            .values(finished_at=func.now(), status="done", counts=counts)
        )
        session.commit()
        if errors:
            logger.warning(
                "scan of source root %d finished with %d failed batch(es)", root_id, len(errors)
            )
        unsettled = [d.rel_path for d in decisions if d.action is Action.UNSETTLED]
        summary = ScanSummary(scan_run_id=run_id, counts=counts, errors=errors, unsettled=unsettled)
        return summary, run_id
    finally:
        session.close()


def _collect(
    session: Session,
    root_id: int,
    root_name: str,
    root_path: Path,
    settings: Settings,
    kind: str,
    rel_paths: Sequence[str] | None,
) -> tuple[list[FileStat], list[KnownFile]]:
    """Walk (or stat) the root and load the ``file`` rows to compare against."""
    rules = IgnoreRules(root_path, settings.scan.ignore)
    if kind == "targeted":
        if not rel_paths:
            raise ValueError("targeted scans require rel_paths")
        requested = [_normalize_rel(path) for path in rel_paths]
        on_disk, _missing = stat_paths(
            root_path, requested, rules, settle_seconds=settings.scan.settle_seconds
        )
        stmt = select(File).where(
            File.source_root_id == root_id,
            or_(File.rel_path.in_(requested), File.status == "missing"),
        )
    else:
        on_disk = list(
            walk_root(
                root_path,
                rules,
                settle_seconds=settings.scan.settle_seconds,
                skip_placeholders=settings.scan.skip_cloud_placeholders,
            )
        )
        stmt = select(File).where(File.source_root_id == root_id, File.status != "deleted")
    rows = session.execute(stmt).scalars().all()
    logger.debug(
        "scan of root '%s' (%s): %d file(s) on disk, %d known row(s)",
        root_name,
        kind,
        len(on_disk),
        len(rows),
    )
    known = [
        KnownFile(
            id=row.id,
            rel_path=row.rel_path,
            size=row.size,
            mtime_ns=row.mtime_ns,
            content_hash=row.content_hash,
            status=row.status,
            missing_since=row.missing_since,
            document_id=row.document_id,
        )
        for row in rows
    ]
    return on_disk, known


def _normalize_rel(path: str) -> str:
    """Normalize a requested path the same way the walker does."""
    return unicodedata.normalize("NFC", str(path).replace("\\", "/").strip("/"))


def _hash_always_for(settings: Settings, root_name: str) -> bool:
    """Resolve ``hash_always``: per-root YAML override, else the global value."""
    for root_config in settings.roots:
        if root_config.name == root_name and root_config.hash_always is not None:
            return root_config.hash_always
    return settings.scan.hash_always


def _chunks(decisions: list[Decision], size: int) -> Iterator[list[Decision]]:
    """Split the decisions into transaction-sized batches."""
    for start in range(0, len(decisions), size):
        yield decisions[start : start + size]


def _apply_batch(
    session: Session,
    batch: list[Decision],
    root_id: int,
    workspace_id: int,
    now: datetime,
    settings: Settings,
) -> dict[str, int]:
    """Apply one batch of decisions in the caller's transaction."""
    counts: dict[str, int] = {}
    unchanged = [decision for decision in batch if decision.action is Action.UNCHANGED]
    if unchanged:
        ids = [decision.known.id for decision in unchanged]
        session.execute(update(File).where(File.id.in_(ids)).values(last_seen_at=now))
        was_missing = [
            decision.known.id for decision in unchanged if decision.known.status == "missing"
        ]
        if was_missing:
            session.execute(
                update(File)
                .where(File.id.in_(was_missing))
                .values(status="active", missing_since=None)
            )
    for decision in batch:
        if decision.action is not Action.UNCHANGED:
            _apply_decision(session, decision, root_id, workspace_id, now, settings)
        counts[decision.action.value] = counts.get(decision.action.value, 0) + 1
    return counts


def _apply_decision(
    session: Session,
    decision: Decision,
    root_id: int,
    workspace_id: int,
    now: datetime,
    settings: Settings,
) -> None:
    """Persist one decision (task 16 persistence table)."""
    action = decision.action
    if action is Action.TOUCHED:
        values: dict[str, object] = {
            "size": decision.stat.size,
            "mtime_ns": decision.stat.mtime_ns,
            "last_seen_at": now,
        }
        if decision.known.status == "missing":
            values.update(status="active", missing_since=None)
        session.execute(update(File).where(File.id == decision.known.id).values(**values))
    elif action is Action.MODIFIED:
        _apply_modified(session, decision, root_id, workspace_id, now, settings)
    elif action in (Action.NEW, Action.DUPLICATE):
        _apply_new_file(session, decision, root_id, workspace_id)
    elif action is Action.MOVED:
        _apply_moved(session, decision, now)
    elif action is Action.MISSING:
        if decision.known.status == "active":
            session.execute(
                update(File)
                .where(File.id == decision.known.id)
                .values(status="missing", missing_since=now)
            )
    elif action is Action.DELETED:
        _apply_deleted(session, decision, now)
    elif action in (Action.UNSUPPORTED, Action.PLACEHOLDER):
        _upsert_non_document_file(session, decision, root_id, now)
    # UNSETTLED and LOCKED are retried on the next scan; nothing to write.


def _apply_modified(
    session: Session,
    decision: Decision,
    root_id: int,
    workspace_id: int,
    now: datetime,
    settings: Settings,
) -> None:
    """Update the file row and record (or coalesce into) a version of its document.

    An edit within ``versions.min_interval_hours`` of the latest snapshot
    reuses that version row (task 33): content hash, mtime and path are
    updated, the pipeline status resets to ``pending`` and ``extract`` runs
    again; the ``chunk`` handler replaces the version's old chunks. Otherwise
    a new version is appended as before.
    """
    stat = decision.stat
    known = decision.known
    session.execute(
        update(File)
        .where(File.id == known.id)
        .values(
            size=stat.size,
            mtime_ns=stat.mtime_ns,
            content_hash=decision.content_hash,
            last_seen_at=now,
            status="active",
            missing_since=None,
        )
    )
    if known.document_id is None:
        # A placeholder/unsupported row that became readable: give it a document.
        _insert_document_and_version(
            session, root_id, workspace_id, file_id=known.id, decision=decision, version_no=1
        )
        return
    previous = (
        session.execute(
            select(DocumentVersion).where(
                DocumentVersion.document_id == known.document_id,
                DocumentVersion.is_latest.is_(True),
            )
        )
        .scalars()
        .first()
    )
    if previous is not None and should_coalesce(previous, now, settings):
        previous.content_hash = decision.content_hash
        previous.mtime_ns = stat.mtime_ns
        previous.rel_path_at_version = decision.rel_path
        previous.status = "pending"
        previous.ir_json = None
        previous.error = None
        version = previous
    else:
        if previous is not None:
            previous.is_latest = False
        max_no = session.execute(
            select(func.max(DocumentVersion.version_no)).where(
                DocumentVersion.document_id == known.document_id
            )
        ).scalar_one()
        version = DocumentVersion(
            document_id=known.document_id,
            version_no=(max_no or 0) + 1,
            content_hash=decision.content_hash,
            rel_path_at_version=decision.rel_path,
            mtime_ns=stat.mtime_ns,
            status="pending",
            is_latest=True,
        )
        session.add(version)
        session.flush()
        session.execute(
            update(Document)
            .where(Document.id == known.document_id)
            .values(latest_version_id=version.id)
        )
    enqueue(
        session,
        "extract",
        {"version_id": version.id},
        dedupe_key=f"extract:{version.id}",
        priority=Priority.EXTRACT,
    )


def _apply_new_file(
    session: Session,
    decision: Decision,
    root_id: int,
    workspace_id: int,
) -> None:
    """Insert the file row, a document and version 1; DUPLICATE adds a relation."""
    stat = decision.stat
    file_row = File(
        source_root_id=root_id,
        rel_path=decision.rel_path,
        rel_path_norm=normalize_path_key(decision.rel_path),
        size=stat.size,
        mtime_ns=stat.mtime_ns,
        content_hash=decision.content_hash,
        format=stat.format,
        status="active",
    )
    session.add(file_row)
    session.flush()
    document = _insert_document_and_version(
        session, root_id, workspace_id, file_id=file_row.id, decision=decision, version_no=1
    )
    if decision.action is Action.DUPLICATE:
        _insert_duplicate_relation(session, document.id, decision.duplicate_of)


def _insert_document_and_version(
    session: Session,
    root_id: int,
    workspace_id: int,
    *,
    file_id: int,
    decision: Decision,
    version_no: int,
) -> Document:
    """Create a document and one pending version for a file, then enqueue extract."""
    document = Document(
        workspace_id=workspace_id,
        source_root_id=root_id,
        format=decision.stat.format,
        title=None,
        status="current",
    )
    session.add(document)
    session.flush()
    document.current_file_id = file_id
    version = DocumentVersion(
        document_id=document.id,
        version_no=version_no,
        content_hash=decision.content_hash,
        rel_path_at_version=decision.rel_path,
        mtime_ns=decision.stat.mtime_ns,
        status="pending",
        is_latest=True,
    )
    session.add(version)
    session.flush()
    document.latest_version_id = version.id
    session.execute(update(File).where(File.id == file_id).values(document_id=document.id))
    enqueue(
        session,
        "extract",
        {"version_id": version.id},
        dedupe_key=f"extract:{version.id}",
        priority=Priority.EXTRACT,
    )
    return document


def _insert_duplicate_relation(
    session: Session, document_id: int, duplicate_of: KnownFile | None
) -> None:
    """Link a duplicate document to the original by identical content hash."""
    if duplicate_of is None or duplicate_of.document_id is None:
        return
    session.execute(
        pg_insert(DocRelation)
        .values(
            src_document_id=document_id,
            dst_document_id=duplicate_of.document_id,
            type="duplicate_of",
            method="hash",
            confidence=1.0,
            status="auto",
        )
        .on_conflict_do_nothing(index_elements=["src_document_id", "dst_document_id", "type"])
    )


def _apply_moved(session: Session, decision: Decision, now: datetime) -> None:
    """Repoint the moved-from file row to the new path; reindex metadata only."""
    stat = decision.stat
    source = decision.moved_from
    session.execute(
        update(File)
        .where(File.id == source.id)
        .values(
            rel_path=decision.rel_path,
            rel_path_norm=normalize_path_key(decision.rel_path),
            size=stat.size,
            mtime_ns=stat.mtime_ns,
            format=stat.format,
            status="active",
            missing_since=None,
            last_seen_at=now,
        )
    )
    if source.document_id is not None:
        enqueue(
            session,
            "reindex_metadata",
            {"document_id": source.document_id},
            dedupe_key=f"reindex_metadata:{source.document_id}",
            priority=Priority.EMBED,
        )


def _apply_deleted(session: Session, decision: Decision, now: datetime) -> None:
    """Tombstone the file row and its document, then enqueue the index delete."""
    known = decision.known
    session.execute(
        update(File).where(File.id == known.id).values(status="deleted", deleted_at=now)
    )
    if known.document_id is not None:
        session.execute(
            update(Document).where(Document.id == known.document_id).values(status="deleted")
        )
        enqueue(
            session,
            "delete_document",
            {"document_id": known.document_id},
            dedupe_key=f"delete_document:{known.document_id}",
            priority=Priority.EMBED,
        )


def _upsert_non_document_file(
    session: Session,
    decision: Decision,
    root_id: int,
    now: datetime,
) -> None:
    """Insert or update a file row recorded without a document."""
    stat = decision.stat
    if decision.known is not None:
        session.execute(
            update(File)
            .where(File.id == decision.known.id)
            .values(
                status=decision.action.value,
                size=stat.size,
                mtime_ns=stat.mtime_ns,
                format=stat.format,
                last_seen_at=now,
                missing_since=None,
            )
        )
        return
    session.add(
        File(
            source_root_id=root_id,
            rel_path=decision.rel_path,
            rel_path_norm=normalize_path_key(decision.rel_path),
            size=stat.size,
            mtime_ns=stat.mtime_ns,
            content_hash=None,
            format=stat.format,
            status=decision.action.value,
        )
    )


def _mark_run_failed(engine: Engine, run_id: int | None, exc: BaseException) -> None:
    """Record the failure on the ``scan_run`` row; never masks the original error."""
    if run_id is None:
        return
    try:
        with Session(bind=engine) as session:
            session.execute(
                update(ScanRun)
                .where(ScanRun.id == run_id)
                .values(
                    finished_at=func.now(),
                    status="failed",
                    error=str(exc)[:_MAX_ERROR_LENGTH],
                )
            )
            session.commit()
    except Exception:
        logger.exception("could not mark scan run %d as failed", run_id)


def _release_lock(lock_connection: Connection, source_root_id: int) -> None:
    """Release the session advisory lock, tolerating a broken connection."""
    try:
        lock_connection.execute(text(ADVISORY_UNLOCK_SQL), {"rid": source_root_id})
        lock_connection.commit()
    except Exception:
        logger.warning("could not release the scan lock of root %d", source_root_id)
