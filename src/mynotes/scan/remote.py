"""Manifests and uploads from agents: the scan of a remote root (server mode).

An agent lists its folder (``rel_path``, size, mtime, blake3 per file) and the
server classifies that listing exactly as a local scan classifies a walk
(:mod:`mynotes.scan.classify`), with two differences:

- paths are stored prefixed with the root's name (``alice-notes/ideas/a.md``),
  so the existing path ACL isolates one owner's notes from another's;
- a new or modified file whose bytes are not in the blob store yet is not
  written: it is returned in ``upload`` and the agent sends it with ``PUT``,
  after which the same classification runs for that one path.

Every write happens in one transaction under the root's blocking lock
(:func:`mynotes.scan.lock.root_xact_lock`), so uploads of one root queue up
behind each other and behind a running scan.
"""

from __future__ import annotations

import logging
import posixpath
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from mynotes.config import Settings
from mynotes.db.models import Document, DocumentVersion, File, ScanRun, SourceRoot
from mynotes.scan.classify import Action, Decision
from mynotes.scan.formats import format_for
from mynotes.scan.lock import root_xact_lock
from mynotes.scan.scanner import (
    apply_decisions_in_transaction,
    classify_for_root,
    load_known,
    tombstone,
)
from mynotes.scan.walker import FileStat
from mynotes.storage.blobs import BlobStore

logger = logging.getLogger(__name__)

Scope = Literal["full", "paths"]
_NEEDS_BYTES = frozenset({Action.NEW, Action.MODIFIED, Action.DUPLICATE})


@dataclass(frozen=True)
class ManifestEntry:
    """One file as the agent sees it (paths relative to the agent's folder)."""

    rel_path: str
    size: int
    mtime_ns: int
    blake3: str | None = None
    placeholder: bool = False


@dataclass
class ManifestResult:
    """What a manifest (or one upload) did."""

    scan_run_id: int
    counts: dict[str, int] = field(default_factory=dict)
    upload: list[str] = field(default_factory=list)
    actions: dict[str, str] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)


def valid_agent_path(rel_path: str) -> str:
    """Normalize a path from an agent (NFC, posix, no edge slashes); reject escapes.

    Raises:
        ValueError: For an empty, absolute or ``..``-containing path.
    """
    cleaned = unicodedata.normalize("NFC", str(rel_path).replace("\\", "/")).strip("/")
    if not cleaned or cleaned.startswith("/") or "\x00" in cleaned:
        raise ValueError(f"invalid path {rel_path!r}")
    parts = cleaned.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"invalid path {rel_path!r}")
    return posixpath.join(*parts)


def ingest_rel(root: SourceRoot, rel_path: str) -> str:
    """The path as stored for a remote root: the root's name in front."""
    return f"{root.name}/{valid_agent_path(rel_path)}"


def agent_rel(root: SourceRoot, stored_path: str) -> str:
    """The stored path as the agent knows it (the root's name removed)."""
    prefix = f"{root.name}/"
    return stored_path[len(prefix) :] if stored_path.startswith(prefix) else stored_path


def entry_stats(
    root: SourceRoot, entries: Sequence[ManifestEntry]
) -> tuple[list[FileStat], list[str]]:
    """Manifest entries as the walker's :class:`FileStat`; files of ignored extensions are skipped.

    Returns:
        ``(stats, skipped agent paths)``.
    """
    stats: list[FileStat] = []
    skipped: list[str] = []
    for entry in entries:
        fmt = format_for(entry.rel_path)
        if fmt is None:
            skipped.append(entry.rel_path)
            continue
        stored = ingest_rel(root, entry.rel_path)
        stats.append(
            FileStat(
                rel_path=stored,
                abs_path=Path("/remote") / stored,  # never read: the hash comes with the entry
                size=int(entry.size),
                mtime_ns=int(entry.mtime_ns),
                format=fmt,
                placeholder=bool(entry.placeholder),
                settled=True,
                content_hash=entry.blake3,
            )
        )
    return stats, skipped


def split_pending(
    decisions: Sequence[Decision], has_blob: Callable[[str, str], bool]
) -> tuple[list[Decision], list[Decision]]:
    """Decisions that can be applied now, and those waiting for the file's bytes.

    Args:
        decisions: The classifier's output.
        has_blob: ``(content_hash, format) -> bool``, whether the store holds the bytes.
    """
    ready: list[Decision] = []
    pending: list[Decision] = []
    for decision in decisions:
        if decision.action in _NEEDS_BYTES:
            stat = decision.stat
            digest = decision.content_hash
            if stat is None or digest is None or not has_blob(digest, stat.format):
                pending.append(decision)
                continue
        ready.append(decision)
    return ready, pending


def _no_hash_fn(path: Path) -> str:
    raise RuntimeError(f"manifest entries must carry their hash ({path})")


def apply_manifest(
    session: Session,
    root: SourceRoot,
    entries: Sequence[ManifestEntry],
    *,
    scope: Scope,
    paths: Sequence[str] | None,
    settings: Settings,
    store: BlobStore,
    now: datetime | None = None,
) -> ManifestResult:
    """Classify a manifest against the root's rows and persist what needs no bytes.

    Args:
        session: An open session; the caller commits (one transaction).
        root: The remote root.
        entries: The agent's listing (the whole folder, or the listed paths).
        scope: ``full`` compares with every known row; ``paths`` only with
            the given paths (plus missing rows, so renames pair up).
        paths: The agent paths a ``paths`` manifest covers.
        settings: Scan settings (tolerances, delete grace).
        store: The blob store, to know which bytes are present.
        now: The current time.
    """
    now = now or datetime.now(UTC)
    if scope == "paths" and not paths:
        raise ValueError("a paths manifest needs paths")
    root_xact_lock(session, int(root.id))
    run = ScanRun(
        source_root_id=root.id,
        kind="targeted" if scope == "paths" else "incremental",
        trigger="agent",
        counts={},
    )
    session.add(run)
    session.flush()
    on_disk, skipped = entry_stats(root, entries)
    requested = [ingest_rel(root, path) for path in paths] if scope == "paths" else None
    known = load_known(session, int(root.id), rel_paths=requested)
    decisions = classify_for_root(on_disk, known, settings, root.name, now=now)
    for decision in decisions:  # hashes always come from the entries, never from a file
        if decision.stat is not None and decision.content_hash is None:
            decision.content_hash = decision.stat.content_hash

    def has_blob(digest: str, fmt: str) -> bool:
        return store.exists(int(root.workspace_id), int(root.id), digest, fmt)

    ready, pending = split_pending(decisions, has_blob)
    counts = apply_decisions_in_transaction(
        session, ready, int(root.id), int(root.workspace_id), now, settings
    )
    counts["pending_upload"] = len(pending)
    session.execute(
        update(ScanRun)
        .where(ScanRun.id == run.id)
        .values(finished_at=func.now(), status="done", counts=counts)
    )
    result = ManifestResult(scan_run_id=int(run.id), counts=counts, skipped=skipped)
    result.upload = sorted(agent_rel(root, d.rel_path) for d in pending)
    result.actions = {agent_rel(root, d.rel_path): d.action.value for d in ready}
    for decision in pending:
        result.actions[agent_rel(root, decision.rel_path)] = "pending_upload"
    logger.info(
        "manifest for root %r (%s): %s, %d file(s) to upload",
        root.name,
        scope,
        counts,
        len(pending),
    )
    return result


def record_upload(
    session: Session,
    root: SourceRoot,
    entry: ManifestEntry,
    *,
    settings: Settings,
    store: BlobStore,
) -> str:
    """After the blob is stored: classify and persist that one path. Returns the action."""
    result = apply_manifest(
        session,
        root,
        [entry],
        scope="paths",
        paths=[entry.rel_path],
        settings=settings,
        store=store,
    )
    return result.actions.get(entry.rel_path, "unchanged")


def delete_path(
    session: Session,
    root: SourceRoot,
    rel_path: str,
    *,
    force: bool,
    settings: Settings,
    store: BlobStore,
) -> str:
    """Mark a path missing (delete grace applies), or tombstone it at once with ``force``."""
    result = apply_manifest(
        session, root, [], scope="paths", paths=[rel_path], settings=settings, store=store
    )
    action = result.actions.get(rel_path, "unchanged")
    if not force or action == Action.DELETED.value:
        return action
    stored = ingest_rel(root, rel_path)
    for known in load_known(session, int(root.id), rel_paths=[stored]):
        if known.rel_path == stored and known.status != "deleted":
            tombstone(session, known, datetime.now(UTC))
            return Action.DELETED.value
    return action


def move_path(
    session: Session,
    root: SourceRoot,
    old_rel_path: str,
    new_entry: ManifestEntry,
    *,
    settings: Settings,
    store: BlobStore,
) -> ManifestResult:
    """A rename: the old path is gone, the new one present with the same hash."""
    return apply_manifest(
        session,
        root,
        [new_entry],
        scope="paths",
        paths=[old_rel_path, new_entry.rel_path],
        settings=settings,
        store=store,
    )


def heartbeat(session: Session, root: SourceRoot, info: dict[str, Any]) -> None:
    """Record that the root's agent is alive, with what it reported about itself."""
    session.execute(
        update(SourceRoot)
        .where(SourceRoot.id == root.id)
        .values(agent_last_seen_at=datetime.now(UTC), agent_info=info)
    )


def root_status(session: Session, root: SourceRoot) -> dict[str, Any]:
    """Counts an agent (or an admin) wants to see about one remote root."""
    files = dict(
        session.execute(
            select(File.status, func.count())
            .where(File.source_root_id == root.id)
            .group_by(File.status)
        ).all()
    )
    versions = dict(
        session.execute(
            select(DocumentVersion.status, func.count())
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(Document.source_root_id == root.id, DocumentVersion.is_latest.is_(True))
            .group_by(DocumentVersion.status)
        ).all()
    )
    last = session.execute(
        select(ScanRun)
        .where(ScanRun.source_root_id == root.id)
        .order_by(ScanRun.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    return {
        "root": root.name,
        "files": {str(k): int(v) for k, v in files.items()},
        "latest_versions": {str(k): int(v) for k, v in versions.items()},
        "last_manifest": (
            {
                "scan_run_id": int(last.id),
                "kind": last.kind,
                "trigger": last.trigger,
                "status": last.status,
                "counts": dict(last.counts or {}),
                "started_at": last.started_at.isoformat() if last.started_at else None,
            }
            if last is not None
            else None
        ),
        "agent_last_seen_at": (
            root.agent_last_seen_at.isoformat() if root.agent_last_seen_at else None
        ),
        "agent_info": dict(root.agent_info or {}),
    }
