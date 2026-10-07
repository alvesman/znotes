"""Pure change classification for the scanner (PLAN.md §7.1 step 4).

Compares the :class:`~mynotes.scan.walker.FileStat` snapshot of what is on
disk with the ``file`` rows recorded by the previous scan and decides one
:class:`Action` per file. The module is pure: hashing is injected and no
database is touched (persistence lives in :mod:`mynotes.scan.scanner`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path

from mynotes.scan.hashing import FileLockedError
from mynotes.scan.walker import FileStat

logger = logging.getLogger(__name__)

_MOVE_SOURCE_STATUSES = frozenset({"active", "missing"})


@dataclass(frozen=True)
class KnownFile:
    """Snapshot of one ``file`` row from the previous scan.

    Attributes:
        id: The ``file.id`` primary key.
        rel_path: Path relative to the source root, POSIX separators, NFC.
        size: File size in bytes at the last scan.
        mtime_ns: Modification time in nanoseconds at the last scan.
        content_hash: Content hash at the last scan (None for rows that were
            never hashed, e.g. placeholders and unsupported formats).
        status: Row status (``active``, ``missing``, ``unsupported``, ...).
        missing_since: When the file was first reported missing.
        document_id: The document this file belongs to, if any.
    """

    id: int
    rel_path: str
    size: int
    mtime_ns: int
    content_hash: str | None
    status: str
    missing_since: datetime | None
    document_id: int | None


class Action(StrEnum):
    """Possible per-file outcomes of a scan (PLAN.md §7.1)."""

    UNCHANGED = "unchanged"
    TOUCHED = "touched"
    MODIFIED = "modified"
    NEW = "new"
    MOVED = "moved"
    DUPLICATE = "duplicate"
    MISSING = "missing"
    DELETED = "deleted"
    UNSUPPORTED = "unsupported"
    PLACEHOLDER = "placeholder"
    UNSETTLED = "unsettled"
    LOCKED = "locked"


@dataclass
class Decision:
    """One classification result for one path.

    Attributes:
        action: The decided action.
        rel_path: Path relative to the source root, POSIX separators, NFC.
        stat: The on-disk snapshot (None for MISSING/DELETED decisions).
        known: The matched known row, when the path existed before.
        content_hash: The hash computed by this scan, when one was computed.
        moved_from: For MOVED, the known row the file was moved from.
        duplicate_of: For DUPLICATE, the known file with the same content.
    """

    action: Action
    rel_path: str
    stat: FileStat | None = None
    known: KnownFile | None = None
    content_hash: str | None = None
    moved_from: KnownFile | None = None
    duplicate_of: KnownFile | None = None


def classify(
    on_disk: list[FileStat],
    known: list[KnownFile],
    *,
    hash_fn: Callable[[Path], str],
    mtime_tolerance_ns: int,
    hash_always: bool,
    delete_grace: timedelta,
    now: datetime,
    full_scan: bool = False,
) -> list[Decision]:
    """Compare the on-disk snapshot with the known file rows.

    The rules are applied in the order given by the task specification:
    unsupported/placeholder/unsettled files are never hashed, known paths
    take the stat fast path or are re-hashed, and new paths are matched
    against known files that are no longer on disk (moves) before being
    classified as duplicates or new documents.

    Args:
        on_disk: Files seen by the walker (order does not matter).
        known: ``file`` rows of the previous scan; rows with status
            ``deleted`` are treated as unknown paths.
        hash_fn: Content hash function; a :class:`FileLockedError` (or any
            other read error) turns into a LOCKED decision.
        mtime_tolerance_ns: Allowed ``|mtime difference|`` for the stat fast
            path.
        hash_always: Skip the stat fast path (roots with unreliable mtimes).
        delete_grace: How long a file stays ``missing`` before it is deleted.
        now: Current time, used for grace comparisons and ``missing_since``.
        full_scan: True for full verification scans (always hash).

    Returns:
        One :class:`Decision` per on-disk file and per known file that is
        neither on disk nor a move source, sorted by ``rel_path``.
    """
    live_known = [row for row in known if row.status != "deleted"]
    known_by_path = {row.rel_path: row for row in live_known}
    on_disk_paths = {stat.rel_path for stat in on_disk}

    decisions: list[Decision] = []
    new_entries: list[tuple[FileStat, str]] = []
    for stat in sorted(on_disk, key=lambda item: item.rel_path):
        row = known_by_path.get(stat.rel_path)
        if stat.format == "unsupported":
            decisions.append(Decision(Action.UNSUPPORTED, stat.rel_path, stat=stat, known=row))
            continue
        if stat.placeholder:
            decisions.append(Decision(Action.PLACEHOLDER, stat.rel_path, stat=stat, known=row))
            continue
        if not stat.settled:
            decisions.append(Decision(Action.UNSETTLED, stat.rel_path, stat=stat, known=row))
            continue
        if row is not None:
            _classify_known(
                decisions,
                stat,
                row,
                hash_fn=hash_fn,
                mtime_tolerance_ns=mtime_tolerance_ns,
                hash_always=hash_always,
                full_scan=full_scan,
            )
        else:
            digest = _hash_stat(stat, None, hash_fn, decisions)
            if digest is not None:
                new_entries.append((stat, digest))

    _classify_new_paths(decisions, new_entries, live_known, on_disk_paths)
    _classify_vanished(decisions, live_known, on_disk_paths, delete_grace=delete_grace, now=now)
    decisions.sort(key=lambda item: item.rel_path)
    return decisions


def _classify_known(
    decisions: list[Decision],
    stat: FileStat,
    row: KnownFile,
    *,
    hash_fn: Callable[[Path], str],
    mtime_tolerance_ns: int,
    hash_always: bool,
    full_scan: bool,
) -> None:
    """Classify one on-disk file that has a known row (rules 2 and 3)."""
    fast_path = (
        not hash_always
        and not full_scan
        and stat.size == row.size
        and abs(stat.mtime_ns - row.mtime_ns) <= mtime_tolerance_ns
    )
    if fast_path:
        decisions.append(Decision(Action.UNCHANGED, stat.rel_path, stat=stat, known=row))
        return
    digest = _hash_stat(stat, row, hash_fn, decisions)
    if digest is None:
        return
    if row.content_hash == digest:
        decisions.append(
            Decision(Action.TOUCHED, stat.rel_path, stat=stat, known=row, content_hash=digest)
        )
    else:
        decisions.append(
            Decision(Action.MODIFIED, stat.rel_path, stat=stat, known=row, content_hash=digest)
        )


def _hash_stat(
    stat: FileStat,
    known: KnownFile | None,
    hash_fn: Callable[[Path], str],
    decisions: list[Decision],
) -> str | None:
    """Hash one file, appending a LOCKED decision when it cannot be read.

    Returns:
        The digest, or None when a LOCKED decision was recorded instead.
    """
    try:
        return hash_fn(stat.abs_path)
    except FileLockedError:
        decisions.append(Decision(Action.LOCKED, stat.rel_path, stat=stat, known=known))
        return None
    except OSError as exc:
        logger.warning("Cannot read %s during scan: %s", stat.abs_path, exc)
        decisions.append(Decision(Action.LOCKED, stat.rel_path, stat=stat, known=known))
        return None


def _classify_new_paths(
    decisions: list[Decision],
    new_entries: list[tuple[FileStat, str]],
    live_known: list[KnownFile],
    on_disk_paths: set[str],
) -> None:
    """Classify paths without a known row as MOVED, DUPLICATE or NEW (rule 4)."""
    sources_by_hash: dict[str, list[KnownFile]] = {}
    present_by_hash: dict[str, KnownFile] = {}
    for row in sorted(live_known, key=lambda item: item.rel_path):
        if not row.content_hash:
            continue
        if row.status in _MOVE_SOURCE_STATUSES and row.rel_path not in on_disk_paths:
            sources_by_hash.setdefault(row.content_hash, []).append(row)
        elif row.rel_path in on_disk_paths:
            present_by_hash.setdefault(row.content_hash, row)

    new_by_hash: dict[str, list[FileStat]] = {}
    for stat, digest in new_entries:
        new_by_hash.setdefault(digest, []).append(stat)

    used_source_ids: set[int] = set()
    for digest, stats in new_by_hash.items():
        stats.sort(key=lambda item: item.rel_path)
        sources = sources_by_hash.get(digest, [])
        present = present_by_hash.get(digest)
        fallback: KnownFile | None = None
        for stat in stats:
            if sources:
                source = sources.pop(0)
                used_source_ids.add(source.id)
                if fallback is None:
                    fallback = source
                decisions.append(
                    Decision(
                        Action.MOVED,
                        stat.rel_path,
                        stat=stat,
                        content_hash=digest,
                        moved_from=source,
                    )
                )
            elif fallback is not None or present is not None:
                decisions.append(
                    Decision(
                        Action.DUPLICATE,
                        stat.rel_path,
                        stat=stat,
                        content_hash=digest,
                        duplicate_of=present if present is not None else fallback,
                    )
                )
            else:
                decisions.append(
                    Decision(Action.NEW, stat.rel_path, stat=stat, content_hash=digest)
                )


def _classify_vanished(
    decisions: list[Decision],
    live_known: list[KnownFile],
    on_disk_paths: set[str],
    *,
    delete_grace: timedelta,
    now: datetime,
) -> None:
    """Mark known files that are gone as MISSING or DELETED (rule 5)."""
    moved_ids = {
        decision.moved_from.id for decision in decisions if decision.moved_from is not None
    }
    for row in sorted(live_known, key=lambda item: item.rel_path):
        if row.rel_path in on_disk_paths or row.id in moved_ids:
            continue
        if row.status == "active":
            decisions.append(Decision(Action.MISSING, row.rel_path, known=row))
        elif row.status == "missing":
            expired = row.missing_since is not None and now - row.missing_since >= delete_grace
            action = Action.DELETED if expired else Action.MISSING
            decisions.append(Decision(action, row.rel_path, known=row))
