"""Unit tests for the pure change classification (task 16).

The fake ``hash_fn`` returns per-file digests from a dict and counts calls,
so tests can assert exactly when the scanner hashes and when it takes the
stat fast path.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NamedTuple

from mynotes.scan.classify import Action, Decision, KnownFile, classify
from mynotes.scan.hashing import FileLockedError
from mynotes.scan.walker import FileStat

NOW = datetime(2026, 9, 24, 12, 0, 0, tzinfo=UTC)
TOLERANCE_NS = 2_000_000_000
MTIME = 1_000_000_000_000
GRACE = timedelta(seconds=60)


class FakeHasher:
    """Hash function double: digests per file name, raising for locked files."""

    def __init__(self, digests: dict[str, str], locked: set[str] | None = None) -> None:
        self.digests = digests
        self.locked = locked or set()
        self.calls: list[str] = []

    def __call__(self, path: Path) -> str:
        self.calls.append(path.name)
        if path.name in self.locked:
            raise FileLockedError(path)
        return self.digests[path.name]


def stat(
    rel: str,
    *,
    size: int = 100,
    mtime_ns: int = MTIME,
    fmt: str = "md",
    placeholder: bool = False,
    settled: bool = True,
) -> FileStat:
    """Build a FileStat for ``rel`` with sensible defaults."""
    return FileStat(
        rel_path=rel,
        abs_path=Path("/data") / rel,
        size=size,
        mtime_ns=mtime_ns,
        format=fmt,
        placeholder=placeholder,
        settled=settled,
    )


def known(
    rel: str,
    *,
    id: int = 1,
    size: int = 100,
    mtime_ns: int = MTIME,
    content_hash: str | None = None,
    status: str = "active",
    missing_since: datetime | None = None,
    document_id: int | None = 100,
) -> KnownFile:
    """Build a KnownFile for ``rel`` with sensible defaults."""
    return KnownFile(
        id=id,
        rel_path=rel,
        size=size,
        mtime_ns=mtime_ns,
        content_hash="h-" + rel if content_hash is None else content_hash,
        status=status,
        missing_since=missing_since,
        document_id=document_id,
    )


class Result(NamedTuple):
    """Classification output plus the fake hasher's call log."""

    decisions: list[Decision]
    hash_calls: list[str]


def classify_one(on_disk: list[FileStat], rows: list[KnownFile], **kwargs: object) -> Result:
    """Run classify with standard test parameters and a per-name fake hasher."""
    override = kwargs.pop("digest", None)
    digests = {
        item.rel_path: "h-" + item.rel_path if override is None else str(override)
        for item in on_disk
    }
    hasher = FakeHasher(digests, locked=kwargs.pop("locked", None))
    decisions = classify(
        on_disk,
        rows,
        hash_fn=hasher,
        mtime_tolerance_ns=kwargs.pop("mtime_tolerance_ns", TOLERANCE_NS),
        hash_always=kwargs.pop("hash_always", False),
        delete_grace=kwargs.pop("delete_grace", GRACE),
        now=kwargs.pop("now", NOW),
        **kwargs,
    )
    return Result(decisions, hasher.calls)


def by_path(decisions: list[Decision]) -> dict[str, Decision]:
    """Map decisions by rel_path."""
    return {decision.rel_path: decision for decision in decisions}


def test_unchanged_by_stat_skips_hashing() -> None:
    """Same size and mtime within tolerance → UNCHANGED without hashing."""
    result = classify_one([stat("a.md")], [known("a.md")])

    assert len(result.decisions) == 1
    decision = result.decisions[0]
    assert decision.action is Action.UNCHANGED
    assert decision.known is not None and decision.known.rel_path == "a.md"
    assert decision.content_hash is None
    assert result.hash_calls == []


def test_hash_always_hashes_and_touched_when_same_hash() -> None:
    """hash_always=True skips the fast path; same hash → TOUCHED (stat only)."""
    result = classify_one([stat("a.md")], [known("a.md")], hash_always=True)

    decision = by_path(result.decisions)["a.md"]
    assert decision.action is Action.TOUCHED
    assert decision.content_hash == "h-a.md"
    assert result.hash_calls == ["a.md"]


def test_full_scan_also_hashes() -> None:
    """full_scan=True behaves like hash_always for the fast path."""
    result = classify_one([stat("a.md")], [known("a.md")], full_scan=True)

    assert by_path(result.decisions)["a.md"].action is Action.TOUCHED
    assert result.hash_calls == ["a.md"]


def test_modified_content() -> None:
    """A known path whose new hash differs → MODIFIED."""
    fresh = stat("a.md", mtime_ns=MTIME + 10 * TOLERANCE_NS)
    result = classify_one([fresh], [known("a.md")], digest="h-new")

    decision = by_path(result.decisions)["a.md"]
    assert decision.action is Action.MODIFIED
    assert decision.content_hash == "h-new"
    assert decision.known is not None and decision.known.id == 1


def test_rename_in_same_scan_is_moved_not_missing() -> None:
    """Old path gone + new path with the same hash → one MOVED, zero MISSING."""
    old = known("old.md", id=7, content_hash="h-same", document_id=500)
    result = classify_one([stat("new.md")], [old], digest="h-same")

    assert len(result.decisions) == 1
    decision = result.decisions[0]
    assert decision.action is Action.MOVED
    assert decision.rel_path == "new.md"
    assert decision.moved_from is not None and decision.moved_from.id == 7
    assert decision.content_hash == "h-same"
    assert not [item for item in result.decisions if item.action is Action.MISSING]


def test_rename_split_across_scans() -> None:
    """Scan 1 marks the old path MISSING; scan 2 moves the missing row."""
    first = classify_one([], [known("old.md", id=7, content_hash="h-same")])
    assert by_path(first.decisions)["old.md"].action is Action.MISSING

    missing = known(
        "old.md",
        id=7,
        content_hash="h-same",
        status="missing",
        missing_since=NOW - timedelta(seconds=30),
    )
    second = classify_one([stat("new.md")], [missing], digest="h-same")

    assert len(second.decisions) == 1
    decision = second.decisions[0]
    assert decision.action is Action.MOVED
    assert decision.moved_from is not None and decision.moved_from.id == 7
    assert not [item for item in second.decisions if item.action is Action.MISSING]


def test_missing_beyond_grace_is_deleted() -> None:
    """A missing row past delete_grace → DELETED."""
    row = known(
        "old.md",
        id=7,
        status="missing",
        missing_since=NOW - timedelta(hours=2),
    )
    result = classify_one([], [row], delete_grace=timedelta(hours=1))

    decision = by_path(result.decisions)["old.md"]
    assert decision.action is Action.DELETED
    assert decision.stat is None
    assert decision.known is not None and decision.known.id == 7


def test_missing_within_grace_stays_missing() -> None:
    """A missing row inside the grace period stays MISSING."""
    row = known(
        "old.md",
        id=7,
        status="missing",
        missing_since=NOW - timedelta(seconds=30),
    )
    result = classify_one([], [row])

    assert by_path(result.decisions)["old.md"].action is Action.MISSING


def test_copy_of_present_file_is_duplicate() -> None:
    """New path with the hash of a still-present file → DUPLICATE."""
    original = known("a.md", id=1, content_hash="h-same", document_id=100)
    result = classify_one([stat("a.md"), stat("b.md")], [original], digest="h-same")
    mapping = by_path(result.decisions)

    assert mapping["b.md"].action is Action.DUPLICATE
    assert mapping["b.md"].duplicate_of is not None and mapping["b.md"].duplicate_of.id == 1
    assert mapping["a.md"].action is Action.UNCHANGED
    assert result.hash_calls == ["b.md"]


def test_two_new_paths_with_one_missing_source() -> None:
    """One missing source, two new paths: first alphabetical moves, rest duplicates."""
    missing = known(
        "old.md",
        id=7,
        content_hash="h-same",
        status="missing",
        missing_since=NOW - timedelta(seconds=30),
        document_id=500,
    )
    result = classify_one([stat("p2.md"), stat("p1.md")], [missing], digest="h-same")
    mapping = by_path(result.decisions)

    assert mapping["p1.md"].action is Action.MOVED
    assert mapping["p1.md"].moved_from is not None and mapping["p1.md"].moved_from.id == 7
    assert mapping["p2.md"].action is Action.DUPLICATE
    assert mapping["p2.md"].duplicate_of is not None and mapping["p2.md"].duplicate_of.id == 7
    assert not [item for item in result.decisions if item.action is Action.MISSING]


def test_unsettled_known_file_is_not_missing() -> None:
    """A known path that is on disk but not settled → UNSETTLED, never MISSING."""
    result = classify_one([stat("a.md", settled=False)], [known("a.md")])

    assert len(result.decisions) == 1
    assert result.decisions[0].action is Action.UNSETTLED
    assert result.hash_calls == []
    assert not [item for item in result.decisions if item.action is Action.MISSING]


def test_unsupported_and_placeholder_are_not_hashed() -> None:
    """.doc → UNSUPPORTED; a zero-byte .pdf → PLACEHOLDER."""
    result = classify_one(
        [stat("notes.doc", fmt="unsupported"), stat("empty.pdf", size=0, placeholder=True)],
        [],
    )
    mapping = by_path(result.decisions)

    assert mapping["notes.doc"].action is Action.UNSUPPORTED
    assert mapping["empty.pdf"].action is Action.PLACEHOLDER
    assert result.hash_calls == []


def test_locked_file_is_locked_and_known_file_not_missing() -> None:
    """A file that cannot be read → LOCKED, and never marked missing."""
    result = classify_one([stat("a.md", size=200)], [known("a.md")], locked={"a.md"})

    assert by_path(result.decisions)["a.md"].action is Action.LOCKED
    assert not [item for item in result.decisions if item.action is Action.MISSING]


def test_deleted_known_path_reappearing_is_new() -> None:
    """A path whose row is deleted is treated as a new path."""
    row = known("a.md", id=7, content_hash="h-old", status="deleted", document_id=500)
    result = classify_one([stat("a.md")], [row], digest="h-new")

    decision = by_path(result.decisions)["a.md"]
    assert decision.action is Action.NEW
    assert decision.known is None
    assert decision.content_hash == "h-new"


def test_missing_file_reappearing_unchanged_is_unchanged() -> None:
    """A missing row whose path is back with the same stat → UNCHANGED.

    The persistence layer restores status/missing_since for such rows.
    """
    row = known(
        "a.md",
        id=7,
        status="missing",
        missing_since=NOW - timedelta(seconds=30),
    )
    result = classify_one([stat("a.md")], [row])

    decision = by_path(result.decisions)["a.md"]
    assert decision.action is Action.UNCHANGED
    assert result.hash_calls == []


def test_decisions_are_sorted_by_rel_path() -> None:
    """The returned decisions have a deterministic order."""
    rows = [
        known("a.md", id=1, status="missing", missing_since=NOW - timedelta(hours=2)),
        known("b.md", id=2),
        known("c.md", id=3),
    ]
    result = classify_one(
        [stat("b.md"), stat("z.md"), stat("a.md", fmt="unsupported")],
        rows,
        delete_grace=timedelta(hours=1),
    )

    assert [decision.rel_path for decision in result.decisions] == [
        "a.md",
        "b.md",
        "c.md",
        "z.md",
    ]
    assert [decision.action for decision in result.decisions] == [
        Action.UNSUPPORTED,
        Action.UNCHANGED,
        Action.MISSING,
        Action.NEW,
    ]
