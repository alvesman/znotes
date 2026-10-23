"""Unit tests for the pure parts of the agent manifest handling."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mynotes.scan import remote
from mynotes.scan.classify import Action, Decision
from mynotes.scan.walker import FileStat

ROOT = SimpleNamespace(id=7, workspace_id=1, name="alice-notes")


def test_paths_are_validated_normalized_and_prefixed() -> None:
    assert remote.valid_agent_path("ideas\\a.md") == "ideas/a.md"
    assert remote.valid_agent_path("/ideas/a.md/") == "ideas/a.md"
    for bad in ("", "..", "a/../b", "./a", "a//b"):
        with pytest.raises(ValueError):
            remote.valid_agent_path(bad)
    assert remote.ingest_rel(ROOT, "ideas/a.md") == "alice-notes/ideas/a.md"
    assert remote.agent_rel(ROOT, "alice-notes/ideas/a.md") == "ideas/a.md"
    assert remote.agent_rel(ROOT, "other/x.md") == "other/x.md"


def test_entry_stats_detect_formats_and_skip_unknown_extensions() -> None:
    entries = [
        remote.ManifestEntry("ideas/a.md", 10, 5, "ab" * 32),
        remote.ManifestEntry("old.doc", 10, 5, "cd" * 32),
        remote.ManifestEntry("photo.jpg", 10, 5, "ef" * 32),
        remote.ManifestEntry("cloud.pdf", 0, 5, None, placeholder=True),
    ]
    stats, skipped = remote.entry_stats(ROOT, entries)
    assert skipped == ["photo.jpg"]
    by_path = {stat.rel_path: stat for stat in stats}
    assert by_path["alice-notes/ideas/a.md"].format == "md"
    assert by_path["alice-notes/ideas/a.md"].content_hash == "ab" * 32
    assert by_path["alice-notes/old.doc"].format == "unsupported"
    assert by_path["alice-notes/cloud.pdf"].placeholder is True
    assert all(stat.settled for stat in stats)


def _decision(action: Action, rel: str, digest: str | None, fmt: str = "md") -> Decision:
    stat = FileStat(rel, Path("/remote") / rel, 1, 1, fmt, False, True, digest)
    return Decision(action, rel, stat=stat, content_hash=digest)


def test_split_pending_holds_back_files_without_bytes() -> None:
    decisions = [
        _decision(Action.NEW, "r/a.md", "aa" * 32),
        _decision(Action.MODIFIED, "r/b.md", "bb" * 32),
        _decision(Action.DUPLICATE, "r/c.md", "aa" * 32),
        Decision(Action.MISSING, "r/d.md"),
        _decision(Action.TOUCHED, "r/e.md", None),
    ]
    ready, pending = remote.split_pending(decisions, lambda digest, fmt: digest == "aa" * 32)
    assert [d.rel_path for d in pending] == ["r/b.md"]
    assert [d.rel_path for d in ready] == ["r/a.md", "r/c.md", "r/d.md", "r/e.md"]
