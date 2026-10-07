"""Unit tests for watcher hint batching and the default batch handler (task 26).

Pure functions only: paths need not exist (``to_rel_path`` resolves
non-strictly), ``scan_root`` is replaced by a recorder, and the shared state
file is redirected into ``tmp_path``. No network, Docker or model endpoints.
"""

from __future__ import annotations

import json
import os
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from watchfiles import Change

from mynotes.config import get_settings, reset_settings_cache
from mynotes.scan.ignore import DEFAULT_IGNORES, IgnoreRules
from mynotes.scan.scanner import ScanSummary
from mynotes.scan.watch import (
    _PENDING,
    HintBatch,
    _defer_retry,
    _take_due_batch,
    batch_changes,
    handle_batch,
    watcher_status_rows,
)


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run with no MYNOTES_* variables, no user config/state dirs, clean retries."""
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("MYNOTES_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(
        "mynotes.config.platformdirs.user_config_dir",
        lambda *args, **kwargs: str(tmp_path / "user-config"),
    )
    monkeypatch.setattr(
        "mynotes.scan.watch.platformdirs.user_state_dir",
        lambda *args, **kwargs: str(tmp_path / "user-state"),
    )
    _PENDING.clear()
    reset_settings_cache()
    yield
    _PENDING.clear()
    reset_settings_cache()


def make_changes(root: Path, names: list[str], change: Change = Change.added):
    """Build ``(Change, path)`` pairs for names inside ``root``."""
    return [(change, str(root / name)) for name in names]


def test_rel_paths_posix_nfc_and_ignored_dropped(tmp_path: Path) -> None:
    nfd_name = unicodedata.normalize("NFD", "ação.md")
    changes = [
        (Change.added, str(tmp_path / "a.md")),
        (Change.modified, str(tmp_path / "~$x.docx")),
        (Change.added, str(tmp_path / "node_modules" / "pkg" / "x.js")),
        (Change.deleted, str(tmp_path / "gone.md")),
        (Change.added, str(tmp_path / nfd_name)),
    ]
    batch = batch_changes(tmp_path, 3, changes, IgnoreRules(tmp_path, DEFAULT_IGNORES), 100)

    assert batch is not None
    assert batch.root_id == 3
    assert batch.overflow is False
    assert batch.rel_paths == {"a.md", "gone.md", "ação.md"}
    assert all("\\" not in path for path in batch.rel_paths)


def test_deleted_directory_path_is_kept(tmp_path: Path) -> None:
    changes = [(Change.deleted, str(tmp_path / "sub" / "dir"))]
    batch = batch_changes(tmp_path, 1, changes, IgnoreRules(tmp_path, DEFAULT_IGNORES), 100)

    assert batch is not None
    assert batch.rel_paths == {"sub/dir"}


def test_batch_above_limit_overflows_and_empty_paths(tmp_path: Path) -> None:
    rules = IgnoreRules(tmp_path, DEFAULT_IGNORES)
    changes = make_changes(tmp_path, [f"f{i}.md" for i in range(6)])

    assert batch_changes(tmp_path, 1, changes, rules, max_batch_paths=5) == HintBatch(
        1, set(), overflow=True
    )

    batch = batch_changes(tmp_path, 1, changes[:5], rules, max_batch_paths=5)
    assert batch is not None
    assert batch.overflow is False
    assert len(batch.rel_paths) == 5


def test_outside_root_and_only_ignored_changes_yield_none(tmp_path: Path) -> None:
    rules = IgnoreRules(tmp_path, DEFAULT_IGNORES)
    changes: list[tuple[Change, str]] = [
        (Change.added, str(tmp_path.parent / "elsewhere.md")),
        (Change.modified, "~$relative.docx"),
        (Change.deleted, str(tmp_path / "node_modules" / "x.js")),
    ]
    assert batch_changes(tmp_path, 1, changes, rules, 10) is None


def test_empty_changes_yield_none(tmp_path: Path) -> None:
    assert batch_changes(tmp_path, 1, [], IgnoreRules(tmp_path, DEFAULT_IGNORES), 10) is None


@dataclass
class ScanRecorder:
    """Records ``scan_root`` calls; can pretend the advisory lock was busy."""

    skipped_locked: bool = False
    unsettled: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, root_id: int, **kwargs: Any) -> ScanSummary:
        self.calls.append({"root_id": root_id, **kwargs})
        if self.skipped_locked:
            return ScanSummary(scan_run_id=0, counts={"skipped_locked": 1})
        counts = {"unsettled": len(self.unsettled)} if self.unsettled else {}
        return ScanSummary(scan_run_id=1, counts=counts, unsettled=list(self.unsettled))


def test_handle_batch_overflow_runs_incremental_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = ScanRecorder()
    monkeypatch.setattr("mynotes.scan.watch.scan_root", recorder)

    handle_batch(HintBatch(7, set(), overflow=True))

    assert recorder.calls == [{"root_id": 7, "kind": "incremental", "trigger": "watch"}]


def test_handle_batch_runs_targeted_scan_with_sorted_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = ScanRecorder()
    monkeypatch.setattr("mynotes.scan.watch.scan_root", recorder)

    handle_batch(HintBatch(7, {"b.md", "a/c.md", "a.md"}))

    assert recorder.calls == [
        {
            "root_id": 7,
            "kind": "targeted",
            "trigger": "watch",
            "rel_paths": ["a.md", "a/c.md", "b.md"],
        }
    ]


def test_busy_lock_defers_merged_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = ScanRecorder(skipped_locked=True)
    monkeypatch.setattr("mynotes.scan.watch.scan_root", recorder)

    handle_batch(HintBatch(7, {"a.md"}))
    handle_batch(HintBatch(7, {"b.md"}, overflow=True))
    assert len(recorder.calls) == 2
    assert len(_PENDING) == 1

    pending = _PENDING[7]
    assert pending.rel_paths == {"a.md", "b.md"}
    assert pending.overflow is True
    assert pending.due > 0

    pending.due = 0.0
    assert _take_due_batch() == HintBatch(7, {"a.md", "b.md"}, overflow=True)
    assert _take_due_batch() is None


def test_unsettled_files_are_retried_after_settle_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """New files skipped as still being written are re-scanned once settled."""
    recorder = ScanRecorder(unsettled=["new.md"])
    monkeypatch.setattr("mynotes.scan.watch.scan_root", recorder)
    monkeypatch.setattr("mynotes.scan.watch.time.monotonic", lambda: 1000.0)

    handle_batch(HintBatch(7, {"new.md", "old.md"}))

    pending = _PENDING[7]
    assert pending.rel_paths == {"new.md"}
    assert pending.overflow is False
    settle = get_settings().scan.settle_seconds
    assert pending.due == 1000.0 + settle + 1.0

    recorder.unsettled = []
    pending.due = 0.0
    batch = _take_due_batch()
    assert batch == HintBatch(7, {"new.md"})
    handle_batch(batch)
    assert recorder.calls[-1]["rel_paths"] == ["new.md"]
    assert _PENDING == {}


def test_settled_scan_defers_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mynotes.scan.watch.scan_root", ScanRecorder())

    handle_batch(HintBatch(7, {"a.md"}))

    assert _PENDING == {}


def test_defer_retry_custom_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mynotes.scan.watch.time.monotonic", lambda: 50.0)
    _defer_retry(HintBatch(3, {"x.md"}), delay_s=11.0)
    assert _PENDING[3].due == 61.0


def test_defer_retry_keeps_earliest_due() -> None:
    _defer_retry(HintBatch(2, {"x.md"}))
    first_due = _PENDING[2].due
    _defer_retry(HintBatch(2, {"y.md"}))
    assert _PENDING[2].rel_paths == {"x.md", "y.md"}
    assert _PENDING[2].due == first_due


def test_status_rows_flag_stale_entries(tmp_path: Path) -> None:
    state_dir = tmp_path / "user-state"
    state_dir.mkdir(parents=True)
    now = datetime.now(UTC)
    payload = {
        "fresh": {"state": "native", "backend": "native", "since": now.isoformat()},
        "old": {
            "state": "polling",
            "backend": "polling",
            "since": (now - timedelta(minutes=5)).isoformat(),
        },
        "broken": {"state": "down", "backend": "native", "since": "not-a-date"},
    }
    (state_dir / "watchers.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = {row["root"]: row for row in watcher_status_rows(now=now)}

    assert rows["fresh"]["stale"] is False
    assert rows["fresh"]["state"] == "native"
    assert rows["old"]["stale"] is True
    assert rows["broken"]["stale"] is True


def test_status_rows_empty_when_state_file_missing(tmp_path: Path) -> None:
    assert watcher_status_rows() == []


def test_publish_states_round_trip(tmp_path: Path) -> None:
    from mynotes.scan import watch as watch_module

    class FakeWatcher:
        state = "native"
        backend = "polling"

    watch_module.WATCHERS["obsidian"] = FakeWatcher()  # type: ignore[assignment]
    try:
        watch_module.publish_states()
        payload = watch_module.read_watchers_state()
        assert set(payload) == {"obsidian"}
        assert payload["obsidian"]["state"] == "native"
        assert payload["obsidian"]["backend"] == "polling"
        assert watch_module._parse_since(payload["obsidian"]["since"]) is not None
    finally:
        watch_module.WATCHERS.clear()
