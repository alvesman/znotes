"""File-watcher integration tests on the real filesystem (task 26).

No Docker and no database: the watcher runs against a temporary root with a
fake ``on_batch`` collector. Marked ``integration`` because it uses real
threads and wall-clock time. Generous timeouts keep it stable on Linux,
macOS and Windows.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from mynotes.config import Settings, reset_settings_cache
from mynotes.scan.watch import HintBatch, RootWatcher

pytestmark = [pytest.mark.integration, pytest.mark.watcher]

_NATIVE_TIMEOUT_S = 10.0
_BATCH_TIMEOUT_S = 5.0
_RECOVER_TIMEOUT_S = 15.0


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No MYNOTES_* variables, no user config/state dirs outside tmp_path."""
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
    reset_settings_cache()
    yield
    reset_settings_cache()


class Collector:
    """Fake ``on_batch`` that records every batch it receives."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.batches: list[HintBatch] = []
        self._received = threading.Event()

    def __call__(self, batch: HintBatch) -> None:
        with self._lock:
            self.batches.append(batch)
        self._received.set()

    def wait(self, timeout_s: float) -> bool:
        return self._received.wait(timeout_s)

    def paths(self) -> set[str]:
        with self._lock:
            merged: set[str] = set()
            for batch in self.batches:
                if not batch.overflow:
                    merged |= batch.rel_paths
            return merged

    def has_overflow(self) -> bool:
        with self._lock:
            return any(batch.overflow for batch in self.batches)


def wait_until(predicate: Callable[[], bool], timeout_s: float) -> bool:
    """Poll ``predicate`` until it is true or ``timeout_s`` elapsed."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def wait_state(watcher: RootWatcher, state: str, timeout_s: float) -> bool:
    return wait_until(lambda: watcher.state == state, timeout_s)


def make_settings(**watch_overrides: Any) -> Settings:
    settings = Settings()
    for key, value in watch_overrides.items():
        setattr(settings.scan.watch, key, value)
    return settings


def start_watcher(
    root: Path, settings: Settings, collector: Collector
) -> tuple[RootWatcher, threading.Thread, threading.Event]:
    """Start a ``RootWatcher`` in a daemon thread and wait until it watches."""
    stop = threading.Event()
    watcher = RootWatcher(1, root, settings, on_batch=collector)
    thread = threading.Thread(target=watcher.run, args=(stop,), daemon=True)
    thread.start()
    assert wait_state(watcher, watcher.backend, _NATIVE_TIMEOUT_S), watcher.state
    time.sleep(0.5)
    return watcher, thread, stop


def stop_watcher(thread: threading.Thread, stop: threading.Event) -> None:
    stop.set()
    thread.join(5)


def test_native_backend_emits_batch_for_created_file(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    collector = Collector()

    watcher, thread, stop = start_watcher(root, make_settings(), collector)
    try:
        (root / "a.md").write_text("hello")
        assert collector.wait(_BATCH_TIMEOUT_S)
        assert wait_until(lambda: "a.md" in collector.paths(), _BATCH_TIMEOUT_S)
        assert watcher.state == "native"
    finally:
        stop_watcher(thread, stop)


def test_polling_backend_emits_batch_for_created_file(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    collector = Collector()

    watcher, thread, stop = start_watcher(
        root, make_settings(backend="polling", poll_delay_ms=200), collector
    )
    try:
        assert watcher.backend == "polling"
        (root / "a.md").write_text("hello")
        assert collector.wait(_BATCH_TIMEOUT_S)
        assert wait_until(lambda: "a.md" in collector.paths(), _BATCH_TIMEOUT_S)
        assert watcher.state == "polling"
    finally:
        stop_watcher(thread, stop)


def test_rename_emits_both_paths(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.md").write_text("hello")
    collector = Collector()

    _watcher, thread, stop = start_watcher(root, make_settings(), collector)
    try:
        (root / "a.md").rename(root / "b.md")
        assert wait_until(lambda: {"a.md", "b.md"} <= collector.paths(), _BATCH_TIMEOUT_S * 2)
    finally:
        stop_watcher(thread, stop)


def test_stop_event_stops_thread_quickly(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    collector = Collector()

    _watcher, thread, stop = start_watcher(root, make_settings(), collector)
    stop.set()
    thread.join(3.0)
    assert not thread.is_alive()


def test_missing_root_goes_down_and_recovers_with_catchup(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    collector = Collector()

    watcher, thread, stop = start_watcher(root, make_settings(), collector)
    try:
        shutil.rmtree(root)
        assert wait_state(watcher, "down", _NATIVE_TIMEOUT_S), watcher.state

        root.mkdir()
        assert wait_until(collector.has_overflow, _RECOVER_TIMEOUT_S)
    finally:
        stop_watcher(thread, stop)
