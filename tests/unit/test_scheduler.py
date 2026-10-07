"""Unit tests for the scan/reconcile scheduler (task 25).

A fake clock drives every schedule decision; ``scan_fn``/``enqueue_fn`` are
recorders and the source-root lookup is stubbed, so no database, Docker or
model endpoint is touched. Fixed wall-clock times are built from local time so
the cron tests pass in any timezone.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from mynotes.config import Settings, reset_settings_cache
from mynotes.jobs.registry import Priority
from mynotes.scan.scheduler import RECONCILE_DEDUPE_KEY, ScheduledTask, Scheduler

_DAY = datetime(2026, 9, 24)  # a Thursday

_ROOT_IDS = [7, 9]


def local_dt(hour: int, minute: int = 0) -> datetime:
    """Return the given local wall-clock time on ``_DAY`` as aware UTC."""
    return _DAY.replace(hour=hour, minute=minute, tzinfo=None).astimezone(UTC)


class FakeClock:
    """Callable ``now`` replacement with manual advancement."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now = self.now + timedelta(**delta)


@dataclass
class Recorder:
    """Records ``scan_fn``/``enqueue_fn`` calls; ``scan_fn`` can be made to raise."""

    scans: list[tuple[int, dict[str, str]]] = field(default_factory=list)
    enqueues: list[dict[str, Any]] = field(default_factory=list)
    scan_error: Exception | None = None

    def scan(self, root_id: int, **kwargs: str) -> None:
        if self.scan_error is not None:
            raise self.scan_error
        self.scans.append((root_id, kwargs))

    def enqueue(self, job_type: str, payload: dict[str, Any], **kwargs: Any) -> None:
        self.enqueues.append({"type": job_type, "payload": payload, **kwargs})


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run with no MYNOTES_* variables, no user config and no repo .env."""
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("MYNOTES_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(
        "mynotes.config.platformdirs.user_config_dir",
        lambda *args, **kwargs: str(tmp_path / "user-config"),
    )
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture()
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """Stub the enabled roots, recent documents and linked documents; recorder."""
    monkeypatch.setattr("mynotes.scan.scheduler.enabled_root_ids", lambda settings: list(_ROOT_IDS))
    monkeypatch.setattr("mynotes.scan.scheduler.recent_document_ids", lambda settings: [])
    monkeypatch.setattr("mynotes.scan.scheduler.linked_document_ids", lambda: [])
    return Recorder()


def make_scheduler(clock: FakeClock, recorder: Recorder, **overrides: Any) -> Scheduler:
    """Build a scheduler from default settings with schedule ``overrides``."""
    settings = Settings()
    for key, value in overrides.items():
        setattr(settings.scan.schedule, key, value)
    return Scheduler(settings, now=clock, scan_fn=recorder.scan, enqueue_fn=recorder.enqueue)


def task_by_name(scheduler: Scheduler, name: str) -> ScheduledTask:
    """Return the scheduled task with the given name."""
    return next(task for task in scheduler.tasks() if task.name == name)


def test_interval_mode_runs_startup_then_every_interval(recorder: Recorder) -> None:
    clock = FakeClock(local_dt(10))
    scheduler = make_scheduler(clock, recorder, interval_minutes=15)

    assert scheduler.tick() == ["startup"]
    assert recorder.scans == [
        (root_id, {"kind": "incremental", "trigger": "startup"}) for root_id in _ROOT_IDS
    ]

    clock.advance(minutes=10)
    assert scheduler.tick() == []
    assert len(recorder.scans) == 2

    clock.advance(minutes=5)
    assert scheduler.tick() == ["incremental-scan"]
    assert recorder.scans[2:] == [
        (root_id, {"kind": "incremental", "trigger": "schedule"}) for root_id in _ROOT_IDS
    ]
    assert task_by_name(scheduler, "incremental-scan").next_run == local_dt(10, 30)


def test_cron_next_runs_land_on_weekday_mornings(recorder: Recorder) -> None:
    clock = FakeClock(local_dt(21))
    scheduler = make_scheduler(
        clock, recorder, mode="cron", cron="*/30 8-20 * * 1-5", run_on_start=False
    )
    incremental = task_by_name(scheduler, "incremental-scan")

    when = clock.now
    for _ in range(30):
        when = incremental.compute_next(when)
        local = when.astimezone()
        assert local.weekday() < 5
        assert 8 <= local.hour <= 20
        assert local.minute in (0, 30)

    clock.now = incremental.next_run
    for name in ("full-scan", "reconcile", "version-sweep", "link-sweep", "maintenance"):
        task_by_name(scheduler, name).next_run = clock.now + timedelta(hours=1)
    assert scheduler.tick() == ["incremental-scan"]
    assert all(
        kwargs == {"kind": "incremental", "trigger": "schedule"} for _, kwargs in recorder.scans
    )


def test_full_scan_runs_once_per_day(recorder: Recorder) -> None:
    clock = FakeClock(local_dt(1))
    scheduler = make_scheduler(clock, recorder, mode="manual", run_on_start=False)
    full = task_by_name(scheduler, "full-scan")
    assert full.next_run.astimezone().strftime("%H:%M") == "03:00"

    clock.now = full.next_run
    assert scheduler.tick() == ["full-scan"]
    assert all(kwargs == {"kind": "full", "trigger": "schedule"} for _, kwargs in recorder.scans)
    assert len(recorder.scans) == 2

    assert scheduler.tick() == []

    clock.now = full.next_run
    task_by_name(scheduler, "reconcile").next_run = clock.now + timedelta(hours=1)
    task_by_name(scheduler, "link-sweep").next_run = clock.now + timedelta(hours=1)
    task_by_name(scheduler, "maintenance").next_run = clock.now + timedelta(hours=1)
    assert scheduler.tick() == ["full-scan", "version-sweep"]
    assert len(recorder.scans) == 4


def test_missed_runs_are_not_replayed_after_sleep(recorder: Recorder) -> None:
    clock = FakeClock(local_dt(10))
    scheduler = make_scheduler(clock, recorder, interval_minutes=15)
    assert scheduler.tick() == ["startup"]

    clock.advance(hours=5)
    assert scheduler.tick() == ["incremental-scan"]
    assert len(recorder.scans) == 4

    incremental = task_by_name(scheduler, "incremental-scan")
    assert incremental.next_run == clock.now + timedelta(minutes=15)

    assert scheduler.tick() == []
    assert len(recorder.scans) == 4


def test_manual_mode_skips_incremental_but_keeps_startup(recorder: Recorder) -> None:
    clock = FakeClock(local_dt(10))
    scheduler = make_scheduler(clock, recorder, mode="manual")

    assert "incremental-scan" not in [task.name for task in scheduler.tasks()]
    assert scheduler.tick() == ["startup"]
    assert all(kwargs["trigger"] == "startup" for _, kwargs in recorder.scans)


def test_action_exception_does_not_stop_other_tasks(
    recorder: Recorder, caplog: pytest.LogCaptureFixture
) -> None:
    clock = FakeClock(local_dt(10))
    scheduler = make_scheduler(clock, recorder, run_on_start=False)
    recorder.scan_error = RuntimeError("boom")
    for task in scheduler.tasks():
        task.next_run = clock.now

    with caplog.at_level(logging.ERROR):
        ran = scheduler.tick()

    assert ran == [
        "incremental-scan",
        "full-scan",
        "reconcile",
        "version-sweep",
        "link-sweep",
        "maintenance",
    ]
    assert recorder.scans == []
    assert "scheduled task incremental-scan failed" in caplog.text
    assert all(task.next_run > clock.now for task in scheduler.tasks())


def test_reconcile_enqueues_with_daily_dedupe_key(recorder: Recorder) -> None:
    clock = FakeClock(local_dt(10))
    scheduler = make_scheduler(
        clock, recorder, mode="manual", run_on_start=False, full_scan_cron=None
    )
    reconcile = scheduler.tasks()[0]
    assert reconcile.name == "reconcile"
    assert reconcile.next_run.astimezone().strftime("%H:%M") == "03:30"

    clock.now = reconcile.next_run
    assert scheduler.tick() == ["reconcile"]
    assert recorder.enqueues == [
        {
            "type": "reconcile",
            "payload": {"fix": True},
            "dedupe_key": RECONCILE_DEDUPE_KEY,
            "priority": Priority.MAINTENANCE,
        }
    ]
    assert scheduler.tick() == []


def test_maintenance_enqueues_the_three_nightly_jobs(recorder: Recorder) -> None:
    """The 04:30 task enqueues purge/clean/prune with daily dedupe keys."""
    clock = FakeClock(local_dt(10))
    scheduler = make_scheduler(
        clock, recorder, mode="manual", run_on_start=False, full_scan_cron=None
    )
    maintenance = task_by_name(scheduler, "maintenance")
    assert maintenance.next_run.astimezone().strftime("%H:%M") == "04:30"

    clock.now = maintenance.next_run
    assert scheduler.tick() == ["reconcile", "version-sweep", "link-sweep", "maintenance"]
    assert recorder.enqueues[1:] == [
        {
            "type": job_type,
            "payload": {},
            "dedupe_key": f"{job_type}:daily",
            "priority": Priority.MAINTENANCE,
        }
        for job_type in ("purge_tombstones", "clean_caches", "prune_ops")
    ]
    assert scheduler.tick() == []


def test_version_sweep_enqueues_detection_for_recent_documents(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 04:00 sweep enqueues one deduplicated version_detect per document."""
    monkeypatch.setattr("mynotes.scan.scheduler.recent_document_ids", lambda settings: [11, 22])
    clock = FakeClock(local_dt(10))
    scheduler = make_scheduler(
        clock, recorder, mode="manual", run_on_start=False, full_scan_cron=None
    )
    sweep = task_by_name(scheduler, "version-sweep")
    assert sweep.next_run.astimezone().strftime("%H:%M") == "04:00"

    clock.now = sweep.next_run
    assert scheduler.tick() == ["reconcile", "version-sweep"]
    assert recorder.enqueues[1:] == [
        {
            "type": "version_detect",
            "payload": {"document_id": 11},
            "dedupe_key": "version_detect:11",
            "priority": Priority.VERSION_DETECT,
        },
        {
            "type": "version_detect",
            "payload": {"document_id": 22},
            "dedupe_key": "version_detect:22",
            "priority": Priority.VERSION_DETECT,
        },
    ]
    assert scheduler.tick() == []


def test_link_sweep_enqueues_resolution_for_linked_documents(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 04:15 sweep enqueues one deduplicated link_resolve per linked document."""
    monkeypatch.setattr("mynotes.scan.scheduler.linked_document_ids", lambda: [33])
    clock = FakeClock(local_dt(10))
    scheduler = make_scheduler(
        clock, recorder, mode="manual", run_on_start=False, full_scan_cron=None
    )
    sweep = task_by_name(scheduler, "link-sweep")
    assert sweep.next_run.astimezone().strftime("%H:%M") == "04:15"

    clock.now = sweep.next_run
    assert scheduler.tick() == ["reconcile", "version-sweep", "link-sweep"]
    assert recorder.enqueues[1:] == [
        {
            "type": "link_resolve",
            "payload": {"document_id": 33},
            "dedupe_key": "link_resolve:33",
            "priority": Priority.VERSION_DETECT,
        }
    ]
    assert scheduler.tick() == []
