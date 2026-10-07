"""Scheduled scans and maintenance inside the worker (PLAN.md §7.8).

The :class:`Scheduler` owns every time-based trigger: incremental scans on an
interval or a cron expression, a nightly full verification scan, the nightly
``reconcile`` job, the nightly version-detection and link-resolution sweeps
and the catch-up scan on startup. No OS scheduler is
involved. Missed runs (laptop asleep) are not replayed: after a run, the next
run is computed from *now*, so one tick after a long sleep performs a single
scan. Cron expressions are evaluated in local time and stored in UTC.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Any

from croniter import croniter
from sqlalchemy import or_, select

from mynotes.config import Settings
from mynotes.jobs.queue import enqueue
from mynotes.jobs.registry import Priority
from mynotes.scan.scanner import scan_root

logger = logging.getLogger(__name__)

RECONCILE_CRON = "30 3 * * *"
"""The reconcile job runs daily at 03:30 local time."""

RECONCILE_DEDUPE_KEY = "reconcile:daily"

VERSION_SWEEP_CRON = "0 4 * * *"
"""The version-detection sweep runs daily at 04:00 local time."""

VERSION_SWEEP_WINDOW_DAYS = 7

LINK_SWEEP_CRON = "15 4 * * *"
"""The link-resolution sweep runs daily at 04:15 local time (task 39)."""

MAINTENANCE_CRON = "30 4 * * *"
"""The maintenance jobs (tombstone purge, cache clean, op pruning) run at 04:30."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _enqueue_job(
    job_type: str,
    payload: dict[str, Any],
    *,
    dedupe_key: str | None = None,
    priority: int = 100,
) -> int | None:
    """Insert a job through a short-lived session; the default ``enqueue_fn``."""
    from mynotes.db.session import session_scope

    with session_scope() as session:
        return enqueue(session, job_type, payload, dedupe_key=dedupe_key, priority=priority)


def enabled_root_ids(settings: Settings) -> list[int]:
    """Return the ids of the workspace's enabled source roots.

    Args:
        settings: Application settings (provides the workspace name).

    Returns:
        Source root ids in ascending order. Unit tests stub this function.
    """
    from mynotes.db.models import SourceRoot, Workspace
    from mynotes.db.session import session_scope

    stmt = (
        select(SourceRoot.id)
        .join(Workspace, Workspace.id == SourceRoot.workspace_id)
        .where(Workspace.name == settings.workspace, SourceRoot.enabled.is_(True))
        .order_by(SourceRoot.id)
    )
    with session_scope() as session:
        return [int(root_id) for root_id in session.execute(stmt).scalars()]


def recent_document_ids(settings: Settings, days: int = VERSION_SWEEP_WINDOW_DAYS) -> list[int]:
    """Return the ids of documents created or modified in the last ``days`` days.

    Args:
        settings: Application settings (unused; the sweep covers every
            non-deleted document). Unit tests stub this function.
        days: How far back to look.

    Returns:
        Document ids in ascending order.
    """
    from mynotes.db.models import Document
    from mynotes.db.session import session_scope

    cutoff = datetime.now(UTC) - timedelta(days=days)
    stmt = (
        select(Document.id)
        .where(
            Document.status != "deleted",
            or_(Document.updated_at >= cutoff, Document.created_at >= cutoff),
        )
        .order_by(Document.id)
    )
    with session_scope() as session:
        return [int(document_id) for document_id in session.execute(stmt).scalars()]


def linked_document_ids() -> list[int]:
    """Return the ids of documents whose latest version has resolvable links.

    Returns:
        Non-deleted documents with at least one wikilink or Markdown link in
        their latest version (see
        :func:`mynotes.graph.links.documents_with_resolvable_links`), in
        ascending order. Unit tests stub this function.
    """
    from mynotes.db.session import session_scope
    from mynotes.graph.links import documents_with_resolvable_links

    with session_scope() as session:
        return documents_with_resolvable_links(session)


def _cron_next(expression: str, now: datetime) -> datetime:
    """Return the next fire time of ``expression`` after ``now``.

    Args:
        expression: A cron expression, evaluated in local time.
        now: An aware UTC datetime.

    Returns:
        The next occurrence converted to UTC.
    """
    iterator = croniter(expression, now.astimezone())
    return iterator.get_next(datetime).astimezone(UTC)


@dataclass
class ScheduledTask:
    """One schedulable activity owned by the :class:`Scheduler`.

    Attributes:
        name: Stable identifier, e.g. ``"incremental-scan"``.
        next_run: Earliest instant the task is due (aware UTC).
        compute_next: Given ``now``, returns the next due instant; called
            after every run so missed schedules are never replayed.
        action: The work itself; exceptions are logged, not fatal.
    """

    name: str
    next_run: datetime
    compute_next: Callable[[datetime], datetime]
    action: Callable[[], None]


class Scheduler:
    """Runs scans and maintenance jobs on interval or cron schedules."""

    def __init__(
        self,
        settings: Settings,
        *,
        now: Callable[[], datetime] = _utc_now,
        scan_fn: Callable[..., Any] = scan_root,
        enqueue_fn: Callable[..., Any] | None = None,
    ) -> None:
        """Build the tasks configured in ``settings.scan.schedule``.

        Args:
            settings: Application settings; only ``scan.schedule`` is read.
            now: Clock; tests inject a fake.
            scan_fn: Scan entry point; tests inject a recorder.
            enqueue_fn: Job enqueue entry point; defaults to the Postgres
                queue through a short-lived session.
        """
        self.settings = settings
        self._now = now
        self._scan_fn = scan_fn
        self._enqueue_fn = enqueue_fn if enqueue_fn is not None else _enqueue_job
        self._tasks = self._build_tasks()
        self._startup_pending = settings.scan.schedule.run_on_start

    def _build_tasks(self) -> list[ScheduledTask]:
        """Create the scheduled tasks from the scan schedule settings."""
        schedule = self.settings.scan.schedule
        now = self._now()
        tasks: list[ScheduledTask] = []
        if schedule.mode == "interval":
            interval = timedelta(minutes=schedule.interval_minutes)
            tasks.append(
                ScheduledTask(
                    name="incremental-scan",
                    next_run=now + interval,
                    compute_next=lambda current: current + interval,
                    action=partial(self._scan_roots, "incremental", "schedule"),
                )
            )
        elif schedule.mode == "cron":
            if schedule.cron:
                tasks.append(
                    ScheduledTask(
                        name="incremental-scan",
                        next_run=_cron_next(schedule.cron, now),
                        compute_next=partial(_cron_next, schedule.cron),
                        action=partial(self._scan_roots, "incremental", "schedule"),
                    )
                )
            else:
                logger.warning(
                    "scan.schedule.mode is 'cron' but scan.schedule.cron is not set;"
                    " incremental scan is not scheduled"
                )
        if schedule.full_scan_cron:
            tasks.append(
                ScheduledTask(
                    name="full-scan",
                    next_run=_cron_next(schedule.full_scan_cron, now),
                    compute_next=partial(_cron_next, schedule.full_scan_cron),
                    action=partial(self._scan_roots, "full", "schedule"),
                )
            )
        tasks.append(
            ScheduledTask(
                name="reconcile",
                next_run=_cron_next(RECONCILE_CRON, now),
                compute_next=partial(_cron_next, RECONCILE_CRON),
                action=self._enqueue_reconcile,
            )
        )
        tasks.append(
            ScheduledTask(
                name="version-sweep",
                next_run=_cron_next(VERSION_SWEEP_CRON, now),
                compute_next=partial(_cron_next, VERSION_SWEEP_CRON),
                action=self._enqueue_version_sweep,
            )
        )
        tasks.append(
            ScheduledTask(
                name="link-sweep",
                next_run=_cron_next(LINK_SWEEP_CRON, now),
                compute_next=partial(_cron_next, LINK_SWEEP_CRON),
                action=self._enqueue_link_sweep,
            )
        )
        tasks.append(
            ScheduledTask(
                name="maintenance",
                next_run=_cron_next(MAINTENANCE_CRON, now),
                compute_next=partial(_cron_next, MAINTENANCE_CRON),
                action=self._enqueue_maintenance,
            )
        )
        return tasks

    def tasks(self) -> list[ScheduledTask]:
        """Return the scheduled tasks with their current ``next_run``."""
        return list(self._tasks)

    def tick(self) -> list[str]:
        """Run every task whose ``next_run`` is due, and the startup scan once.

        Missed runs are not replayed: each due task runs once and its
        ``next_run`` is recomputed from the current time. An exception in one
        action is logged and never stops the scheduler or the other tasks.

        Returns:
            The names of the tasks that ran, in execution order.
        """
        ran: list[str] = []
        if self._startup_pending:
            self._startup_pending = False
            try:
                self._scan_roots("incremental", "startup")
            except Exception:
                logger.exception("startup scan failed")
            ran.append("startup")
        now = self._now()
        for task in self._tasks:
            if task.next_run > now:
                continue
            try:
                task.action()
            except Exception:
                logger.exception("scheduled task %s failed", task.name)
            task.next_run = task.compute_next(now)
            ran.append(task.name)
            logger.info("scheduled task %s ran; next run %s", task.name, task.next_run)
        return ran

    def run(self, stop: threading.Event, tick_s: float = 5.0) -> None:
        """Tick every ``tick_s`` seconds until ``stop`` is set.

        Args:
            stop: Event set by the worker on shutdown; the loop returns soon
                after it is set.
            tick_s: Seconds between ticks.
        """
        logger.info("scheduler started (tick every %.1fs)", tick_s)
        while not stop.is_set():
            try:
                self.tick()
            except Exception:
                logger.exception("scheduler tick failed")
            stop.wait(tick_s)
        logger.info("scheduler stopped")

    def _scan_roots(self, kind: str, trigger: str) -> None:
        """Scan every enabled source root; ``scan_root`` holds the advisory lock."""
        for root_id in enabled_root_ids(self.settings):
            self._scan_fn(root_id, kind=kind, trigger=trigger)

    def _enqueue_reconcile(self) -> None:
        """Enqueue the nightly reconcile job, deduplicated per day."""
        self._enqueue_fn(
            "reconcile",
            {"fix": True},
            dedupe_key=RECONCILE_DEDUPE_KEY,
            priority=Priority.MAINTENANCE,
        )

    def _enqueue_version_sweep(self) -> None:
        """Enqueue ``version_detect`` for the documents of the last week."""
        for document_id in recent_document_ids(self.settings):
            self._enqueue_fn(
                "version_detect",
                {"document_id": document_id},
                dedupe_key=f"version_detect:{document_id}",
                priority=Priority.VERSION_DETECT,
            )

    def _enqueue_link_sweep(self) -> None:
        """Enqueue ``link_resolve`` for every document that has resolvable links."""
        for document_id in linked_document_ids():
            self._enqueue_fn(
                "link_resolve",
                {"document_id": document_id},
                dedupe_key=f"link_resolve:{document_id}",
                priority=Priority.VERSION_DETECT,
            )

    def _enqueue_maintenance(self) -> None:
        """Enqueue the nightly purge/clean/prune jobs, deduplicated per day."""
        from mynotes.jobs.maintenance import MAINTENANCE_JOB_TYPES, daily_dedupe_key

        for job_type in MAINTENANCE_JOB_TYPES:
            self._enqueue_fn(
                job_type,
                {},
                dedupe_key=daily_dedupe_key(job_type),
                priority=Priority.MAINTENANCE,
            )
