"""Threaded worker that claims jobs from the Postgres queue and runs handlers.

One :class:`Worker` runs ``concurrency`` job threads plus a reaper thread
(crash recovery every 60 s). Long-running helper loops from later tasks
(outbox syncer, scheduler, watcher) register themselves with
:func:`register_background_thread` and are started and stopped together with
the job threads; each target receives the worker's stop :class:`threading.Event`
and must return soon after it is set.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from collections.abc import Callable, Sequence

from sqlalchemy.orm import Session

from mynotes.config import get_settings
from mynotes.db.session import get_engine, session_scope
from mynotes.index.sync import sync_loop
from mynotes.jobs import (
    maintenance,  # noqa: F401  registers the purge/clean/prune handlers
    pipeline,  # noqa: F401  registers the extract/chunk handlers
)
from mynotes.jobs.endpoint_stats import publish_loop
from mynotes.jobs.queue import ClaimedJob, claim, complete, fail, reap_stale, retry_later
from mynotes.jobs.registry import JobContext, RetryLater, get_handler, registered_types
from mynotes.llm.errors import EndpointUnavailable
from mynotes.scan.scheduler import Scheduler

logger = logging.getLogger(__name__)

BackgroundTarget = Callable[[threading.Event], None]

_BACKGROUND_THREADS: list[tuple[str, BackgroundTarget]] = []

_REAP_INTERVAL_S = 60.0
_ENDPOINT_RETRY_DELAY_S = 60.0


def register_background_thread(name: str, target: BackgroundTarget) -> None:
    """Register a long-running loop that the next :meth:`Worker.start` launches.

    Args:
        name: Thread name, shown in logs.
        target: Callable receiving the worker's stop event; it must return
            soon after the event is set.
    """
    _BACKGROUND_THREADS.append((name, target))
    logger.debug("registered background thread %r", name)


register_background_thread("outbox-sync", sync_loop)
register_background_thread("scheduler", lambda stop: Scheduler(get_settings()).run(stop))
register_background_thread("endpoint-stats", publish_loop)


def unregister_background_thread(name: str) -> None:
    """Remove a previously registered background thread; a no-op when absent."""
    _BACKGROUND_THREADS[:] = [entry for entry in _BACKGROUND_THREADS if entry[0] != name]
    logger.debug("unregistered background thread %r", name)


def _default_worker_id(thread_index: int | str) -> str:
    """Return ``hostname:pid:thread_index``."""
    return f"{socket.gethostname()}:{os.getpid()}:{thread_index}"


class Worker:
    """Consumes jobs from the Postgres queue with a pool of daemon threads."""

    def __init__(
        self,
        concurrency: int = 4,
        poll_interval_s: float = 1.0,
        types: Sequence[str] | None = None,
        worker_id: str | None = None,
    ) -> None:
        """Store the pool configuration.

        Args:
            concurrency: Number of job threads.
            poll_interval_s: Wait between polls when nothing was claimed.
            types: Restrict claiming to these job types; ``None`` claims every
                type that has a registered handler (re-evaluated per claim).
            worker_id: Overrides the default ``hostname:pid:thread_index``.
        """
        self.concurrency = concurrency
        self.poll_interval_s = poll_interval_s
        self.types = list(types) if types is not None else None
        self._worker_id = worker_id
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    def _worker_id_for(self, thread_index: int | str) -> str:
        return self._worker_id or _default_worker_id(thread_index)

    def start(self) -> None:
        """Start the job threads, the reaper and all registered background threads."""
        if self._threads:
            logger.warning("worker already started")
            return
        self._stop_event.clear()
        for index in range(self.concurrency):
            self._threads.append(
                threading.Thread(
                    target=self._job_loop,
                    args=(index,),
                    name=f"mynotes-worker-{index}",
                    daemon=True,
                )
            )
        self._threads.append(
            threading.Thread(target=self._reaper_loop, name="mynotes-reaper", daemon=True)
        )
        for name, target in list(_BACKGROUND_THREADS):
            self._threads.append(
                threading.Thread(target=target, args=(self._stop_event,), name=name, daemon=True)
            )
        for thread in self._threads:
            thread.start()
        logger.info(
            "worker started: %d job thread(s), %d background thread(s), id prefix %s",
            self.concurrency,
            len(self._threads) - self.concurrency - 1,
            self._worker_id_for(0),
        )

    def stop(self, timeout_s: float = 30) -> None:
        """Set the stop event and join all threads."""
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=timeout_s)
        self._threads.clear()

    def run_once(self) -> bool:
        """Claim and synchronously run one batch of claimed jobs.

        Returns:
            ``True`` when a job ran (whatever the outcome), ``False`` when
            nothing was claimable. Intended for tests and one-shot runs.
        """
        return self._claim_and_run(self._worker_id_for(0))

    def _job_loop(self, index: int) -> None:
        worker_id = self._worker_id_for(index)
        while not self._stop_event.is_set():
            try:
                ran = self._claim_and_run(worker_id)
            except Exception:
                logger.exception("job thread %s hit an unexpected error", worker_id)
                ran = False
            if not ran:
                self._stop_event.wait(self.poll_interval_s)

    def _claim_and_run(self, worker_id: str) -> bool:
        claimed = self._claim(worker_id)
        if not claimed:
            return False
        # claim() never returns more than ``limit`` rows by construction, but
        # execute every claimed job so an anomaly can never strand one in
        # ``running`` until the reaper requeues it 15 minutes later.
        for job in claimed:
            self._execute(job, worker_id)
        return True

    def _claim(self, worker_id: str) -> list[ClaimedJob]:
        types = self.types if self.types is not None else registered_types()
        session = Session(bind=get_engine())
        try:
            return claim(session, worker_id, types=types, limit=1)
        finally:
            session.close()

    def _execute(self, job: ClaimedJob, worker_id: str) -> None:
        try:
            job_handler = get_handler(job.type)
        except KeyError:
            logger.error("claimed job %d has type %r without a handler", job.id, job.type)
            with session_scope() as session:
                fail(session, job.id, f"no handler registered for type {job.type!r}")
            return
        context = JobContext(worker_id=worker_id, settings=get_settings())
        try:
            job_handler(job.payload, context)
        except RetryLater as exc:
            logger.info(
                "job %d (%s) retry later in %.0fs: %s", job.id, job.type, exc.delay_s, exc.reason
            )
            with session_scope() as session:
                retry_later(session, job.id, exc.delay_s, exc.reason)
        except EndpointUnavailable:
            logger.warning("job %d (%s): endpoint unavailable, retrying in 60s", job.id, job.type)
            with session_scope() as session:
                retry_later(session, job.id, _ENDPOINT_RETRY_DELAY_S, "endpoint unavailable")
        except Exception as exc:
            logger.exception("job %d (%s) raised", job.id, job.type)
            with session_scope() as session:
                fail(session, job.id, f"{type(exc).__name__}: {exc}")
        else:
            with session_scope() as session:
                complete(session, job.id)

    def _reaper_loop(self) -> None:
        while not self._stop_event.wait(_REAP_INTERVAL_S):
            try:
                with session_scope() as session:
                    reaped = reap_stale(session)
                if reaped:
                    logger.info("reaper requeued %d stale job(s)", reaped)
            except Exception:
                logger.exception("job reaper failed")
