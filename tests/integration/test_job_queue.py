"""Integration tests for the Postgres job queue and the threaded worker.

Queue-level tests use the rolled-back ``db_session`` transaction. Worker tests
need committed rows because the worker claims through its own connection, so
they run against the session-scoped container via ``session_scope`` and clean
up after themselves; job types unique per test keep claims unambiguous.
"""

from __future__ import annotations

import os
import socket
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from mynotes.config import Settings
from mynotes.db.models import Job
from mynotes.db.session import get_engine, session_scope
from mynotes.jobs import worker as worker_module
from mynotes.jobs.queue import (
    backoff_seconds,
    claim,
    complete,
    counts,
    enqueue,
    fail,
    reap_stale,
    retry_later,
)
from mynotes.jobs.registry import JobContext, Priority, RetryLater, handler
from mynotes.jobs.worker import Worker, register_background_thread

pytestmark = pytest.mark.integration

_CALLS: list[tuple[dict[str, Any], JobContext]] = []


@handler("test_ok")
def _handle_ok(payload: dict[str, Any], ctx: JobContext) -> None:
    _CALLS.append((payload, ctx))


@handler("test_fail")
def _handle_fail(payload: dict[str, Any], ctx: JobContext) -> None:
    raise RuntimeError("boom")


@handler("test_later")
def _handle_later(payload: dict[str, Any], ctx: JobContext) -> None:
    raise RetryLater(5, "x")


@pytest.fixture(autouse=True)
def _reset_module_state() -> Iterator[None]:
    """Keep the global call log and background-thread registry clean per test."""
    yield
    _CALLS.clear()
    worker_module._BACKGROUND_THREADS.clear()


@pytest.fixture
def worker_env(settings_with_db: Settings, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Container settings, also visible to the worker's JobContext factory."""
    monkeypatch.setattr("mynotes.jobs.worker.get_settings", lambda: settings_with_db)
    return settings_with_db


def _unique_type(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


def _delete_type(session: Session, job_type: str) -> None:
    session.execute(delete(Job).where(Job.type == job_type))


def test_enqueue_dedupes_and_releases_after_complete(db_session: Session) -> None:
    jtype = _unique_type("dedupe")
    first = enqueue(db_session, jtype, {"n": 1}, dedupe_key="d1")
    assert isinstance(first, int)
    assert enqueue(db_session, jtype, {"n": 2}, dedupe_key="d1") is None
    complete(db_session, first)
    second = enqueue(db_session, jtype, {"n": 3}, dedupe_key="d1")
    assert isinstance(second, int) and second != first


def test_claim_orders_by_priority_then_run_after(db_session: Session) -> None:
    jtype = _unique_type("order")
    enqueue(db_session, jtype, {"name": "b"}, priority=Priority.CHUNK)
    enqueue(db_session, jtype, {"name": "a"}, priority=Priority.EXTRACT)
    enqueue(
        db_session,
        jtype,
        {"name": "c"},
        priority=Priority.CHUNK,
        run_after=datetime.now(UTC) - timedelta(seconds=60),
    )
    claimed = [claim(db_session, "w", types=[jtype])[0] for _ in range(3)]
    assert [job.payload["name"] for job in claimed] == ["a", "c", "b"]
    assert claim(db_session, "w", types=[jtype]) == []


def test_claim_skips_future_run_after(db_session: Session) -> None:
    jtype = _unique_type("future")
    enqueue(db_session, jtype, {"n": 1}, run_after=datetime.now(UTC) + timedelta(hours=1))
    assert claim(db_session, "w", types=[jtype]) == []
    enqueue(db_session, jtype, {"n": 2})
    assert len(claim(db_session, "w", types=[jtype])) == 1


def test_concurrent_claims_are_exactly_once(settings_with_db: Settings) -> None:
    jtype = _unique_type("conc")
    with session_scope() as session:
        ids = [enqueue(session, jtype, {"i": i}) for i in range(20)]
    assert len(ids) == 20

    claimed_ids: list[int] = []
    lock = threading.Lock()

    def claim_loop() -> None:
        while True:
            session = Session(bind=get_engine())
            try:
                batch = claim(session, f"w-{threading.get_ident()}", types=[jtype], limit=1)
            finally:
                session.close()
            if not batch:
                return
            with lock:
                claimed_ids.extend(job.id for job in batch)

    threads = [threading.Thread(target=claim_loop) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not any(thread.is_alive() for thread in threads)
    assert len(claimed_ids) == 20
    assert set(claimed_ids) == set(ids)

    with session_scope() as session:
        _delete_type(session, jtype)


def test_fail_backoff_then_dead(db_session: Session) -> None:
    jtype = _unique_type("fail")
    jid = enqueue(db_session, jtype, {}, max_attempts=2)
    assert claim(db_session, "w", types=[jtype])[0].attempts == 1

    assert fail(db_session, jid, "first error") == "queued"
    job = db_session.get(Job, jid)
    assert job is not None
    assert job.status == "queued"
    assert job.attempts == 1
    assert job.last_error == "first error"
    assert job.run_after > datetime.now(UTC) + timedelta(seconds=backoff_seconds(1) - 10)
    assert job.locked_by is None and job.locked_at is None

    db_session.execute(update(Job).where(Job.id == jid).values(run_after=func.now()))
    assert claim(db_session, "w", types=[jtype])[0].attempts == 2
    assert fail(db_session, jid, "e" * 3000) == "dead"
    job = db_session.get(Job, jid)
    assert job is not None
    assert job.status == "dead"
    assert job.finished_at is not None
    assert job.last_error == "e" * 2000


def test_worker_retry_later_keeps_attempts(worker_env: Settings) -> None:
    with session_scope() as session:
        jid = enqueue(session, "test_later", {"n": 1})
    before = datetime.now(UTC)
    assert Worker(concurrency=1, poll_interval_s=0.05).run_once() is True
    with session_scope() as session:
        job = session.get(Job, jid)
        assert job is not None
        assert job.status == "queued"
        assert job.attempts == 0
        assert job.last_error == "x"
        assert before + timedelta(seconds=4) <= job.run_after
        assert job.run_after <= datetime.now(UTC) + timedelta(seconds=6)
        _delete_type(session, "test_later")


def test_reap_stale_resets_old_locks(db_session: Session) -> None:
    jtype = _unique_type("reap")
    jid = enqueue(db_session, jtype, {})
    claimed = claim(db_session, "w", types=[jtype])[0]
    db_session.execute(
        update(Job).where(Job.id == jid).values(locked_at=datetime.now(UTC) - timedelta(hours=1))
    )
    assert reap_stale(db_session) >= 1
    job = db_session.get(Job, jid)
    assert job is not None
    assert job.status == "queued"
    assert job.locked_by is None and job.locked_at is None
    assert claim(db_session, "w", types=[jtype])[0].id == claimed.id


def test_worker_run_once_executes_handler(worker_env: Settings) -> None:
    with session_scope() as session:
        jid = enqueue(session, "test_ok", {"n": 7})
    worker = Worker(concurrency=1, poll_interval_s=0.05)
    for _ in range(10):
        if not worker.run_once():
            break
    with session_scope() as session:
        job = session.get(Job, jid)
        assert job is not None
        assert job.status == "done"
        assert job.finished_at is not None
    assert len(_CALLS) == 1
    payload, ctx = _CALLS[0]
    assert payload == {"n": 7}
    assert ctx.settings is worker_env
    assert ctx.worker_id == f"{socket.gethostname()}:{os.getpid()}:0"


def test_worker_fail_requeues_with_backoff(worker_env: Settings) -> None:
    with session_scope() as session:
        jid = enqueue(session, "test_fail", {})
    assert Worker(concurrency=1, poll_interval_s=0.05).run_once() is True
    with session_scope() as session:
        job = session.get(Job, jid)
        assert job is not None
        assert job.status == "queued"
        assert job.attempts == 1
        assert job.last_error == "RuntimeError: boom"
        assert job.run_after > datetime.now(UTC)
        _delete_type(session, "test_fail")


def test_claim_never_returns_more_than_limit(worker_env: Settings) -> None:
    """``claim(limit=1)`` returns at most one job per call, stranding none.

    Regression guard for the task-51 suite flake: the previous
    ``WHERE id IN (SELECT ... LIMIT n FOR UPDATE SKIP LOCKED)`` form could
    claim two jobs in one call when Postgres re-evaluated the subquery under
    a concurrent update (EvalPlanQual); ``run_once`` executed only the first
    and the second stayed ``running`` until the reaper requeued it 15 minutes
    later. The ``UPDATE ... FROM (SELECT ... LIMIT n)`` form joins each job
    row to at most one candidate row, so the limit holds.
    """
    jtype = _unique_type("limit")
    try:
        with session_scope() as session:
            for n in range(6):
                enqueue(session, jtype, {"n": n})
        seen: list[int] = []
        while True:
            with session_scope() as session:
                batch = claim(session, f"{socket.gethostname()}:probe:0", types=[jtype], limit=1)
            if not batch:
                break
            assert len(batch) == 1
            seen.append(batch[0].id)
            with session_scope() as session:
                complete(session, batch[0].id)
        assert len(seen) == 6
        with session_scope() as session:
            statuses = dict(
                session.execute(
                    select(Job.status, func.count()).where(Job.type == jtype).group_by(Job.status)
                ).all()
            )
        assert statuses == {"done": 6}
    finally:
        with session_scope() as session:
            _delete_type(session, jtype)


def test_run_once_executes_every_claimed_job(
    worker_env: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ``claim()`` ever returned two jobs, ``run_once`` executes both.

    Guards the defensive loop in ``Worker._claim_and_run``: a queue anomaly
    must surface as two completed jobs, never as one job left ``running``.
    """
    jtype = _unique_type("batch")
    try:
        with session_scope() as session:
            first = enqueue(session, jtype, {"n": 1})
            second = enqueue(session, jtype, {"n": 2})
        assert first is not None and second is not None
        from mynotes.jobs.queue import ClaimedJob
        from mynotes.jobs.registry import _HANDLERS as _registry_handlers

        _registry_handlers[jtype] = _handle_ok

        def double_claim(
            session: Session, worker_id: str, types: Any = None, limit: int = 1
        ) -> list[ClaimedJob]:
            return [
                ClaimedJob(id=first, type=jtype, payload={"n": 1}, attempts=1, max_attempts=5),
                ClaimedJob(id=second, type=jtype, payload={"n": 2}, attempts=1, max_attempts=5),
            ]

        monkeypatch.setattr(worker_module, "claim", double_claim)
        assert Worker(concurrency=1, poll_interval_s=0.05, types=[jtype]).run_once() is True
        assert [payload["n"] for payload, _ in _CALLS] == [1, 2]
        with session_scope() as session:
            statuses = [session.get(Job, jid).status for jid in (first, second)]
        assert statuses == ["done", "done"]
    finally:
        from mynotes.jobs.registry import _HANDLERS as _registry_handlers

        _registry_handlers.pop(jtype, None)
        with session_scope() as session:
            _delete_type(session, jtype)


def test_background_thread_lifecycle(worker_env: Settings) -> None:
    started = threading.Event()
    stopped = threading.Event()

    def target(stop_event: threading.Event) -> None:
        started.set()
        stop_event.wait(10)
        stopped.set()

    register_background_thread(f"bg_{uuid4().hex[:6]}", target)
    worker = Worker(concurrency=1, poll_interval_s=0.05, types=[])
    worker.start()
    assert started.wait(5)
    worker.stop(timeout_s=5)
    assert stopped.wait(5)
    assert worker._threads == []


def test_retry_later_does_not_consume_attempt(db_session: Session) -> None:
    jtype = _unique_type("later")
    jid = enqueue(db_session, jtype, {})
    assert claim(db_session, "w", types=[jtype])[0].attempts == 1
    retry_later(db_session, jid, 120.0, "hold on")
    job = db_session.get(Job, jid)
    assert job is not None
    assert job.status == "queued"
    assert job.attempts == 0
    assert job.last_error == "hold on"
    assert job.run_after > datetime.now(UTC) + timedelta(seconds=100)
    assert job.locked_by is None


def test_counts_by_status(db_session: Session) -> None:
    jtype = _unique_type("counts")
    before = counts(db_session)
    done_id = enqueue(db_session, jtype, {})
    queued_id = enqueue(db_session, jtype, {})
    running_id = enqueue(db_session, jtype, {})
    complete(db_session, done_id)
    claim(db_session, "w", types=[jtype], limit=1)
    after = counts(db_session)
    assert after.get("done", 0) == before.get("done", 0) + 1
    assert after.get("running", 0) == before.get("running", 0) + 1
    assert after.get("queued", 0) == before.get("queued", 0) + 1
    statuses = {
        row.status
        for row in db_session.execute(
            select(Job.status).where(Job.id.in_([done_id, queued_id, running_id]))
        )
    }
    assert statuses == {"done", "queued", "running"}
