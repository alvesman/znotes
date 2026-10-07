"""Postgres-backed job queue on the ``job`` table (task 08 schema).

All functions take an open :class:`sqlalchemy.orm.Session` and let the caller
control the transaction, except :func:`claim`, which commits so the row locks
from ``FOR UPDATE SKIP LOCKED`` are released immediately and the job is
visibly ``running`` to other workers.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from sqlalchemy import func, literal, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from mynotes.db.models import Job

logger = logging.getLogger(__name__)

_MAX_ERROR_LENGTH = 2_000
_BACKOFF_BASE_S = 30
_BACKOFF_MAX_S = 3_600

_ACTIVE_STATUSES = ("queued", "running")


@dataclass(frozen=True)
class ClaimedJob:
    """A job claimed by a worker, as returned by :func:`claim`."""

    id: int
    type: str
    payload: dict[str, Any]
    attempts: int
    max_attempts: int


def backoff_seconds(attempts: int) -> int:
    """Return the retry backoff for ``attempts`` consumed so far.

    Args:
        attempts: The job's attempt count (1-based).

    Returns:
        ``min(30 * 2**(n-1), 3600)`` seconds.
    """
    return min(_BACKOFF_BASE_S * 2 ** (attempts - 1), _BACKOFF_MAX_S)


def enqueue(
    session: Session,
    type: str,
    payload: dict[str, Any],
    *,
    dedupe_key: str | None = None,
    priority: int = 100,
    run_after: datetime | None = None,
    max_attempts: int = 5,
) -> int | None:
    """Insert a new queued job.

    Args:
        session: Open session; the caller commits.
        type: Job type string (e.g. ``"extract"``).
        payload: JSON payload passed to the handler.
        dedupe_key: When set, only one job with this key may be ``queued`` or
            ``running`` at a time (partial unique index from migration 0002).
        priority: Lower is claimed first; use :class:`mynotes.jobs.registry.Priority`.
        run_after: Earliest claim time; defaults to now.
        max_attempts: Attempts before the job goes ``dead``.

    Returns:
        The new job id, or ``None`` when an active job with the same
        dedupe key exists (deduplicated).
    """
    stmt = pg_insert(Job).values(
        type=type,
        payload=payload,
        dedupe_key=dedupe_key,
        priority=priority,
        run_after=run_after if run_after is not None else func.now(),
        max_attempts=max_attempts,
    )
    if dedupe_key is not None:
        # The status values are inlined literals, not bound parameters: on a
        # generic prepared-statement plan Postgres cannot infer the partial
        # index from a parameterized conflict predicate (InvalidColumnReference
        # from the 11th execution of the statement on one connection).
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["dedupe_key"],
            index_where=Job.status.in_(
                [literal(status, literal_execute=True) for status in _ACTIVE_STATUSES]
            ),
        )
    row = session.execute(stmt.returning(Job.id)).first()
    if row is None:
        logger.debug("job type=%s dedupe_key=%s deduplicated", type, dedupe_key)
        return None
    return int(row.id)


def claim(
    session: Session,
    worker_id: str,
    types: Sequence[str] | None = None,
    limit: int = 1,
) -> list[ClaimedJob]:
    """Atomically claim up to ``limit`` queued jobs for a worker.

    A single ``UPDATE ... FROM (SELECT ... FOR UPDATE SKIP LOCKED LIMIT n)``
    statement marks jobs ``running``; rows are ordered by priority, then
    ``run_after``, then id. The candidate rows are joined one-to-one into the
    update, so more rows than ``limit`` can never be claimed (the equivalent
    ``WHERE id IN (SELECT ... LIMIT n)`` form can claim extra rows when
    Postgres re-evaluates the subquery under a concurrent update of a
    candidate row). Commits the session so the lock is released and the jobs
    are visibly running; pass a dedicated short-lived session.

    Args:
        session: Session used for the claiming transaction (committed here).
        worker_id: Identifier stored in ``job.locked_by``.
        types: Restrict to these job types; ``None`` claims any type.
        limit: Maximum number of jobs to claim.

    Returns:
        The claimed jobs in claim order (empty when nothing was available).
    """
    candidates = select(Job.id).where(Job.status == "queued", Job.run_after <= func.now())
    if types is not None:
        candidates = candidates.where(Job.type.in_(types))
    candidates = (
        candidates.order_by(Job.priority, Job.run_after, Job.id)
        .limit(limit)
        .with_for_update(skip_locked=True)
        .subquery()
    )
    stmt = (
        update(Job)
        .values(
            status="running", locked_by=worker_id, locked_at=func.now(), attempts=Job.attempts + 1
        )
        .where(Job.id == candidates.c.id)
        .returning(Job.id, Job.type, Job.payload, Job.attempts, Job.max_attempts)
    )
    rows = session.execute(stmt).all()
    session.commit()
    claimed = [
        ClaimedJob(
            id=r.id,
            type=r.type,
            payload=r.payload,
            attempts=r.attempts,
            max_attempts=r.max_attempts,
        )
        for r in rows
    ]
    if claimed:
        logger.debug("worker %s claimed jobs %s", worker_id, [job.id for job in claimed])
    return claimed


def complete(session: Session, job_id: int) -> None:
    """Mark a job ``done``. The caller commits."""
    session.execute(
        update(Job).where(Job.id == job_id).values(status="done", finished_at=func.now())
    )


def fail(session: Session, job_id: int, error: str) -> Literal["queued", "dead"]:
    """Record a job failure.

    With attempts left, the job is requeued with the exponential backoff from
    :func:`backoff_seconds`; otherwise it is marked ``dead``. Clears the lock
    columns and stores ``error`` (truncated to 2,000 characters) in
    ``last_error``. The caller commits.

    Args:
        session: Open session.
        job_id: The claimed job's id.
        error: Error description to store.

    Returns:
        ``"queued"`` when the job will be retried, ``"dead"`` otherwise.
    """
    error = error[:_MAX_ERROR_LENGTH]
    row = session.execute(select(Job.attempts, Job.max_attempts).where(Job.id == job_id)).one()
    attempts, max_attempts = row.attempts, row.max_attempts
    unlock = {"locked_by": None, "locked_at": None, "last_error": error}
    if attempts < max_attempts:
        session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(
                status="queued",
                run_after=datetime.now(UTC) + timedelta(seconds=backoff_seconds(attempts)),
                **unlock,
            )
        )
        logger.info("job %d failed (attempt %d/%d), retrying later", job_id, attempts, max_attempts)
        return "queued"
    session.execute(
        update(Job).where(Job.id == job_id).values(status="dead", finished_at=func.now(), **unlock)
    )
    logger.error("job %d failed permanently after %d attempts: %s", job_id, attempts, error)
    return "dead"


def retry_later(session: Session, job_id: int, delay_s: float, reason: str) -> None:
    """Requeue a claimed job after ``delay_s`` without consuming an attempt.

    Reverses the attempt increment done by :func:`claim`, so the next claim
    sees the same attempt count as before. The caller commits.
    """
    session.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(
            status="queued",
            run_after=datetime.now(UTC) + timedelta(seconds=delay_s),
            attempts=Job.attempts - 1,
            locked_by=None,
            locked_at=None,
            last_error=reason[:_MAX_ERROR_LENGTH],
        )
    )


def reap_stale(session: Session, older_than_s: int = 900) -> int:
    """Requeue ``running`` jobs whose lock is older than ``older_than_s``.

    Crash recovery for workers that died while holding jobs. The caller
    commits.

    Args:
        session: Open session.
        older_than_s: Lock age threshold in seconds.

    Returns:
        The number of requeued jobs.
    """
    result = session.execute(
        update(Job)
        .where(
            Job.status == "running",
            Job.locked_at.is_not(None),
            Job.locked_at < func.now() - timedelta(seconds=older_than_s),
        )
        .values(status="queued", locked_by=None, locked_at=None)
    )
    if result.rowcount:
        logger.info("reaped %d stale running job(s)", result.rowcount)
    return result.rowcount


def counts(session: Session) -> dict[str, int]:
    """Return the number of jobs per status (statuses with no rows are absent)."""
    rows = session.execute(select(Job.status, func.count()).group_by(Job.status)).all()
    return {status: count for status, count in rows}
