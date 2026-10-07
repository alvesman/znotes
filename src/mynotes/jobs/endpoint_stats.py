"""Model-endpoint statistics across processes (improvement plan step 8).

Every process keeps in-memory guard snapshots (:func:`mynotes.llm.limits.all_snapshots`)
for the endpoints *it* called. The Status page runs in the web process, so it
could not show the worker's calls, which are most of them. The worker now
publishes its snapshots to ``endpoint_stat`` every :data:`PUBLISH_INTERVAL_S`
seconds and on stop; :func:`endpoint_views` merges the local snapshots with
the rows published by the other roles, each tagged with its process and age.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from mynotes.db.models import EndpointStat
from mynotes.llm.limits import all_snapshots

logger = logging.getLogger(__name__)

PUBLISH_INTERVAL_S = 30.0
"""Seconds between two publications by the worker."""

_role_lock = threading.Lock()
_process_role = "cli"


def set_process_role(role: str) -> None:
    """Name this process for the statistics (``worker``, ``web``; ``cli`` by default)."""
    global _process_role
    with _role_lock:
        _process_role = role


def process_role() -> str:
    """The name this process publishes and labels its snapshots with."""
    with _role_lock:
        return _process_role


def publish_snapshots(session: Session, role: str | None = None) -> int:
    """Upsert this process's guard snapshots as ``role`` rows.

    Args:
        session: An open session/transaction; rows are written, not committed.
        role: The process role; the current one when omitted.

    Returns:
        The number of rows written.
    """
    role = role or process_role()
    now = datetime.now(UTC)
    count = 0
    for snapshot in all_snapshots():
        values = {
            "process_role": role,
            "name": snapshot.name,
            "state": snapshot.state,
            "calls": int(snapshot.calls),
            "error_rate": float(snapshot.error_rate),
            "p50_ms": snapshot.p50_ms,
            "p95_ms": snapshot.p95_ms,
            "last_error": snapshot.last_error,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "updated_at": now,
        }
        session.execute(
            pg_insert(EndpointStat)
            .values(**values)
            .on_conflict_do_update(
                constraint="pk_endpoint_stat",
                set_={k: v for k, v in values.items() if k not in ("process_role", "name")},
            )
        )
        count += 1
    return count


def published_snapshots(
    session: Session, *, exclude_role: str | None = None
) -> list[dict[str, Any]]:
    """The rows other processes published, newest first, as status dicts."""
    statement = select(EndpointStat).order_by(EndpointStat.process_role, EndpointStat.name)
    if exclude_role:
        statement = statement.where(EndpointStat.process_role != exclude_role)
    now = datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    for stat in session.execute(statement).scalars():
        rows.append(
            {
                "name": stat.name,
                "state": stat.state,
                "calls": int(stat.calls),
                "p50_ms": stat.p50_ms,
                "p95_ms": stat.p95_ms,
                "error_rate": float(stat.error_rate),
                "last_error": stat.last_error,
                "process": stat.process_role,
                "host": stat.host,
                "pid": int(stat.pid),
                "age_s": max(0.0, (now - stat.updated_at).total_seconds()),
            }
        )
    return rows


def endpoint_views(session: Session | None) -> list[dict[str, Any]]:
    """Local snapshots (``process`` = this role, ``age_s`` 0) plus the published ones.

    Args:
        session: An open session for the published rows; None skips them.
    """
    role = process_role()
    views: list[dict[str, Any]] = [
        {**asdict(snapshot), "process": role, "age_s": 0.0} for snapshot in all_snapshots()
    ]
    if session is not None:
        views.extend(published_snapshots(session, exclude_role=role))
    return views


def publish_loop(stop: threading.Event) -> None:
    """Worker background thread: publish every interval and once more on stop."""
    from mynotes.db.session import session_scope

    while True:
        stopped = stop.wait(PUBLISH_INTERVAL_S)
        try:
            with session_scope() as session:
                publish_snapshots(session)
        except Exception as exc:  # the stats must never take the worker down
            logger.warning("endpoint stats not published: %s: %s", type(exc).__name__, exc)
        if stopped:
            return
