"""Small shared state in the ``app_state`` table (watcher states, the drift report).

The worker writes, the web process reads; both also keep the per-machine
JSON files as a fallback, so single-machine mode works without the table
being reachable. Callers treat every failure here as "no state".
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert

from mynotes.db.models import AppState
from mynotes.db.session import session_scope


def get_state(key: str) -> Any | None:
    """The stored JSON value for ``key``, or None."""
    with session_scope() as session:
        row = session.get(AppState, key)
        return row.value if row is not None else None


def set_state(key: str, value: Any) -> None:
    """Store ``value`` (JSON-serializable) under ``key``, replacing what was there."""
    now = datetime.now(UTC)
    with session_scope() as session:
        session.execute(
            pg_insert(AppState)
            .values(key=key, value=value, updated_at=now)
            .on_conflict_do_update(index_elements=["key"], set_={"value": value, "updated_at": now})
        )
