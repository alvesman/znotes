"""Per-root locks shared by scans and the ingest API (PLAN.md §7.8).

:func:`mynotes.scan.scanner.scan_root` takes a *session-level try-lock* and
skips the scan when another one holds it. Ingest writes (an agent's manifest
or upload) instead take a *transaction-level blocking lock* on the same key,
so concurrent uploads of one root queue up behind each other and behind a
running scan rather than failing. Both use ``hashtext('scan:' || root_id)``.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session

ROOT_XACT_LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtext('scan:' || CAST(:rid AS bigint)))"


def root_xact_lock(session: Session, root_id: int) -> None:
    """Block until this transaction holds the root's lock; released at commit or rollback."""
    session.execute(text(ROOT_XACT_LOCK_SQL), {"rid": root_id})
