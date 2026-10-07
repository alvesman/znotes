"""SQLAlchemy engine and session helpers for the mynotes Postgres database.

The engine is created lazily from ``settings.postgres.dsn`` (task 02) and
cached process-wide; :func:`reset_engine` drops the cache for tests. DSNs use
the ``postgresql+psycopg://`` scheme (SQLAlchemy 2.0 + psycopg 3).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from mynotes.config import get_settings

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

_engine: Engine | None = None


def get_engine() -> Engine:
    """Return the cached engine, creating it from ``postgres.dsn`` on first use.

    Returns:
        The process-wide SQLAlchemy engine with ``pool_pre_ping`` enabled.

    Raises:
        RuntimeError: When no DSN is configured.
    """
    global _engine
    if _engine is None:
        dsn = get_settings().postgres.dsn
        value = dsn.get_secret_value() if dsn is not None else None
        if not value:
            raise RuntimeError("MYNOTES_PG_DSN is not set. See docs/infrastructure.md")
        _engine = create_engine(value, pool_pre_ping=True)
    return _engine


def reset_engine() -> None:
    """Dispose the cached engine so the next call re-reads settings (tests)."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional session bound to :func:`get_engine`.

    Commits when the block completes, rolls back on any error and always
    closes the session.

    Yields:
        A SQLAlchemy session in a transaction.
    """
    session = Session(bind=get_engine())
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()
