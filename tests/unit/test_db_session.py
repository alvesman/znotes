"""Unit tests for the db session helpers (no database needed)."""

from __future__ import annotations

import pytest

from mynotes.config import Settings
from mynotes.db.session import get_engine, reset_engine


def test_get_engine_requires_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing DSN raises a clear error pointing at the infrastructure docs."""
    settings = Settings(postgres={"dsn": None})
    monkeypatch.setattr("mynotes.db.session.get_settings", lambda: settings)
    reset_engine()
    try:
        with pytest.raises(RuntimeError, match="MYNOTES_PG_DSN is not set"):
            get_engine()
    finally:
        reset_engine()


def test_get_engine_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """The engine is created once, honours pool_pre_ping, and reset_engine drops it."""
    settings = Settings(postgres={"dsn": "postgresql+psycopg://user:pass@localhost:5432/db"})
    monkeypatch.setattr("mynotes.db.session.get_settings", lambda: settings)
    reset_engine()
    try:
        first = get_engine()
        assert get_engine() is first
        assert first.pool._pre_ping is True
        reset_engine()
        assert get_engine() is not first
    finally:
        reset_engine()
