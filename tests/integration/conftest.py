"""Fixtures for integration tests: disposable Postgres and OpenSearch containers.

Session-scoped fixtures start one ``postgres:18.6`` container (the tag pinned
in ``docker-compose.yml``) and apply migration ``0001_corpus`` through the
``-x dsn=...`` Alembic override, and one ``opensearchproject/opensearch:3.8.0``
container without security. The function-scoped ``db_session`` wraps each test
in a transaction that is rolled back afterwards, so tests never persist data;
``os_settings`` hands out unique OpenSearch aliases per test for the same reason.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from testcontainers.community.opensearch import OpenSearchContainer
from testcontainers.community.postgres import PostgresContainer

from mynotes.config import Settings
from mynotes.db.session import reset_engine

_REPO_ROOT = Path(__file__).resolve().parents[2]
_POSTGRES_IMAGE = "postgres:18.6"
_OPENSEARCH_IMAGE = "opensearchproject/opensearch:3.8.0"


def _alembic_config(dsn: str) -> Config:
    """Build an Alembic config that migrates ``dsn`` via the ``-x dsn`` override."""
    cfg = Config(str(_REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_REPO_ROOT / "alembic"))
    cfg.cmd_opts = SimpleNamespace(x=[f"dsn={dsn}"])
    return cfg


@pytest.fixture(scope="session")
def pg_container() -> Iterator[PostgresContainer]:
    """Start a Postgres 18.6 container for the whole test session."""
    with PostgresContainer(_POSTGRES_IMAGE, driver="psycopg") as container:
        yield container


@pytest.fixture(scope="session")
def migrated_dsn(pg_container: PostgresContainer) -> Iterator[str]:
    """Run ``alembic upgrade head`` against the container and yield its DSN."""
    dsn = pg_container.get_connection_url()
    command.upgrade(_alembic_config(dsn), "head")
    yield dsn


@pytest.fixture(scope="session")
def alembic_cfg(migrated_dsn: str) -> Config:
    """An Alembic config targeting the migrated container database."""
    return _alembic_config(migrated_dsn)


@pytest.fixture
def db_session(migrated_dsn: str) -> Iterator[Session]:
    """A session bound to a transaction that is rolled back after the test."""
    engine = create_engine(migrated_dsn)
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        transaction.rollback()
        connection.close()
        engine.dispose()


@pytest.fixture
def settings_with_db(monkeypatch: pytest.MonkeyPatch, migrated_dsn: str) -> Iterator[Settings]:
    """Point ``get_settings()`` and the engine cache at the container database."""
    settings = Settings(postgres={"dsn": migrated_dsn})
    monkeypatch.setattr("mynotes.config.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.db.session.get_settings", lambda: settings)
    reset_engine()
    yield settings
    reset_engine()


@pytest.fixture(scope="session")
def os_container() -> Iterator[OpenSearchContainer]:
    """Start an OpenSearch 3.8.0 container (compose tag) without security."""
    with OpenSearchContainer(_OPENSEARCH_IMAGE, security_enabled=False) as container:
        yield container


@pytest.fixture
def os_settings(
    monkeypatch: pytest.MonkeyPatch, os_container: OpenSearchContainer
) -> Iterator[Settings]:
    """Settings pointing at the container, with dim 8 and unique aliases per test."""
    config = os_container.get_config()
    suffix = uuid4().hex[:8]
    settings = Settings(
        opensearch={
            "url": f"http://{config['host']}:{config['port']}",
            "chunks_alias": f"t{suffix}_chunks",
            "entities_alias": f"t{suffix}_entities",
        },
        models={"embedding": {"dim": 8}},
    )
    monkeypatch.setattr("mynotes.index.opensearch.get_settings", lambda: settings)
    yield settings
