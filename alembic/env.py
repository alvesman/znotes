"""Alembic migration environment.

The target URL comes from a ``-x dsn=...`` override (used by the integration
fixtures) or from the mynotes settings (``MYNOTES_PG_DSN`` through the config
module of task 02). ``alembic.ini`` intentionally holds no URL.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from mynotes.config import get_settings
from mynotes.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _resolve_url() -> str:
    """Return the target DSN: ``-x dsn=...`` wins, else ``settings.postgres.dsn``.

    Returns:
        A ``postgresql+psycopg://`` URL.

    Raises:
        RuntimeError: When no DSN is configured.
    """
    override = context.get_x_argument(as_dictionary=True).get("dsn")
    if override:
        return override
    settings_dsn = get_settings().postgres.dsn
    value = settings_dsn.get_secret_value() if settings_dsn is not None else None
    if not value:
        raise RuntimeError("MYNOTES_PG_DSN is not set. See docs/infrastructure.md")
    return value


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting (``alembic upgrade --sql``)."""
    context.configure(
        url=_resolve_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run the migrations against a live connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        url=_resolve_url(),
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
