"""Integration tests for ``run_init`` and ``collect_status`` (task 11)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
import sqlalchemy as sa
from testcontainers.community.opensearch import OpenSearchContainer

from mynotes.config import RootConfig, Settings
from mynotes.core.errors import NotFound
from mynotes.core.service import trigger_scan
from mynotes.core.setup import collect_status, run_init
from mynotes.db.models import SourceRoot
from mynotes.db.session import reset_engine, session_scope
from mynotes.index.opensearch import current_physical_index, get_client

pytestmark = pytest.mark.integration


@pytest.fixture
def init_settings(
    monkeypatch: pytest.MonkeyPatch,
    migrated_dsn: str,
    os_container: OpenSearchContainer,
    tmp_path: Path,
) -> Iterator[Settings]:
    """Settings pointing at the containers, with one root inside ``tmp_path``."""
    root = tmp_path / "notes"
    root.mkdir()
    config = os_container.get_config()
    suffix = uuid4().hex[:8]
    settings = Settings(
        postgres={"dsn": migrated_dsn},
        opensearch={
            "url": f"http://{config['host']}:{config['port']}",
            "chunks_alias": f"init{suffix}_chunks",
            "entities_alias": f"init{suffix}_entities",
        },
        models={"embedding": {"dim": 8}},
        roots=[{"name": "ideas", "path": str(root)}],
    )
    monkeypatch.setattr("mynotes.config.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.core.setup.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.db.session.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.index.opensearch.get_settings", lambda: settings)
    reset_engine()
    yield settings
    reset_engine()


def test_run_init_reports_no_fail(init_settings: Settings) -> None:
    results = run_init()
    assert [result for result in results if result.level == "fail"] == []


def test_run_init_registers_source_root(init_settings: Settings) -> None:
    run_init()
    expected = str(Path(init_settings.roots[0].path).resolve())
    with session_scope() as session:
        row = session.execute(
            sa.select(SourceRoot.abs_path, SourceRoot.enabled).where(SourceRoot.name == "ideas")
        ).one()
    assert row.abs_path == expected
    assert row.enabled is True


def _enabled(name: str) -> bool:
    with session_scope() as session:
        return bool(
            session.execute(
                sa.select(SourceRoot.enabled).where(SourceRoot.name == name)
            ).scalar_one()
        )


def test_run_init_disables_removed_root_and_reenables_it(
    init_settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root dropped from the configuration is disabled, and restored when re-added."""
    other = tmp_path / "other"
    other.mkdir()
    init_settings.roots.append(RootConfig(name="other", path=str(other)))
    run_init()
    assert _enabled("other") is True

    init_settings.roots.pop()
    results = run_init()
    assert _enabled("other") is False
    assert _enabled("ideas") is True
    disabled = [r for r in results if r.name == "root 'other'"]
    assert len(disabled) == 1 and disabled[0].level == "warn"
    assert "disabled" in disabled[0].detail
    monkeypatch.setattr("mynotes.core.service.get_settings", lambda: init_settings)
    with pytest.raises(NotFound):
        trigger_scan("other")

    init_settings.roots.append(RootConfig(name="other", path=str(other)))
    results = run_init()
    assert _enabled("other") is True
    assert any(r.name == "root 'other'" and "re-enabled" in r.detail for r in results)


def test_run_init_creates_chunks_alias(init_settings: Settings) -> None:
    run_init()
    client = get_client()
    assert current_physical_index(client, init_settings.opensearch.chunks_alias) is not None


def test_collect_status_reports_empty_stack(init_settings: Settings) -> None:
    run_init()
    status = collect_status()
    assert status["postgres"]["documents_by_status"] == {}
    assert status["postgres"]["files_by_status"] == {}
    assert status["postgres"]["versions"] == 0
    assert status["postgres"]["chunks"] == 0
    assert status["postgres"]["jobs"] == {"queued": 0, "running": 0, "failed": 0}
    assert status["postgres"]["outbox_unprocessed"] == 0
    alias_info = status["opensearch"]["aliases"][init_settings.opensearch.chunks_alias]
    assert alias_info["physical"] is not None
    assert alias_info["docs"] == 0
    assert status["opensearch"]["problems"] == []
    assert status["roots"][0]["source_root_id"] is not None
