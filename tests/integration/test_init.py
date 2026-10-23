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


def test_remote_roots_survive_init_and_are_never_scanned(init_settings: Settings) -> None:
    """A remote root is not in mynotes.yaml: init keeps it, the scheduler and watcher skip it."""
    from mynotes.core import roots
    from mynotes.core.acl import grant_prefixes
    from mynotes.core.service import trigger_scan
    from mynotes.scan.scheduler import enabled_root_ids
    from mynotes.scan.watch import watched_roots

    run_init()
    root_id = roots.add_remote_root("alice-notes", "alice")
    listed = {info.name: info for info in roots.list_roots()}
    assert listed["alice-notes"].kind == "remote" and listed["alice-notes"].owner == "alice"
    assert listed["alice-notes"].abs_path is None
    assert grant_prefixes(["alice"]) == ["alice-notes"]

    results = run_init()  # a second init must not disable the remote root
    assert all(result.ok for result in results), results
    enabled = {info.name: info.enabled for info in roots.list_roots()}
    assert enabled["ideas"] is True and enabled["alice-notes"] is True
    assert root_id not in enabled_root_ids(init_settings)
    assert all(name != "alice-notes" for name, _id, _path in watched_roots(init_settings))
    local_enabled = [info for info in roots.list_roots() if info.kind == "local" and info.enabled]
    assert len(trigger_scan()) == len(local_enabled)  # remote roots are never scanned here

    with pytest.raises(ValueError):
        roots.add_remote_root("alice-notes", "alice")  # taken
    with pytest.raises(ValueError):
        roots.add_remote_root("bad/name", "alice")
    assert roots.disable_remote_root("alice-notes") == 0
    assert {info.name: info.enabled for info in roots.list_roots()}["alice-notes"] is False
    _remove_remote_root("alice-notes", "alice")


def test_init_refuses_a_local_root_named_like_a_remote_root(
    init_settings: Settings, tmp_path: Path
) -> None:
    from mynotes.core import roots

    run_init()
    roots.add_remote_root("shared", "bob")
    try:
        folder = tmp_path / "shared"
        folder.mkdir()
        init_settings.roots.append(type(init_settings.roots[0])(name="shared", path=str(folder)))
        results = run_init()
        failed = [result for result in results if result.level == "fail"]
        assert failed and "remote root" in failed[0].detail
    finally:
        init_settings.roots.pop()
        _remove_remote_root("shared", "bob")


def _remove_remote_root(name: str, owner: str) -> None:
    """Drop the rows a remote-root test created, so the shared database stays clean."""
    from sqlalchemy import delete

    from mynotes.db.models import AclGrant, SourceRoot
    from mynotes.db.session import session_scope

    with session_scope() as session:
        session.execute(delete(AclGrant).where(AclGrant.principal == owner))
        session.execute(
            delete(SourceRoot).where(SourceRoot.name == name, SourceRoot.kind == "remote")
        )
