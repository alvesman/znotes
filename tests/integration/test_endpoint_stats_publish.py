"""The worker publishes its endpoint snapshots; the status merges them (step 8)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import delete

from mynotes.config import Settings
from mynotes.db.models import EndpointStat
from mynotes.db.session import reset_engine, session_scope
from mynotes.jobs import endpoint_stats
from mynotes.llm.limits import EndpointSnapshot

pytestmark = pytest.mark.integration


@pytest.fixture
def env(migrated_dsn: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    settings = Settings(postgres={"dsn": migrated_dsn})
    for target in ("mynotes.config.get_settings", "mynotes.db.session.get_settings"):
        monkeypatch.setattr(target, lambda: settings)
    reset_engine()
    yield settings
    with session_scope() as session:
        session.execute(delete(EndpointStat))
    endpoint_stats.set_process_role("cli")
    reset_engine()


def _snap(name: str, calls: int, error: str | None = None) -> EndpointSnapshot:
    return EndpointSnapshot(
        name=name,
        state="closed",
        calls=calls,
        p50_ms=50.0,
        p95_ms=90.0,
        error_rate=0.1 if error else 0.0,
        last_error=error,
    )


def test_publish_upserts_and_other_roles_read_with_age(
    env: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        endpoint_stats, "all_snapshots", lambda: [_snap("llm", 5), _snap("embedding", 50)]
    )
    endpoint_stats.set_process_role("worker")
    with session_scope() as session:
        assert endpoint_stats.publish_snapshots(session) == 2
    # a second publication with new counts replaces the rows (one per role and endpoint)
    monkeypatch.setattr(endpoint_stats, "all_snapshots", lambda: [_snap("llm", 7, "timeout")])
    with session_scope() as session:
        assert endpoint_stats.publish_snapshots(session) == 1
        rows = {
            (r.process_role, r.name): (r.calls, r.last_error)
            for r in session.query(EndpointStat).all()
        }
    assert rows == {("worker", "llm"): (7, "timeout"), ("worker", "embedding"): (50, None)}

    # the web process sees its own snapshots first, then the worker's rows with an age
    endpoint_stats.set_process_role("web")
    monkeypatch.setattr(endpoint_stats, "all_snapshots", lambda: [_snap("reranker", 1)])
    with session_scope() as session:
        views = endpoint_stats.endpoint_views(session)
    assert [(v["process"], v["name"], v["calls"]) for v in views] == [
        ("web", "reranker", 1),
        ("worker", "embedding", 50),
        ("worker", "llm", 7),
    ]
    assert views[1]["age_s"] >= 0.0 and views[1]["host"]
    # the worker itself does not see its own published rows twice
    endpoint_stats.set_process_role("worker")
    with session_scope() as session:
        assert [v["process"] for v in endpoint_stats.endpoint_views(session)] == ["worker"]
