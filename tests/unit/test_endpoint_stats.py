"""Unit tests for the cross-process endpoint statistics (improvement plan step 8)."""

from __future__ import annotations

import pytest

from mynotes.jobs import endpoint_stats
from mynotes.llm.limits import EndpointSnapshot


def _snap(name: str, calls: int = 3) -> EndpointSnapshot:
    return EndpointSnapshot(
        name=name,
        state="closed",
        calls=calls,
        p50_ms=10.0,
        p95_ms=20.0,
        error_rate=0.0,
        last_error=None,
    )


def test_process_role_defaults_to_cli_and_can_be_set() -> None:
    endpoint_stats.set_process_role("cli")
    assert endpoint_stats.process_role() == "cli"
    endpoint_stats.set_process_role("worker")
    assert endpoint_stats.process_role() == "worker"
    endpoint_stats.set_process_role("cli")


def test_endpoint_views_without_session_labels_local_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(endpoint_stats, "all_snapshots", lambda: [_snap("llm"), _snap("embedding")])
    endpoint_stats.set_process_role("web")
    try:
        views = endpoint_stats.endpoint_views(None)
    finally:
        endpoint_stats.set_process_role("cli")
    assert [(v["name"], v["process"], v["age_s"]) for v in views] == [
        ("llm", "web", 0.0),
        ("embedding", "web", 0.0),
    ]
    assert views[0]["calls"] == 3 and views[0]["p50_ms"] == 10.0


def test_endpoint_views_appends_published_rows_of_other_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(endpoint_stats, "all_snapshots", lambda: [_snap("llm")])
    published = [
        {"name": "llm", "process": "worker", "age_s": 12.0, "calls": 40},
        {"name": "embedding", "process": "worker", "age_s": 12.0, "calls": 400},
    ]
    monkeypatch.setattr(
        endpoint_stats,
        "published_snapshots",
        lambda session, exclude_role=None: [r for r in published if r["process"] != exclude_role],
    )
    endpoint_stats.set_process_role("web")
    try:
        views = endpoint_stats.endpoint_views(session=object())  # type: ignore[arg-type]
    finally:
        endpoint_stats.set_process_role("cli")
    assert [(v["process"], v["name"]) for v in views] == [
        ("web", "llm"),
        ("worker", "llm"),
        ("worker", "embedding"),
    ]
