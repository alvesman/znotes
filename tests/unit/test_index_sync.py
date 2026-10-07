"""Unit tests for the outbox syncer's bulk request sizing (task 51)."""

from __future__ import annotations

from typing import Any

import pytest

from mynotes.index import sync as sync_module


@pytest.fixture
def captured_bulk(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the opensearch bulk helper with one that records its kwargs."""
    calls: list[dict[str, Any]] = []

    def fake_bulk(client: Any, actions: list[dict[str, Any]], **kwargs: Any) -> tuple[list, list]:
        calls.append({"actions": actions, **kwargs})
        return [], []

    monkeypatch.setattr(sync_module, "bulk_helper", fake_bulk)
    return calls


def test_send_limits_each_bulk_request_to_200_documents(
    captured_bulk: list[dict[str, Any]],
) -> None:
    """The bulk helper is asked to split requests at 200 docs (task 51).

    4096-dim vectors serialize to roughly 40 KB per chunk document; 200 docs
    keep every request near 8 MB, below the ~20 MB request budget. The
    splitting itself happens inside the opensearchpy helper (it streams the
    actions in ``chunk_size`` batches).
    """
    actions = [
        {"_op_type": "index", "_index": "perf_chunks", "_id": str(index), "_source": {}}
        for index in range(450)
    ]
    errors = sync_module._send(client=object(), actions=actions)
    assert errors == {}
    assert len(captured_bulk) == 1
    assert captured_bulk[0]["chunk_size"] == 200
    assert len(captured_bulk[0]["actions"]) == 450


def test_send_reports_failed_documents(
    monkeypatch: pytest.MonkeyPatch, captured_bulk: list[dict[str, Any]]
) -> None:
    """Failed items are returned as ``{doc_id: error}``; 404 deletes pass."""
    actions = [
        {"_op_type": "index", "_index": "perf_chunks", "_id": "a", "_source": {}},
        {"_op_type": "delete", "_index": "perf_chunks", "_id": "gone"},
    ]

    def fake_bulk_with_errors(
        client: Any, actions: list[dict[str, Any]], **kwargs: Any
    ) -> tuple[list, list]:
        return [], [
            {"index": {"_id": "a", "error": {"type": "mapper_parsing_exception", "reason": "bad"}}},
            {"delete": {"_id": "gone", "status": 404, "error": {"type": "document_missing"}}},
        ]

    monkeypatch.setattr(sync_module, "bulk_helper", fake_bulk_with_errors)
    errors = sync_module._send(client=object(), actions=actions)
    assert errors == {"a": "mapper_parsing_exception: bad"}
