"""Unit tests for version-aware retrieval (task 35): expander, notes, recency prior."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from fakes import FakeReranker
from mynotes.retrieval import version_expand
from mynotes.retrieval.rerank import _project_recency_multipliers, rerank_hits
from mynotes.retrieval.types import SearchFilters, SearchHit


def _hit(
    chunk_id: str,
    document_id: int,
    *,
    doc_status: str = "current",
    doc_date: datetime | None = None,
    tags: list[str] | None = None,
    path: str | None = None,
    text: str = "corpo do trecho da nota",
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id=document_id,
        version_id=document_id * 10,
        rank=1,
        score=0.0,
        text=text,
        title=None,
        section=None,
        heading_path="",
        path=path or f"notes/doc{document_id}.md",
        page=None,
        slide=None,
        doc_date=doc_date,
        doc_status=doc_status,
        is_latest=True,
        is_table=False,
        format="md",
        language=None,
        tags=tags or [],
    )


def _analysis(intent: str = "current", filters: SearchFilters | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        text="orcamento da ideia",
        filters=filters if filters is not None else SearchFilters(),
        intent=intent,
    )


@contextmanager
def _dummy_session() -> Any:
    yield object()


def test_current_intent_expands_an_archived_hit_to_its_head(
    monkeypatch: Any,
) -> None:
    """A hit from an archived document pulls in the head document's chunks."""
    monkeypatch.setattr(version_expand, "session_scope", _dummy_session)
    monkeypatch.setattr(
        version_expand,
        "head_of_chain",
        lambda session, document_id: 202 if document_id == 101 else document_id,
    )
    captured: dict[str, Any] = {}

    def fake_hybrid_search(query: str, filters: SearchFilters, *, size: int = 0) -> list[SearchHit]:
        captured["query"] = query
        captured["filters"] = filters
        captured["size"] = size
        return [_hit("new-1", 202, path="idea_v2.md")]

    monkeypatch.setattr(version_expand, "hybrid_search", fake_hybrid_search)

    out = version_expand.VersionExpander().expand(
        _analysis(), [_hit("old-1", 101, doc_status="archived", path="idea.md")]
    )

    assert [hit.chunk_id for hit in out] == ["new-1"]
    assert captured["query"] == "orcamento da ideia"
    assert captured["filters"].document_ids == [202]
    assert captured["size"] == 3


def test_history_intent_searches_whole_chains_with_superseded(
    monkeypatch: Any,
) -> None:
    """The history intent searches every chain member, superseded included."""
    monkeypatch.setattr(version_expand, "session_scope", _dummy_session)
    monkeypatch.setattr(
        version_expand, "_chain_members", lambda session, document_id: [101, 102, 202]
    )
    captured: dict[str, Any] = {}

    def fake_hybrid_search(query: str, filters: SearchFilters, *, size: int = 0) -> list[SearchHit]:
        captured["filters"] = filters
        captured["size"] = size
        return [_hit("old-1", 101, doc_status="superseded", path="idea.md")]

    monkeypatch.setattr(version_expand, "hybrid_search", fake_hybrid_search)

    out = version_expand.VersionExpander().expand(
        _analysis("history"), [_hit("h-1", 202, path="idea_v2.md")]
    )

    assert [hit.chunk_id for hit in out] == ["old-1"]
    assert captured["filters"].document_ids == [101, 102, 202]
    assert captured["filters"].include_superseded is True
    assert captured["filters"].include_old_versions is False
    assert captured["size"] == 3


def test_expander_ignores_documents_without_a_chain(monkeypatch: Any) -> None:
    """Documents that are their own head and have no duplicate never expand."""
    monkeypatch.setattr(version_expand, "session_scope", _dummy_session)
    monkeypatch.setattr(version_expand, "head_of_chain", lambda session, document_id: document_id)
    monkeypatch.setattr(version_expand, "_duplicate_targets", lambda session, ids: {})

    def failing_search(query: str, filters: SearchFilters, *, size: int = 0) -> list[SearchHit]:
        raise AssertionError("hybrid_search must not be called")

    monkeypatch.setattr(version_expand, "hybrid_search", failing_search)

    out = version_expand.VersionExpander().expand(_analysis(), [_hit("h-1", 5)])

    assert out == []


def test_version_notes_renders_relation_lines(monkeypatch: Any) -> None:
    """The version notes block carries one line per active relation."""
    monkeypatch.setattr(version_expand, "session_scope", _dummy_session)
    monkeypatch.setattr(
        version_expand,
        "_relation_lines",
        lambda session, document_ids, acl: ['"idea_v2.md" supersedes "idea.md" (2026-09-10)'],
    )
    monkeypatch.setattr(version_expand, "_timeline_lines", lambda session, intent, entries: [])
    result = SimpleNamespace(analysis=SimpleNamespace(intent="current", filters=SearchFilters()))

    block = version_expand.VersionNotes().block(result, [_hit("c-1", 202, path="idea_v2.md")])

    assert block == '"idea_v2.md" supersedes "idea.md" (2026-09-10)'


def test_version_notes_returns_none_without_chains(monkeypatch: Any) -> None:
    """Nothing to say means no block at all."""
    monkeypatch.setattr(version_expand, "session_scope", _dummy_session)
    monkeypatch.setattr(version_expand, "_relation_lines", lambda session, document_ids, acl: [])
    monkeypatch.setattr(version_expand, "_timeline_lines", lambda session, intent, entries: [])
    result = SimpleNamespace(analysis=SimpleNamespace(intent="current", filters=SearchFilters()))

    block = version_expand.VersionNotes().block(result, [_hit("c-1", 5)])

    assert block is None


def test_recency_prior_boosts_the_newer_document_of_a_project_pair() -> None:
    """Two documents sharing a project tag: the newest gets the 5% boost."""
    older = _hit(
        "old",
        1,
        doc_date=datetime(2026, 3, 1, tzinfo=UTC),
        tags=["project:alfa"],
        text="orcamento da ideia",
    )
    newer = _hit(
        "new",
        2,
        doc_date=datetime(2026, 9, 1, tzinfo=UTC),
        tags=["project:alfa"],
        text="orcamento da ideia",
    )

    out = rerank_hits("orcamento da ideia", [older, newer], FakeReranker(), top=5)

    assert out[0].chunk_id == "new"
    assert out[1].chunk_id == "old"
    assert out[0].rerank_score == out[1].rerank_score
    assert [hit.rank for hit in out] == [1, 2]
    multipliers = _project_recency_multipliers([older, newer])
    assert multipliers == {2: 1.05}


def test_recency_prior_needs_two_known_dates_in_the_group() -> None:
    """A single known doc_date (or no project tag) never boosts anyone."""
    dated = _hit("a", 1, doc_date=datetime(2026, 9, 1, tzinfo=UTC), tags=["project:alfa"])
    undated = _hit("b", 2, tags=["project:alfa"])
    untagged = _hit("c", 3, doc_date=datetime(2026, 1, 1, tzinfo=UTC))
    other_tag = _hit("d", 4, doc_date=datetime(2026, 2, 1, tzinfo=UTC), tags=["topic:x"])

    assert _project_recency_multipliers([dated, undated, untagged, other_tag]) == {}
