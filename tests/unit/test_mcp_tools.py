"""Unit tests for the MCP tool layer (task 46).

The tests call the undecorated ``_*_impl`` implementations directly with the
facade monkeypatched (no database, no model endpoints) and check the compact
JSON shapes, the snippet/markdown truncation and the error-dict contract.
The import test runs a fresh interpreter and asserts nothing reaches stdout
(the MCP protocol channel).
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

import pytest

from mynotes.core import service
from mynotes.core.principal import LOCAL
from mynotes.mcp import server
from mynotes.retrieval.analyze import QueryAnalysis
from mynotes.retrieval.answer import AnswerResult
from mynotes.retrieval.assemble import Citation
from mynotes.retrieval.pipeline import RetrievalResult
from mynotes.retrieval.types import SearchFilters, SearchHit

_SEARCH_KEYS = {
    "rank",
    "title",
    "path",
    "page",
    "slide",
    "section",
    "snippet",
    "doc_date",
    "is_current",
    "document_id",
}


def _hit(**overrides: Any) -> SearchHit:
    values: dict[str, Any] = dict(
        chunk_id="c1",
        document_id=11,
        version_id=3,
        rank=1,
        score=0.01,
        text="word " * 120,
        title="Plano",
        section="Metas",
        heading_path="Plano > Metas",
        path="work/plano.md",
        page=None,
        slide=None,
        doc_date=datetime(2026, 3, 1, tzinfo=UTC),
        doc_status="current",
        is_latest=True,
        is_table=False,
        format="md",
        language="en",
        tags=["project:x"],
    )
    values.update(overrides)
    return SearchHit(**values)


def _analysis(filters: SearchFilters) -> QueryAnalysis:
    return QueryAnalysis(
        raw="q", text="q", filters=filters, intent="current", as_of=None, language=None
    )


def _result(hits: list[SearchHit], filters: SearchFilters) -> RetrievalResult:
    return RetrievalResult(
        analysis=_analysis(filters), hits=hits, candidates=len(hits), relevant=True
    )


def _view(**overrides: Any) -> service.DocumentView:
    values: dict[str, Any] = dict(
        document_id=11,
        title="Plano",
        path="work/plano.md",
        format="md",
        status="current",
        doc_date=datetime(2026, 3, 1, tzinfo=UTC),
        language="en",
        tags=["project:x"],
        latest_version_no=2,
        markdown="y" * 25_000,
        sections=[],
    )
    values.update(overrides)
    return service.DocumentView(**values)


def test_search_notes_truncates_snippets_and_maps_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_search(
        query: str,
        filters: SearchFilters | None = None,
        *,
        principal: Any = LOCAL,
        rerank: bool = True,
    ) -> RetrievalResult:
        seen.update(query=query, filters=filters, principal=principal)
        return _result([_hit(rank=2, text="x" * 700, slide=4)], filters or SearchFilters())

    monkeypatch.setattr(service, "search", fake_search)
    rows = server._search_notes_impl(
        "orçamento tag:project:x", tags=["project:x"], folder="work", limit=5
    )
    assert isinstance(rows, list)
    assert set(rows[0]) == _SEARCH_KEYS
    assert rows[0]["snippet"] == "x" * 500
    assert rows[0]["rank"] == 2
    assert rows[0]["title"] == "Plano"
    assert rows[0]["path"] == "work/plano.md"
    assert rows[0]["slide"] == 4
    assert rows[0]["section"] == "Metas"
    assert rows[0]["doc_date"] == "2026-03-01T00:00:00+00:00"
    assert rows[0]["is_current"] is True
    assert rows[0]["document_id"] == 11
    assert seen["query"] == "orçamento tag:project:x"
    assert seen["filters"].tags == ["project:x"]
    assert seen["filters"].folders == ["work"]
    assert seen["filters"].include_superseded is False
    assert seen["principal"] == LOCAL


def test_search_notes_history_flag_sets_history_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[SearchFilters] = []

    def fake_search(
        query: str,
        filters: SearchFilters | None = None,
        *,
        principal: Any = LOCAL,
        rerank: bool = True,
    ) -> RetrievalResult:
        del query, principal, rerank
        assert filters is not None
        seen.append(filters)
        return _result([], filters)

    monkeypatch.setattr(service, "search", fake_search)
    rows = server._search_notes_impl("versões antigas", history=True)
    assert rows == []
    assert seen[0].include_superseded is True


def test_ask_notes_maps_answer_citations_and_relevant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    citation = Citation(
        n=1,
        document_id=11,
        title="Plano",
        path="work/plano.md",
        page=2,
        slide=None,
        section=None,
        doc_date=datetime(2026, 3, 1, tzinfo=UTC),
        is_current=True,
    )

    def fake_ask(
        question: str, filters: SearchFilters | None = None, *, principal: Any = LOCAL
    ) -> AnswerResult:
        del principal
        return AnswerResult(
            question=question,
            answer="Aprovado em março [1].",
            citations=[citation],
            context_citations=[citation],
            relevant=True,
            retrieval=_result([], filters or SearchFilters()),
        )

    monkeypatch.setattr(service, "ask", fake_ask)
    result = server._ask_notes_impl("qual o orçamento?", history=True)
    assert result["answer"] == "Aprovado em março [1]."
    assert result["relevant"] is True
    assert result["citations"] == [
        {
            "n": 1,
            "title": "Plano",
            "path": "work/plano.md",
            "page": 2,
            "slide": None,
            "doc_date": "2026-03-01T00:00:00+00:00",
            "is_current": True,
        }
    ]


def test_get_document_truncates_markdown_with_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service, "get_document", lambda ref, *, principal=LOCAL: _view())
    result = server._get_document_impl("work/plano.md")
    assert result["document_id"] == 11
    assert result["title"] == "Plano"
    assert result["path"] == "work/plano.md"
    assert result["format"] == "md"
    assert result["status"] == "current"
    assert result["doc_date"] == "2026-03-01T00:00:00+00:00"
    assert result["tags"] == ["project:x"]
    assert result["markdown"].startswith("y" * server._MARKDOWN_CHARS)
    assert "truncated" in result["markdown"]
    assert "25000" in result["markdown"]


def test_get_document_short_markdown_is_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        service, "get_document", lambda ref, *, principal=LOCAL: _view(markdown="# Plano")
    )
    result = server._get_document_impl("work/plano.md")
    assert result["markdown"] == "# Plano"


def test_get_document_ambiguous_returns_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def ambiguous(ref: str, *, principal: Any = LOCAL) -> service.DocumentView:
        del principal
        raise service.Ambiguous(
            "2 documents match 'plano'",
            [
                {"document_id": 1, "title": "Plano A", "path": "a/plano.md"},
                {"document_id": 2, "title": "Plano B", "path": "b/plano.md"},
            ],
        )

    monkeypatch.setattr(service, "get_document", ambiguous)
    result = server._get_document_impl("plano")
    assert result["error"] == "ambiguous"
    assert [candidate["document_id"] for candidate in result["candidates"]] == [1, 2]


def test_facade_exception_returns_error_dict_no_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("database exploded")

    monkeypatch.setattr(service, "search", boom)
    result = server._search_notes_impl("q")
    assert result["error"] == "runtimeerror"
    assert result["message"]


def test_not_found_returns_error_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(ref: str, *, principal: Any = LOCAL) -> service.DocumentView:
        del principal
        raise service.NotFound("no document matches 'x'")

    monkeypatch.setattr(service, "get_document", missing)
    result = server._get_document_impl("x")
    assert result["error"] == "not_found"
    assert result["message"] == "no document matches 'x'"


def test_document_history_maps_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    infos = [
        service.VersionInfo(
            version_no=2,
            created_at=datetime(2026, 5, 1, tzinfo=UTC),
            doc_date=datetime(2026, 4, 1, tzinfo=UTC),
            rel_path="work/plano.md",
            status="embedded",
            chunk_count=9,
            is_latest=True,
            changed_sections=["Metas"],
        )
    ]
    monkeypatch.setattr(service, "document_history", lambda ref, *, principal=LOCAL: infos)
    rows = server._document_history_impl("work/plano.md")
    assert rows == [
        {
            "version_no": 2,
            "created_at": "2026-05-01T00:00:00+00:00",
            "doc_date": "2026-04-01T00:00:00+00:00",
            "changed_sections": ["Metas"],
            "is_latest": True,
        }
    ]


def test_list_tags_maps_summaries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        service,
        "list_tags",
        lambda prefix, *, principal=LOCAL: [
            service.TagView(name="project:x", kind="project", document_count=3, sources=["folder"])
        ],
    )
    rows = server._list_tags_impl("proj")
    assert rows == [{"tag": "project:x", "kind": "project", "documents": 3}]


def test_entity_neighbors_returns_facade_result(monkeypatch: pytest.MonkeyPatch) -> None:
    facade = {
        "entity": {"id": 5, "name": "Projeto X"},
        "hops": 2,
        "neighbors": [{"id": 9, "name": "Ana", "score": 0.5, "types": ["works_on"]}],
    }
    seen: dict[str, Any] = {}

    def fake_neighbors(ref: str, hops: int = 1, *, principal: Any = LOCAL) -> dict[str, Any]:
        seen.update(ref=ref, hops=hops, principal=principal)
        return facade

    monkeypatch.setattr(service, "entity_neighbors", fake_neighbors)
    assert server._entity_neighbors_impl("projeto x", hops=0) == facade
    assert seen == {"ref": "projeto x", "hops": 1, "principal": LOCAL}


def test_project_timeline_maps_items(monkeypatch: pytest.MonkeyPatch) -> None:
    items = [
        service.TimelineItem(
            date=datetime(2026, 3, 9, tzinfo=UTC),
            kind="document",
            text="Plano aprovado",
            status="active",
            document_path="work/plano.md",
        ),
        service.TimelineItem(
            date=None,
            kind="attribute",
            text="estado: pausado",
            status="stale",
            document_path=None,
        ),
    ]
    seen: list[str] = []

    def fake_timeline(ref: str, *, principal: Any = LOCAL) -> list[service.TimelineItem]:
        del principal
        seen.append(ref)
        return items

    monkeypatch.setattr(service, "timeline", fake_timeline)
    rows = server._project_timeline_impl(" beta-launch ")
    assert seen == ["project:beta-launch"]
    assert rows == [
        {
            "date": "2026-03-09",
            "kind": "document",
            "text": "Plano aprovado",
            "status": "active",
            "document_path": "work/plano.md",
        },
        {
            "date": None,
            "kind": "attribute",
            "text": "estado: pausado",
            "status": "stale",
            "document_path": None,
        },
    ]


def test_scan_status_returns_compact_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        service,
        "status",
        lambda *, principal=LOCAL: {
            "config_file": None,
            "workspace": "default",
            "roots": [],
            "postgres": {
                "files_by_status": {"active": 3},
                "documents_by_status": {"current": 2},
                "versions": 4,
                "chunks": 30,
                "jobs": {"queued": 2, "running": 1, "failed": 0},
                "outbox_unprocessed": 5,
            },
            "unsupported": {"count": 0, "examples": []},
            "opensearch": {"aliases": {}, "problems": []},
            "endpoints": [
                {
                    "name": "llm",
                    "state": "closed",
                    "calls": 3,
                    "p50_ms": 900.0,
                    "p95_ms": 1200.0,
                    "error_rate": 0.0,
                    "last_error": None,
                },
                {"error": "unavailable: boom"},
            ],
        },
    )
    monkeypatch.setattr(
        server,
        "_last_scans",
        lambda: [
            {
                "root": "ideas",
                "kind": "incremental",
                "status": "done",
                "started_at": "2026-09-25T08:00:00+00:00",
                "finished_at": "2026-09-25T08:00:03+00:00",
                "counts": {"changed": 1},
                "error": None,
            }
        ],
    )
    monkeypatch.setattr(
        server,
        "_kg_backlog",
        lambda: {
            "queued_kg_extract_jobs": 1,
            "groups_total": 4,
            "groups_cached": 3,
            "groups_pending": 1,
        },
    )
    result = server._scan_status_impl()
    assert set(result) == {"last_scans", "queue", "kg_backlog", "endpoints"}
    assert result["last_scans"][0]["root"] == "ideas"
    assert result["queue"] == {
        "queued": 2,
        "running": 1,
        "failed": 0,
        "outbox_unprocessed": 5,
    }
    assert result["kg_backlog"]["groups_pending"] == 1
    assert result["endpoints"] == [
        {"name": "llm", "state": "closed", "calls": 3, "error_rate": 0.0, "last_error": None}
    ]


def test_note_resource_returns_unquoted_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        service,
        "get_document",
        lambda ref, *, principal=LOCAL: _view(markdown="# Plano\n\ntexto"),
    )
    assert server.note("work%2Fplano.md") == "# Plano\n\ntexto"


def test_note_resource_raises_for_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(ref: str, *, principal: Any = LOCAL) -> service.DocumentView:
        del principal
        raise service.NotFound("no document matches 'x.md'")

    monkeypatch.setattr(service, "get_document", missing)
    with pytest.raises(ValueError, match="no document matches"):
        server.note("x.md")


def test_module_import_writes_nothing_to_stdout() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import mynotes.mcp.server"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout == ""


def test_server_registers_all_tools_and_resource() -> None:
    tools = [tool.name for tool in asyncio.run(server.mcp.list_tools())]
    templates = asyncio.run(server.mcp.list_resource_templates())
    assert tools == [
        "search_notes",
        "ask_notes",
        "get_document",
        "document_history",
        "list_tags",
        "entity_neighbors",
        "project_timeline",
        "scan_status",
    ]
    assert [template.uriTemplate for template in templates] == ["note://{path}"]
