"""Unit tests for the server-rendered web UI (task 47).

The facade is monkeypatched with canned results, so the tests never touch
Postgres, OpenSearch or a model endpoint. They pin the page contract: the
routes, the CSP header, the escaping of model text and the per-format
citation actions.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from mynotes.api.app import create_app
from mynotes.api.auth import get_principal
from mynotes.core import service
from mynotes.core.principal import LOCAL, Principal
from mynotes.retrieval.analyze import QueryAnalysis
from mynotes.retrieval.pipeline import RetrievalResult
from mynotes.retrieval.types import SearchFilters, SearchHit
from mynotes.tags.store import TagSummary
from mynotes.web.routes import STATIC_DIR

CSP = (
    "default-src 'self'; img-src 'self' data:; style-src 'self'; "
    "script-src 'self'; frame-ancestors 'none'"
)


def _hit(**overrides: object) -> SearchHit:
    values: dict[str, object] = {
        "chunk_id": "1:2",
        "document_id": 5,
        "version_id": 9,
        "rank": 1,
        "score": 0.02,
        "text": "texto do resultado",
        "title": "Plano",
        "section": "Metas",
        "heading_path": "Plano > Metas",
        "path": "work/plano.md",
        "page": None,
        "slide": None,
        "doc_date": datetime(2026, 3, 1, tzinfo=UTC),
        "doc_status": "current",
        "is_latest": True,
        "is_table": False,
        "format": "md",
        "language": "pt",
        "tags": ["project:alfa"],
    }
    values.update(overrides)
    return SearchHit(**values)  # type: ignore[arg-type]


def _result(hits: list[SearchHit]) -> RetrievalResult:
    return RetrievalResult(
        analysis=QueryAnalysis(
            raw="q", text="q", filters=SearchFilters(), intent="current", as_of=None, language="en"
        ),
        hits=hits,
        candidates=len(hits),
        relevant=True,
        timings_ms={"hybrid": 1},
    )


def _client() -> TestClient:
    return TestClient(create_app())


def test_root_redirects_to_chat() -> None:
    response = _client().get("/", follow_redirects=False)
    assert response.status_code in (301, 302, 307, 308)
    assert response.headers["location"] == "/ui/chat"


def test_chat_page_renders_form_and_script() -> None:
    response = _client().get("/ui/chat")
    assert response.status_code == 200, response.text
    assert 'id="ask-form"' in response.text
    assert "<textarea" in response.text
    assert 'id="history"' in response.text
    assert 'id="tags"' in response.text
    assert 'src="/static/chat.js?v=' in response.text


def test_search_page_lists_tag_options(monkeypatch: pytest.MonkeyPatch) -> None:
    tags = [
        TagSummary(name="aaa-unused", kind="other", document_count=0, sources=[]),
        TagSummary(name="project:alfa", kind="project", document_count=3, sources=["manual"]),
        TagSummary(name="topic:kg", kind="topic", document_count=1, sources=["llm"]),
        TagSummary(name="zeta", kind="other", document_count=7, sources=["folder"]),
    ]
    monkeypatch.setattr(service, "list_tags", lambda *args, **kwargs: tags)
    response = _client().get("/ui/search")
    assert response.status_code == 200, response.text
    html = response.text
    assert 'value="project:alfa"' in html
    assert 'value="topic:kg"' in html
    # Tags nobody uses are hidden; the rest are ordered most used first.
    assert 'value="aaa-unused"' not in html
    positions = [html.index(f'value="{name}"') for name in ("zeta", "project:alfa", "topic:kg")]
    assert positions == sorted(positions)
    # The tag picker is a list of checkboxes at the bottom, with a filter box.
    assert 'type="checkbox" name="tags"' in html
    assert "<select" not in html
    assert html.index('name="formats"') < html.index('name="tags"')
    assert html.index('name="rerank"') < html.index('class="tag-filter"')
    assert 'id="tag-filter"' in html and 'id="tag-clear"' in html
    assert 'src="/static/search.js?v=' in html
    assert html.index('src="/static/markdown.js?v=') < html.index('src="/static/search.js?v=')
    assert 'hx-post="/ui/partials/search"' in response.text
    assert 'hx-target="#results"' in response.text
    for fmt in ("md", "pdf", "docx", "epub", "html"):
        assert f'value="{fmt}"' in response.text


def test_search_page_without_tags_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "list_tags", lambda *args, **kwargs: [])
    response = _client().get("/ui/search")
    assert response.status_code == 200, response.text
    assert "No tags yet." in response.text
    assert 'id="tag-filter"' not in response.text


def test_search_script_is_served() -> None:
    response = _client().get("/static/search.js")
    assert response.status_code == 200
    assert "requestSubmit" in response.text


def test_partial_search_renders_hits_and_escapes_snippet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_search(
        query: str,
        filters: SearchFilters | None = None,
        *,
        principal: Principal = LOCAL,
        rerank: bool = True,
    ) -> RetrievalResult:
        seen["query"] = query
        seen["filters"] = filters
        seen["principal"] = principal
        seen["rerank"] = rerank
        return _result(
            [
                _hit(
                    text=f"<script>alert(1)</script> {('palavra ' * 60)}",
                    format="pdf",
                    page=4,
                )
            ]
        )

    monkeypatch.setattr(service, "search", fake_search)
    response = _client().post(
        "/ui/partials/search",
        data={
            "query": "orçamento #tag:x",
            "tags": "project:alfa",
            "folder": "work",
            "formats": ["md", "bogus"],
            "since": "2026-09",
            "history": "1",
            "rerank": "1",
        },
    )
    assert response.status_code == 200, response.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text
    assert "<script>alert(1)</script>" not in response.text
    assert 'href="/v1/files/5/raw#page=4"' in response.text
    assert "historical" not in response.text
    filters = seen["filters"]
    assert isinstance(filters, SearchFilters)
    assert filters.tags == ["project:alfa"]
    assert filters.folders == ["work"]
    assert filters.formats == ["md"]
    assert filters.since == datetime(2026, 9, 1, tzinfo=UTC)
    assert filters.include_superseded is True
    assert seen["query"] == "orçamento #tag:x"
    assert seen["rerank"] is True
    assert seen["principal"] == LOCAL


def test_empty_query_renders_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_search(*args: object, **kwargs: object) -> RetrievalResult:
        raise AssertionError("search must not run for an empty query")

    monkeypatch.setattr(service, "search", fail_search)
    response = _client().post("/ui/partials/search", data={"query": "   "})
    assert response.status_code == 200, response.text
    assert "No results." in response.text


def test_csp_header_on_ui_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "list_tags", lambda *args, **kwargs: [])
    monkeypatch.setattr(service, "search", lambda *args, **kwargs: _result([]))
    client = _client()
    for method, path, kwargs in (
        ("get", "/ui/chat", {}),
        ("get", "/ui/search", {}),
        ("post", "/ui/partials/search", {"data": {"query": "q"}}),
    ):
        response = getattr(client, method)(path, **kwargs)
        assert response.status_code == 200, response.text
        assert response.headers["content-security-policy"] == CSP


def test_team_principal_hides_open_in_app(monkeypatch: pytest.MonkeyPatch) -> None:
    app = create_app()
    app.dependency_overrides[get_principal] = lambda: Principal(
        user="team", acl_prefixes=[], can_open_files=False
    )

    def fake_search(
        query: str,
        filters: SearchFilters | None = None,
        *,
        principal: Principal = LOCAL,
        rerank: bool = True,
    ) -> RetrievalResult:
        return _result(
            [_hit(document_id=7, format="docx", page=None, slide=2, section="Resultados")]
        )

    monkeypatch.setattr(service, "search", fake_search)
    response = TestClient(app).post("/ui/partials/search", data={"query": "q"})
    assert response.status_code == 200, response.text
    assert "Open in app" not in response.text
    assert 'href="/v1/files/7/raw"' in response.text
    assert 'href="/ui/documents/7#sec-resultados"' in response.text


def test_local_principal_shows_open_in_app(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_search(
        query: str,
        filters: SearchFilters | None = None,
        *,
        principal: Principal = LOCAL,
        rerank: bool = True,
    ) -> RetrievalResult:
        return _result([_hit(document_id=7, format="docx", page=None, slide=2)])

    monkeypatch.setattr(service, "search", fake_search)
    monkeypatch.setattr(service, "remote_document_ids", lambda ids: set())
    response = _client().post("/ui/partials/search", data={"query": "q"})
    assert response.status_code == 200, response.text
    assert "Open in app" in response.text
    assert 'hx-post="/v1/files/7/open"' in response.text


def test_uploaded_note_gets_no_open_in_app(monkeypatch: pytest.MonkeyPatch) -> None:
    hits = [_hit(document_id=7, format="docx"), _hit(document_id=8, format="xlsx")]
    asked: list[set[int]] = []

    def fake_remote(ids: object) -> set[int]:
        asked.append(set(ids))  # type: ignore[call-overload]
        return {7}  # document 7 was uploaded by an agent

    monkeypatch.setattr(service, "search", lambda *args, **kwargs: _result(hits))
    monkeypatch.setattr(service, "remote_document_ids", fake_remote)
    response = _client().post("/ui/partials/search", data={"query": "q"})
    assert response.status_code == 200, response.text
    assert asked == [{7, 8}]
    assert 'hx-post="/v1/files/7/open"' not in response.text
    assert 'hx-post="/v1/files/8/open"' in response.text
    assert "/v1/files/7/raw" in response.text  # Download stays


def test_status_is_for_admins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "status", lambda **kwargs: pytest.fail("status was built"))
    bob = Principal(user="bob", acl_prefixes=["bob"])

    def as_bob(request: Request) -> Principal:
        request.state.principal = bob  # as get_principal does, for the menu
        return bob

    app = create_app()
    app.dependency_overrides[get_principal] = as_bob
    client = TestClient(app)
    page = client.get("/ui/status")
    assert page.status_code == 403
    assert "for the administrator" in page.text and "Scan now" not in page.text
    chat = client.get("/ui/chat")
    assert 'href="/ui/status"' not in chat.text and 'href="/ui/reviews"' in chat.text


def test_local_mode_menu_keeps_status() -> None:
    assert 'href="/ui/status"' in _client().get("/ui/chat").text


def test_static_files_are_served() -> None:
    client = _client()
    response = client.get("/static/chat.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    htmx_files = sorted(STATIC_DIR.glob("htmx-*.min.js"))
    assert len(htmx_files) == 1
    response = client.get(f"/static/{htmx_files[0].name}")
    assert response.status_code == 200
    assert len(response.content) > 10000


def test_historical_hit_gets_badge(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_search(
        query: str,
        filters: SearchFilters | None = None,
        *,
        principal: Principal = LOCAL,
        rerank: bool = True,
    ) -> RetrievalResult:
        return _result([_hit(doc_status="archived", is_latest=False, document_id=8)])

    monkeypatch.setattr(service, "search", fake_search)
    response = _client().post("/ui/partials/search", data={"query": "q"})
    assert response.status_code == 200, response.text
    assert "historical" in response.text


def test_far_weaker_search_results_are_collapsed(monkeypatch: pytest.MonkeyPatch) -> None:
    from mynotes.web.routes import split_weak_hits

    hits = [
        _hit(chunk_id="a", document_id=1, rerank_score=0.84),
        _hit(chunk_id="b", document_id=1, rerank_score=0.79),
        _hit(chunk_id="c", document_id=2, rerank_score=0.002),
    ]
    strong, weak = split_weak_hits(hits)
    assert [h.chunk_id for h in strong] == ["a", "b"] and [h.chunk_id for h in weak] == ["c"]
    # without rerank scores nothing is hidden
    plain = [_hit(chunk_id="x"), _hit(chunk_id="y")]
    assert split_weak_hits(plain) == (plain, [])

    def fake_search(query, filters, *, principal=None, rerank=True):
        return _result(hits)

    monkeypatch.setattr(service, "search", fake_search)
    html = _client().post("/ui/partials/search", data={"query": "portainer", "rerank": "1"}).text
    assert html.count('class="hit"') == 3
    assert html.index('class="weak-hits"') > html.rindex('href="/ui/documents/1')
    assert "Show 1 weaker result</summary>" in html


def _chat_record(**overrides: object):
    from mynotes.core.chat_history import ChatRecord

    values = {
        "id": 7,
        "question": "portainer",
        "answer": "**Portainer** manages containers [1].",
        "citations": [
            {
                "n": 1,
                "title": "Portainer",
                "path": "Setup/Portainer.md",
                "page": None,
                "slide": None,
                "section": None,
                "doc_date": "2026-01-05T00:00:00+00:00",
                "is_current": True,
                "document_id": 389,
                "format": "md",
            }
        ],
        "filters": {"history": True},
        "relevant": True,
        "created_at": datetime(2026, 10, 7, 9, 30, tzinfo=UTC),
    }
    values.update(overrides)
    return ChatRecord(**values)  # type: ignore[arg-type]


def test_history_page_groups_a_chat_into_one_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mynotes.core.chat_history import Conversation

    first = _chat_record()
    second = _chat_record(
        id=8,
        question="and the port?",
        answer="9443 [1]",
        created_at=datetime(2026, 10, 7, 9, 31, tzinfo=UTC),
    )
    monkeypatch.setattr(
        "mynotes.core.chat_history.list_conversations",
        lambda principal: ([Conversation(key="abc", turns=[first, second])], 1),
    )
    html = _client().get("/ui/history").text
    assert html.count('class="history-entry"') == 1 and html.count('class="history-turn"') == 2
    assert "2 questions" in html and "and the port?" in html
    assert 'data-cite-prefix="h8-cite-"' in html and 'id="h8-cite-1"' in html
    assert 'href="/ui/history" aria-current="page"' in html
    assert 'hx-post="/ui/partials/history/delete"' in html and "hx-confirm=" in html
    assert 'name="days"' in html and 'value="all"' in html
    assert "portainer" in html and 'data-cite-prefix="h7-cite-"' in html
    assert 'id="h7-cite-1"' in html and "2026-01-05" in html
    assert "T00:00:00" not in html
    assert 'src="/static/markdown.js?v=' in html and 'src="/static/history.js?v=' in html


def test_history_delete_scopes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def fake_delete(*, older_than_days=None, principal=None):
        calls.append(older_than_days)
        return 3

    monkeypatch.setattr("mynotes.core.chat_history.delete_chats", fake_delete)
    monkeypatch.setattr("mynotes.core.chat_history.list_conversations", lambda principal: ([], 0))
    client = _client()
    assert (
        "Deleted 3 chats not used for 30 days."
        in client.post("/ui/partials/history/delete", data={"scope": "older", "days": "30"}).text
    )
    assert (
        "Deleted all chats (3)."
        in client.post("/ui/partials/history/delete", data={"scope": "all"}).text
    )
    assert (
        "Enter a number of days"
        in client.post("/ui/partials/history/delete", data={"scope": "older", "days": "0"}).text
    )
    assert calls == [30, None]


def test_chat_and_search_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "list_tags", lambda *args, **kwargs: [])
    client = _client()
    assert "Include old notes versions</label>" in client.get("/ui/chat").text
    assert "include notes history</label>" in client.get("/ui/search").text


def test_option_checkboxes_are_remembered_per_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    tag = TagSummary(name="project:alfa", kind="project", document_count=3, sources=["manual"])
    monkeypatch.setattr(service, "list_tags", lambda *args, **kwargs: [tag])
    client = _client()
    chat = client.get("/ui/chat").text
    search = client.get("/ui/search").text
    assert 'data-persist="chat.history"' in chat and 'src="/static/prefs.js?v=' in chat
    assert 'data-persist="search.history"' in search and 'data-persist="search.rerank"' in search
    assert 'src="/static/prefs.js?v=' in search
    # filters are not remembered: a stored tag would silently narrow the next search
    assert 'name="tags" value' in search and search.count("data-persist=") == 2


def test_continue_a_chat_from_history(monkeypatch: pytest.MonkeyPatch) -> None:
    from mynotes.core.chat_history import Conversation

    found = Conversation(key="abc-123", turns=[_chat_record()])
    monkeypatch.setattr(
        "mynotes.core.chat_history.get_conversation",
        lambda key, principal: found if key == "abc-123" else None,
    )
    monkeypatch.setattr(
        "mynotes.core.chat_history.list_conversations", lambda principal: ([found], 1)
    )
    client = _client()
    assert (
        'href="/ui/chat?conversation=abc-123">Continue this chat</a>'
        in client.get("/ui/history").text
    )
    resumed = client.get("/ui/chat", params={"conversation": "abc-123"}).text
    assert 'data-conversation-id="abc-123"' in resumed
    assert 'class="qa resumed"' in resumed and "portainer" in resumed
    assert 'id="h7-cite-1"' in resumed and "Start a new chat" in resumed
    # an unknown key (or another person's chat) opens an empty chat
    fresh = client.get("/ui/chat", params={"conversation": "nope"}).text
    assert "data-conversation-id" not in fresh and 'class="qa resumed"' not in fresh
