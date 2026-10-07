"""Contract tests for the HTTP API (task 44).

The facade functions are monkeypatched with canned results, so the tests
never touch Postgres, OpenSearch or a model endpoint. They pin the route
contract: schemas, SSE framing, error mapping and the principal handover.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from mynotes.api.app import create_app
from mynotes.core import service
from mynotes.core.principal import LOCAL, Principal
from mynotes.llm.errors import EndpointUnavailable
from mynotes.retrieval.analyze import QueryAnalysis
from mynotes.retrieval.answer import AnswerEvent, AnswerResult
from mynotes.retrieval.assemble import Citation
from mynotes.retrieval.pipeline import RetrievalResult
from mynotes.retrieval.types import SearchFilters
from mynotes.scan.scanner import ScanSummary

CANDIDATES = [
    {"document_id": 1, "title": "Ideia A", "path": "ideias/a.md"},
    {"document_id": 2, "title": "Ideia B", "path": "ideias/b.md"},
]


def _analysis() -> QueryAnalysis:
    return QueryAnalysis(
        raw="q",
        text="q",
        filters=SearchFilters(),
        intent="current",
        as_of=None,
        language="en",
    )


def _retrieval_result() -> RetrievalResult:
    return RetrievalResult(
        analysis=_analysis(), hits=[], candidates=0, relevant=True, timings_ms={"hybrid": 1}
    )


def _citation() -> Citation:
    return Citation(
        n=1,
        document_id=3,
        title="Ideia",
        path="ideias/ideia.md",
        page=None,
        slide=None,
        section="Contexto",
        doc_date=datetime(2026, 5, 1, tzinfo=UTC),
        is_current=True,
    )


def _answer_result() -> AnswerResult:
    return AnswerResult(
        question="q",
        answer="The answer [1].",
        citations=[_citation()],
        context_citations=[_citation()],
        relevant=True,
        retrieval=_retrieval_result(),
        timings_ms={"llm": 5},
    )


def test_ask_returns_the_answer_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[SearchFilters | None] = {}

    def fake_ask(
        question: str,
        filters: SearchFilters | None = None,
        *,
        principal: Principal = LOCAL,
        conversation_id: str | None = None,
    ) -> AnswerResult:
        seen["question"] = question
        seen["filters"] = filters
        return _answer_result()

    monkeypatch.setattr(service, "ask", fake_ask)
    response = TestClient(create_app()).post(
        "/v1/ask", json={"question": "qual e a ideia?", "filters": {"history": True}}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == "The answer [1]."
    assert body["relevant"] is True
    assert body["timings_ms"] == {"llm": 5}
    citation = body["citations"][0]
    assert citation == {
        "n": 1,
        "title": "Ideia",
        "path": "ideias/ideia.md",
        "page": None,
        "slide": None,
        "section": "Contexto",
        "doc_date": "2026-05-01T00:00:00+00:00",
        "is_current": True,
        "document_id": 3,
        "format": None,
    }
    assert seen["question"] == "qual e a ideia?"
    assert seen["filters"] is not None and seen["filters"].include_superseded is True


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    frames = [frame for frame in text.split("\n\n") if frame.strip()]
    parsed: list[tuple[str, dict]] = []
    for frame in frames:
        lines = frame.splitlines()
        event = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        parsed.append((event, data))
    return parsed


def test_ask_stream_frames_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _answer_result()

    def fake_ask_stream(
        question: str,
        filters: SearchFilters | None = None,
        *,
        principal: Principal = LOCAL,
        conversation_id: str | None = None,
    ) -> Iterator[AnswerEvent]:
        yield AnswerEvent(type="token", data="Hel")
        yield AnswerEvent(type="token", data="lo")
        yield AnswerEvent(type="sources", data=[{"n": 1, "path": "ideias/ideia.md"}])
        yield AnswerEvent(type="done", data=result)

    monkeypatch.setattr(service, "ask_stream", fake_ask_stream)
    client = TestClient(create_app())
    with client.stream("POST", "/v1/ask/stream", json={"question": "q"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["x-accel-buffering"] == "no"
        text = "".join(response.iter_text())
    events = _parse_sse(text)
    assert [event for event, _data in events] == ["token", "token", "sources", "done"]
    assert events[0][1] == {"text": "Hel"}
    assert events[2][1] == {"citations": [{"n": 1, "path": "ideias/ideia.md"}]}
    assert events[3][1] == {"relevant": True, "timings_ms": {"llm": 5}}


def test_ask_stream_emits_error_and_done_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_ask_stream(
        question: str,
        filters: SearchFilters | None = None,
        *,
        principal: Principal = LOCAL,
        conversation_id: str | None = None,
    ) -> Iterator[AnswerEvent]:
        yield AnswerEvent(type="token", data="par")
        raise RuntimeError("boom")

    monkeypatch.setattr(service, "ask_stream", fake_ask_stream)
    client = TestClient(create_app(), raise_server_exceptions=False)
    text = client.post("/v1/ask/stream", json={"question": "q"}).text
    events = _parse_sse(text)
    assert [event for event, _data in events] == ["token", "error", "done"]
    assert events[1][1]["message"]
    assert "boom" not in text
    assert events[2][1] == {"relevant": False, "timings_ms": {}}


def test_lookup_ambiguous_returns_409_with_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_document(ref: str, *, principal: Principal = LOCAL) -> None:
        raise service.Ambiguous("2 documents match 'x'", CANDIDATES)

    monkeypatch.setattr(service, "get_document", fake_get_document)
    response = TestClient(create_app()).get("/v1/documents/lookup", params={"ref": "x"})
    assert response.status_code == 409
    body = response.json()
    assert body["error"] == "ambiguous"
    assert body["candidates"] == CANDIDATES


def test_lookup_not_found_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_document(ref: str, *, principal: Principal = LOCAL) -> None:
        raise service.NotFound("no document matches 'x'")

    monkeypatch.setattr(service, "get_document", fake_get_document)
    response = TestClient(create_app()).get("/v1/documents/lookup", params={"ref": "x"})
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_endpoint_unavailable_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_search(
        query: str,
        filters: SearchFilters | None = None,
        *,
        principal: Principal = LOCAL,
        rerank: bool = True,
    ) -> None:
        raise EndpointUnavailable("embeddings", None, "circuit breaker is open")

    monkeypatch.setattr(service, "search", fake_search)
    response = TestClient(create_app()).post("/v1/search", json={"query": "q"})
    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "endpoint_unavailable"
    assert "circuit breaker" not in body["detail"]


def test_unexpected_error_returns_generic_500(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_status(*, principal: Principal = LOCAL) -> dict:
        raise RuntimeError("internal detail hunter2")

    monkeypatch.setattr(service, "status", fake_status)
    client = TestClient(create_app(), raise_server_exceptions=False)
    response = client.get("/v1/status")
    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "internal_error"
    assert "internal detail" not in response.text
    assert "hunter2" not in response.text


def test_validation_error_returns_422() -> None:
    response = TestClient(create_app()).post("/v1/ask", json={})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == "question"


def test_search_applies_limit_and_reports_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service, "search", lambda *args, **kwargs: _retrieval_result())
    response = TestClient(create_app()).post("/v1/search", json={"query": "q", "limit": 3})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["hits"] == []
    assert body["analysis"] == {
        "intent": "current",
        "text": "q",
        "filters": {
            "tags": [],
            "folders": [],
            "formats": [],
            "since": None,
            "until": None,
            "history": False,
        },
    }


def test_scan_returns_202_and_runs_off_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    called = threading.Event()
    seen: dict[str, object] = {}

    def fake_trigger_scan(
        root: str | None = None, full: bool = False, *, principal: Principal = LOCAL
    ) -> list[ScanSummary]:
        seen["root"] = root
        seen["full"] = full
        called.set()
        return [ScanSummary(scan_run_id=1, counts={"new": 1}, errors=[])]

    monkeypatch.setattr(service, "trigger_scan", fake_trigger_scan)
    response = TestClient(create_app()).post("/v1/scan", json={"root": "ideas", "full": True})
    assert response.status_code == 202
    assert response.json()["accepted"] is True
    assert called.wait(timeout=5)
    assert seen == {"root": "ideas", "full": True}


def test_healthz_reports_down_services(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mynotes.api.routes.admin._ping_postgres", lambda: False)
    monkeypatch.setattr("mynotes.api.routes.admin._ping_opensearch", lambda: False)
    response = TestClient(create_app()).get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "postgres": False, "opensearch": False}


def _document_view() -> service.DocumentView:
    return service.DocumentView(
        document_id=5,
        title="Ideia",
        path="ideias/ideia.md",
        format="md",
        status="current",
        doc_date=None,
        language=None,
        tags=["project:alfa"],
        latest_version_no=2,
        markdown="# Ideia",
        sections=[service.SectionRef(section="Ideia", page=None, slide=None, first_ordinal=0)],
    )


def _spy_result(name: str) -> object:
    if name == "ask":
        return _answer_result()
    if name == "ask_stream":
        return iter([AnswerEvent(type="done", data=_answer_result())])
    if name == "search":
        return _retrieval_result()
    if name == "get_document":
        return _document_view()
    if name == "entity_card":
        return {"id": 1, "name": "Alfa", "type": "project"}
    if name == "entity_neighbors":
        return {"entity": {"id": 1, "name": "Alfa"}, "hops": 1, "neighbors": []}
    return {}


_ROUTE_CASES = [
    ("POST", "/v1/ask", {"question": "q"}, "ask"),
    ("POST", "/v1/ask/stream", {"question": "q"}, "ask_stream"),
    ("POST", "/v1/search", {"query": "q"}, "search"),
    ("GET", "/v1/documents/lookup?ref=x", None, "get_document"),
    ("GET", "/v1/documents/5", None, "get_document"),
    ("GET", "/v1/documents/5/history", None, "document_history"),
    ("GET", "/v1/tags", None, "list_tags"),
    ("POST", "/v1/tags/5/foo/accept", None, "tag_action"),
    ("POST", "/v1/tags/merge", {"from": "a", "into": "b"}, "merge_tags"),
    ("GET", "/v1/versions", None, "list_version_relations"),
    ("POST", "/v1/versions/7/confirm", None, "version_action"),
    ("GET", "/v1/graph/entities?q=alfa", None, "entity_card"),
    ("GET", "/v1/graph/entities/alfa/neighbors", None, "entity_neighbors"),
    ("GET", "/v1/graph/timeline?entity=alfa", None, "timeline"),
    ("GET", "/v1/status", None, "status"),
]


@pytest.mark.parametrize(("method", "path", "body", "name"), _ROUTE_CASES)
def test_every_route_receives_the_local_principal(
    monkeypatch: pytest.MonkeyPatch, method: str, path: str, body: dict | None, name: str
) -> None:
    seen: dict[str, Principal | None] = {}

    def recorder(*args: object, **kwargs: object) -> object:
        seen["principal"] = kwargs.get("principal")  # type: ignore[assignment]
        return _spy_result(name)

    monkeypatch.setattr(service, name, recorder)
    response = TestClient(create_app()).request(method, path, json=body)
    assert response.status_code == 200, response.text
    assert seen["principal"] is LOCAL


def test_document_view_serializes_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "get_document", lambda *args, **kwargs: _document_view())
    response = TestClient(create_app()).get("/v1/documents/5")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["document_id"] == 5
    assert body["sections"] == [
        {"section": "Ideia", "page": None, "slide": None, "first_ordinal": 0}
    ]
    assert body["tags"] == ["project:alfa"]


def test_ask_stream_saves_the_finished_chat(
    monkeypatch: pytest.MonkeyPatch, no_chat_history_writes: list
) -> None:
    def fake_ask_stream(
        question: str,
        filters: SearchFilters | None = None,
        *,
        principal: Principal = LOCAL,
        conversation_id: str | None = None,
    ) -> Iterator[AnswerEvent]:
        yield AnswerEvent(type="token", data="Port")
        yield AnswerEvent(type="token", data="ainer [1]")
        yield AnswerEvent(type="sources", data=[{"n": 1, "path": "Setup/Portainer.md"}])
        yield AnswerEvent(type="done", data=_answer_result())

    monkeypatch.setattr(service, "ask_stream", fake_ask_stream)
    client = TestClient(create_app())
    client.post(
        "/v1/ask/stream",
        json={"question": "portainer", "filters": {"history": True}, "conversation_id": "c1"},
    )
    assert len(no_chat_history_writes) == 1
    saved = no_chat_history_writes[0]
    assert saved["conversation_id"] == "c1"
    assert (saved["question"], saved["answer"]) == ("portainer", "Portainer [1]")
    assert saved["citations"] == [{"n": 1, "path": "Setup/Portainer.md"}]
    assert saved["filters"]["history"] is True and saved["relevant"] is True
    # opting out, and a failed answer, store nothing
    client.post("/v1/ask/stream", json={"question": "q", "save_history": False})
    assert len(no_chat_history_writes) == 1


def test_ask_stream_sends_and_stores_the_rewritten_question(
    monkeypatch: pytest.MonkeyPatch, no_chat_history_writes: list
) -> None:
    def fake_ask_stream(question, filters=None, *, principal=LOCAL, conversation_id=None):
        yield AnswerEvent(type="rewritten", data="Que porta usa o Portainer?")
        yield AnswerEvent(type="token", data="9443")
        yield AnswerEvent(type="sources", data=[])
        yield AnswerEvent(type="done", data=_answer_result())

    monkeypatch.setattr(service, "ask_stream", fake_ask_stream)
    text = (
        TestClient(create_app())
        .post("/v1/ask/stream", json={"question": "e a porta?", "conversation_id": "c1"})
        .text
    )
    events = _parse_sse(text)
    assert events[0] == ("rewritten", {"question": "Que porta usa o Portainer?"})
    assert no_chat_history_writes[-1]["standalone_question"] == "Que porta usa o Portainer?"
