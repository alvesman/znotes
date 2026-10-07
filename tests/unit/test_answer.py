"""Unit tests for answer generation (task 30, PLAN.md §9 step 7).

``retrieve`` and ``assemble_context`` are monkeypatched to return canned
objects and the :class:`FakeChat` answers are scripted, so the tests stay
offline and deterministic.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fakes import FakeChat
from mynotes.retrieval.analyze import QueryAnalysis
from mynotes.retrieval.answer import (
    AnswerResult,
    answer,
    answer_stream,
    citation_to_dict,
    extract_cited,
)
from mynotes.retrieval.assemble import Citation, ContextBundle
from mynotes.retrieval.pipeline import RetrievalResult
from mynotes.retrieval.prompts import NOT_FOUND
from mynotes.retrieval.types import SearchFilters, SearchHit


def _analysis(language: str | None = None) -> QueryAnalysis:
    return QueryAnalysis(
        raw="pergunta",
        text="pergunta",
        filters=SearchFilters(),
        intent="current",
        as_of=None,
        language=language,
    )


def _hit(rank: int = 1) -> SearchHit:
    return SearchHit(
        chunk_id=f"chunk-{rank}",
        document_id=1,
        version_id=11,
        rank=rank,
        score=0.0,
        text="trecho",
        title="Nota",
        section=None,
        heading_path="",
        path="notes/doc1.md",
        page=None,
        slide=None,
        doc_date=datetime(2026, 1, 1, tzinfo=UTC),
        doc_status="current",
        is_latest=True,
        is_table=False,
        format="md",
        language=None,
        tags=[],
    )


def _result(
    *hits: SearchHit, language: str | None = None, relevant: bool = True
) -> RetrievalResult:
    return RetrievalResult(
        analysis=_analysis(language),
        hits=list(hits),
        candidates=len(hits),
        relevant=relevant,
        timings_ms={"hybrid": 1},
    )


def _citation(n: int, *, is_current: bool = True) -> Citation:
    return Citation(
        n=n,
        document_id=n,
        title=f"Doc {n}",
        path=f"notes/doc{n}.md",
        page=None,
        slide=None,
        section=None,
        doc_date=datetime(2026, 1, 1, tzinfo=UTC),
        is_current=is_current,
    )


def _bundle(citations: list[Citation] | None = None) -> ContextBundle:
    return ContextBundle(
        text="texto do contexto",
        citations=citations if citations is not None else [_citation(1)],
        tokens=100,
        has_table=False,
    )


def _patch(
    monkeypatch: pytest.MonkeyPatch, result: RetrievalResult, bundle: ContextBundle | None = None
) -> None:
    monkeypatch.setattr("mynotes.retrieval.answer.retrieve", lambda query, filters=None: result)
    monkeypatch.setattr(
        "mynotes.retrieval.answer.assemble_context",
        lambda res, **kwargs: bundle if bundle is not None else _bundle(),
    )


def test_irrelevant_retrieval_answers_not_found_without_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, _result(_hit(), language="pt", relevant=False))
    chat = FakeChat(["isto não deve ser chamado"])
    result = answer("qual é o orçamento?", chat=chat)
    assert result.answer == NOT_FOUND["pt"]
    assert result.citations == []
    assert result.context_citations == []
    assert result.relevant is False
    assert chat.calls == []
    assert "llm" not in result.timings_ms


def test_empty_hits_answer_not_found_in_english(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _result(language=None, relevant=True))
    chat = FakeChat(["should not be called"])
    result = answer("what was the budget?", chat=chat)
    assert result.answer == NOT_FOUND["en"]
    assert chat.calls == []


def test_citations_in_first_mention_order_and_invalid_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle([_citation(1), _citation(2), _citation(3)])
    _patch(monkeypatch, _result(_hit(), language="pt"), bundle)
    chat = FakeChat(["O orçamento é 20k€ [2]. Aprovado em março [1][9]."])
    result = answer("orçamento?", chat=chat)
    assert [citation.n for citation in result.citations] == [2, 1]
    assert [citation.n for citation in result.context_citations] == [1, 2, 3]
    assert "llm" in result.timings_ms


def test_answer_without_markers_uses_first_context_citation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle([_citation(1), _citation(2)])
    _patch(monkeypatch, _result(_hit()), bundle)
    chat = FakeChat(["A resposta sem marcadores."])
    result = answer("pergunta", chat=chat)
    assert [citation.n for citation in result.citations] == [1]


def test_html_comments_stripped_from_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _result(_hit()))
    chat = FakeChat(["Texto final [1]. <!-- gerado pelo modelo -->"])
    result = answer("pergunta", chat=chat)
    assert result.answer == "Texto final [1]."


def test_messages_reach_the_chat_built_by_the_prompt_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, _result(_hit(), language="pt"))
    chat = FakeChat(["resposta [1]"])
    answer("pergunta", chat=chat)
    system, user = chat.calls[0]
    assert system.role == "system"
    assert user.role == "user"
    assert "Question: pergunta" in user.content
    assert "Responde em português europeu." in user.content


def test_chat_receives_interactive_priority_and_token_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, _result(_hit()))
    chat = FakeChat(["resposta [1]"])
    answer("pergunta", chat=chat)
    assert chat.kwargs[0]["priority"] == "interactive"
    assert chat.kwargs[0]["max_tokens"] == 1500

    stream_chat = FakeChat(["resposta [1]"])
    list(answer_stream("pergunta", chat=stream_chat))
    assert stream_chat.kwargs[0]["priority"] == "interactive"
    assert stream_chat.kwargs[0]["max_tokens"] == 1500


def test_retrieve_receives_the_question_and_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_retrieve(query: str, filters: SearchFilters | None = None) -> RetrievalResult:
        seen["query"] = query
        seen["filters"] = filters
        return _result(_hit())

    monkeypatch.setattr("mynotes.retrieval.answer.retrieve", fake_retrieve)
    monkeypatch.setattr(
        "mynotes.retrieval.answer.assemble_context", lambda res, **kwargs: _bundle()
    )
    filters = SearchFilters(tags=["projeto"])
    answer("pergunta", filters, chat=FakeChat(["resposta [1]"]))
    assert seen["query"] == "pergunta"
    assert seen["filters"] is filters


def test_stream_yields_tokens_then_sources_then_done(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _bundle([_citation(1), _citation(2)])
    _patch(monkeypatch, _result(_hit()), bundle)
    chat = FakeChat(["Um [1] dois [2]."])
    events = list(answer_stream("pergunta", chat=chat))
    assert [event.type for event in events] == ["token", "token", "token", "sources", "done"]
    tokens = "".join(event.data for event in events if event.type == "token")
    assert tokens == "Um [1] dois [2]."
    sources = next(event for event in events if event.type == "sources")
    assert sources.data == [citation_to_dict(citation) for citation in bundle.citations]
    done = events[-1].data
    assert isinstance(done, AnswerResult)
    assert done.answer == "Um [1] dois [2]."
    assert [citation.n for citation in done.citations] == [1, 2]


def test_stream_gated_answer_yields_token_sources_done(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _result(_hit(), language="en", relevant=False))
    events = list(answer_stream("what budget?"))
    assert [event.type for event in events] == ["token", "sources", "done"]
    assert events[0].data == NOT_FOUND["en"]
    assert events[1].data == []
    assert events[2].data.relevant is False


def test_stream_error_yields_user_safe_error_then_done(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _bundle([_citation(1)])
    _patch(monkeypatch, _result(_hit()), bundle)

    class ExplodingChat:
        def stream(self, messages: object, **kwargs: object) -> object:
            raise RuntimeError("boom: internal detail")

    events = list(answer_stream("pergunta", chat=ExplodingChat()))
    assert [event.type for event in events] == ["error", "done"]
    message = events[0].data
    assert isinstance(message, str) and message
    assert "boom" not in message
    done = events[1].data
    assert done.answer == message
    assert done.citations == []
    assert done.context_citations == list(bundle.citations)


def test_extract_cited_dedupes_in_first_mention_order() -> None:
    citations = [_citation(1), _citation(2)]
    cited = extract_cited("a [2] b [1] c [2] d [7]", citations)
    assert [citation.n for citation in cited] == [2, 1]


def test_extract_cited_with_no_context_citations() -> None:
    assert extract_cited("sem marcadores", []) == []
    assert extract_cited("[1]", []) == []


def test_citation_to_dict_serializes_all_fields() -> None:
    data = citation_to_dict(_citation(2))
    assert list(data) == [
        "n",
        "title",
        "path",
        "page",
        "slide",
        "section",
        "doc_date",
        "is_current",
        "document_id",
        "format",
    ]
    assert data == {
        "n": 2,
        "title": "Doc 2",
        "path": "notes/doc2.md",
        "page": None,
        "slide": None,
        "section": None,
        "doc_date": "2026-01-01T00:00:00+00:00",
        "is_current": True,
        "document_id": 2,
        "format": None,
    }
