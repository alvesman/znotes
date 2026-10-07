"""Answer generation over the assembled context (PLAN.md §9 step 7).

:func:`answer` runs the retrieval pipeline, gates on its relevance verdict and
generates one grounded, cited answer with the chat model. :func:`answer_stream`
does the same while yielding :class:`AnswerEvent` pieces for the CLI and later
the HTTP API: tokens as they arrive, one ``sources`` event with the serialized
citations, then ``done`` with the full :class:`AnswerResult` (or ``error`` with
a user-safe message when generation fails).
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from mynotes.llm.chat import ChatMessage, ChatModel, get_chat
from mynotes.retrieval.assemble import Citation, assemble_context
from mynotes.retrieval.pipeline import RetrievalResult, retrieve
from mynotes.retrieval.prompts import NOT_FOUND, build_messages
from mynotes.retrieval.types import SearchFilters

logger = logging.getLogger(__name__)

_MAX_ANSWER_TOKENS = 1500
"""Output cap for one answer."""

_CITATION_RE = re.compile(r"\[(\d+)\]")
"""A ``[n]`` citation marker inside the answer text."""

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
"""A stray HTML comment the model may emit."""

_STREAM_ERROR_MESSAGE = (
    "The answer could not be generated because the model endpoint returned an error. "
    "Please try again."
)


@dataclass
class AnswerResult:
    """The complete outcome of one :func:`answer` call.

    Attributes:
        question: The question exactly as passed in.
        answer: The answer text, with HTML comments stripped.
        citations: The context citations the answer actually cites, in
            first-mention order.
        context_citations: Everything that was in the context.
        relevant: The retrieval relevance verdict.
        retrieval: The underlying retrieval result.
        timings_ms: Milliseconds per stage (retrieval stages plus ``llm``).
    """

    question: str
    answer: str
    citations: list[Citation]
    context_citations: list[Citation]
    relevant: bool
    retrieval: RetrievalResult
    timings_ms: dict[str, int] = field(default_factory=dict)


@dataclass
class AnswerEvent:
    """One streaming answer event.

    Attributes:
        type: ``token`` (a text piece), ``sources`` (serialized citations),
            ``done`` (the :class:`AnswerResult`), ``error`` (a user-safe
            message) or ``rewritten`` (a chat follow-up's standalone question,
            sent before the tokens).
        data: The event payload; a ``str`` for token/error, ``list[dict]`` for
            sources, an :class:`AnswerResult` for done.
    """

    type: Literal["token", "sources", "done", "error", "rewritten"]
    data: Any


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def citation_to_dict(citation: Citation) -> dict[str, Any]:
    """Serialize one citation for JSON consumers (CLI, API, MCP).

    Args:
        citation: The citation to serialize.

    Returns:
        A dict with ``n``, ``title``, ``path``, ``page``, ``slide``,
        ``section``, ``doc_date`` (ISO 8601 or None), ``is_current``,
        ``document_id`` and ``format``.
    """
    return {
        "n": citation.n,
        "title": citation.title,
        "path": citation.path,
        "page": citation.page,
        "slide": citation.slide,
        "section": citation.section,
        "doc_date": citation.doc_date.isoformat() if citation.doc_date is not None else None,
        "is_current": citation.is_current,
        "document_id": citation.document_id,
        "format": citation.format,
    }


def extract_cited(answer_text: str, citations: Sequence[Citation]) -> list[Citation]:
    """The context citations the answer text actually cites.

    ``[n]`` markers are matched against the given citations; numbers without a
    matching citation are ignored and repeats keep only the first mention, in
    first-mention order. An answer that cites nothing falls back to the first
    context citation so every grounded answer names at least one source.

    Args:
        answer_text: The raw answer text from the model.
        citations: The citations that were available in the context.

    Returns:
        The cited citations in first-mention order (never more than given).
    """
    by_number = {citation.n: citation for citation in citations}
    cited: list[Citation] = []
    seen: set[int] = set()
    for match in _CITATION_RE.finditer(answer_text):
        number = int(match.group(1))
        if number not in by_number or number in seen:
            continue
        seen.add(number)
        cited.append(by_number[number])
    if not cited and citations:
        return [citations[0]]
    return cited


def _clean_answer(text: str) -> str:
    """Strip stray HTML comments the model may emit from the answer text."""
    return _HTML_COMMENT_RE.sub("", text).strip()


def _not_found(result: RetrievalResult, question: str, timings: dict[str, int]) -> AnswerResult:
    """The gated answer for an irrelevant or empty retrieval (no LLM call)."""
    return AnswerResult(
        question=question,
        answer=NOT_FOUND[result.analysis.language or "en"],
        citations=[],
        context_citations=[],
        relevant=result.relevant,
        retrieval=result,
        timings_ms=timings,
    )


def answer(
    question: str, filters: SearchFilters | None = None, *, chat: ChatModel | None = None
) -> AnswerResult:
    """Generate one grounded answer for a question.

    The question is retrieved first; when the retrieval gate finds nothing
    relevant (or there are no hits at all) the language-appropriate
    "not found" text is returned without calling the model. Otherwise the
    context is assembled, the prompt built and the answer generated with
    ``priority="interactive"``.

    Args:
        question: The raw user question, possibly with inline filter tokens.
        filters: Base filters from CLI/API options; inline filters win.
        chat: The chat model to use; defaults to the configured answer client.

    Returns:
        The :class:`AnswerResult` for the question.

    Raises:
        EndpointError: When the chat endpoint fails; the streaming variant
            reports such failures as ``error`` events instead.
    """
    result = retrieve(question, filters)
    timings = dict(result.timings_ms)
    if not result.relevant or not result.hits:
        logger.info("answer %r: gated by retrieval, no LLM call", question)
        return _not_found(result, question, timings)
    bundle = assemble_context(result)
    messages = build_messages(
        question, bundle, result.analysis.language, as_of=result.analysis.as_of
    )
    model = chat if chat is not None else get_chat("answer")
    llm_started = time.perf_counter()
    completed = model.complete(messages, priority="interactive", max_tokens=_MAX_ANSWER_TOKENS)
    timings["llm"] = _elapsed_ms(llm_started)
    citations = extract_cited(completed.text, bundle.citations)
    logger.info(
        "answer %r: %d citation(s) in context, %d cited",
        question,
        len(bundle.citations),
        len(citations),
    )
    return AnswerResult(
        question=question,
        answer=_clean_answer(completed.text),
        citations=citations,
        context_citations=list(bundle.citations),
        relevant=result.relevant,
        retrieval=result,
        timings_ms=timings,
    )


def answer_stream(
    question: str,
    filters: SearchFilters | None = None,
    *,
    chat: ChatModel | None = None,
    search_question: str | None = None,
    history: Sequence[ChatMessage] = (),
) -> Iterator[AnswerEvent]:
    """Stream one grounded answer as :class:`AnswerEvent` pieces.

    Tokens are yielded as they arrive from the model, followed by one
    ``sources`` event with the serialized cited citations and a ``done`` event
    carrying the :class:`AnswerResult`. When generation fails, an ``error``
    event with a user-safe message is yielded instead, followed by ``done``;
    retrieval failures still propagate to the caller.

    Args:
        question: The raw user question, possibly with inline filter tokens.
        filters: Base filters from CLI/API options; inline filters win.
        chat: The chat model to use; defaults to the configured answer client.
        search_question: What to retrieve with, when it differs from the question
            (a chat follow-up rewritten as a standalone question).
        history: Earlier chat turns, placed between the system rules and the
            grounded prompt (:func:`mynotes.retrieval.followup.history_messages`).

    Yields:
        ``token`` events while the model streams, then ``sources`` and
        ``done`` (or ``error`` and ``done`` on failure).
    """
    result = retrieve(search_question or question, filters)
    timings = dict(result.timings_ms)
    if not result.relevant or not result.hits:
        logger.info("answer %r: gated by retrieval, no LLM call", question)
        gated = _not_found(result, question, timings)
        yield AnswerEvent(type="token", data=gated.answer)
        yield AnswerEvent(type="sources", data=[])
        yield AnswerEvent(type="done", data=gated)
        return
    bundle = assemble_context(result)
    messages = build_messages(
        question, bundle, result.analysis.language, as_of=result.analysis.as_of
    )
    if history:
        messages = [messages[0], *history, *messages[1:]]
    model = chat if chat is not None else get_chat("answer")
    pieces: list[str] = []
    llm_started = time.perf_counter()
    try:
        for piece in model.stream(messages, priority="interactive", max_tokens=_MAX_ANSWER_TOKENS):
            pieces.append(piece)
            yield AnswerEvent(type="token", data=piece)
    except Exception:
        logger.exception("answer %r: streaming failed", question)
        timings["llm"] = _elapsed_ms(llm_started)
        yield AnswerEvent(type="error", data=_STREAM_ERROR_MESSAGE)
        yield AnswerEvent(
            type="done",
            data=AnswerResult(
                question=question,
                answer=_STREAM_ERROR_MESSAGE,
                citations=[],
                context_citations=list(bundle.citations),
                relevant=result.relevant,
                retrieval=result,
                timings_ms=timings,
            ),
        )
        return
    timings["llm"] = _elapsed_ms(llm_started)
    raw = "".join(pieces)
    citations = extract_cited(raw, bundle.citations)
    logger.info(
        "answer %r: %d citation(s) in context, %d cited",
        question,
        len(bundle.citations),
        len(citations),
    )
    yield AnswerEvent(type="sources", data=[citation_to_dict(citation) for citation in citations])
    yield AnswerEvent(
        type="done",
        data=AnswerResult(
            question=question,
            answer=_clean_answer(raw),
            citations=citations,
            context_citations=list(bundle.citations),
            relevant=result.relevant,
            retrieval=result,
            timings_ms=timings,
        ),
    )
