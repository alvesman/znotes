"""Ask endpoints: JSON answers and SSE streaming (PLAN.md §10.2).

The stream endpoint is a plain ``def`` returning a ``StreamingResponse`` over
a sync generator, so the blocking answer generation never runs on the event
loop (the anti-pattern of an async function draining a sync generator is
deliberately avoided).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from mynotes.api import schemas
from mynotes.api.auth import PrincipalDep
from mynotes.core import service
from mynotes.core.principal import Principal
from mynotes.retrieval.answer import AnswerEvent, citation_to_dict

logger = logging.getLogger(__name__)

router = APIRouter()

_STREAM_ERROR_MESSAGE = "The request failed. Please try again."


def _sse_frame(event: str, data: Any) -> str:
    """Format one SSE frame (``event`` line, JSON ``data`` line, blank line)."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _event_payload(event: AnswerEvent) -> Any:
    """The JSON payload for one answer event."""
    if event.type == "token":
        return {"text": event.data}
    if event.type == "sources":
        return {"citations": event.data}
    if event.type == "done":
        return {"relevant": event.data.relevant, "timings_ms": event.data.timings_ms}
    if event.type == "rewritten":
        return {"question": event.data}
    return {"message": event.data}


def _event_stream(
    question: str,
    filters: schemas.FiltersModel | None,
    principal: Principal,
    *,
    save: bool = True,
    conversation_id: str | None = None,
) -> Iterator[str]:
    """Yield SSE frames for one streamed answer, ending with ``done``.

    A completed answer is stored in the chat history (``save``); a failed one is not.
    """
    try:
        base_filters = filters.to_search_filters() if filters is not None else None
        answer: list[str] = []
        citations: list[dict[str, Any]] = []
        standalone: str | None = None
        for event in service.ask_stream(
            question, base_filters, principal=principal, conversation_id=conversation_id
        ):
            if event.type == "rewritten":
                standalone = str(event.data)
            elif event.type == "token":
                answer.append(str(event.data))
            elif event.type == "sources":
                citations = list(event.data or [])
            elif event.type == "done" and save:
                _save_chat(
                    question,
                    "".join(answer),
                    citations,
                    filters,
                    event,
                    principal,
                    conversation_id,
                    standalone,
                )
            payload = _event_payload(event)
            if event.type == "sources" and principal.can_open_files:
                payload = {"citations": _mark_openable(citations)}
            yield _sse_frame(event.type, payload)
    except Exception:
        logger.exception("ask stream for %r failed", question)
        yield _sse_frame("error", {"message": _STREAM_ERROR_MESSAGE})
        yield _sse_frame("done", {"relevant": False, "timings_ms": {}})


def _mark_openable(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The citations, with ``openable: false`` on notes uploaded by an agent (no "Open in app")."""
    remote = service.remote_document_ids(
        citation["document_id"]
        for citation in citations
        if citation.get("format") in service.OPEN_IN_APP_FORMATS and "document_id" in citation
    )
    return [
        {**citation, "openable": False} if citation.get("document_id") in remote else citation
        for citation in citations
    ]


def _save_chat(
    question: str,
    answer: str,
    citations: list[dict[str, Any]],
    filters: schemas.FiltersModel | None,
    event: AnswerEvent,
    principal: Principal,
    conversation_id: str | None = None,
    standalone: str | None = None,
) -> None:
    """Store a finished answer; a history failure never breaks the answer."""
    from mynotes.core.chat_history import record_chat

    try:
        record_chat(
            question,
            answer,
            citations,
            filters=filters.model_dump(mode="json", exclude_none=True) if filters else {},
            relevant=bool(getattr(event.data, "relevant", True)),
            conversation_id=conversation_id,
            standalone_question=standalone,
            principal=principal,
        )
    except Exception:
        logger.warning("chat history not saved", exc_info=True)


@router.post("/v1/ask")
def ask(payload: schemas.AskRequest, principal: PrincipalDep) -> schemas.AskResponse:
    """Generate one grounded, cited answer for a question."""
    base_filters = payload.filters.to_search_filters() if payload.filters is not None else None
    result = service.ask(payload.question, base_filters, principal=principal)
    return schemas.AskResponse(
        answer=result.answer,
        citations=[
            schemas.CitationModel(**citation_to_dict(citation)) for citation in result.citations
        ],
        relevant=result.relevant,
        timings_ms=result.timings_ms,
    )


@router.post("/v1/ask/stream")
def ask_stream(payload: schemas.AskRequest, principal: PrincipalDep) -> StreamingResponse:
    """Stream one answer as ``token``, ``sources``, ``done`` SSE events.

    A chat follow-up (``conversation_id`` with earlier turns) is first sent as a
    ``rewritten`` event carrying the standalone question it was searched with.
    """
    return StreamingResponse(
        _event_stream(
            payload.question,
            payload.filters,
            principal,
            save=payload.save_history,
            conversation_id=payload.conversation_id,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
