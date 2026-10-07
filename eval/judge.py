"""LLM-as-judge scoring for answer-mode eval runs.

The judge is secondary evidence: retrieval metrics stay primary and any judge
failure is recorded as ``{"judge_error": ...}``, never fatal. The call goes
through :func:`mynotes.llm.structured.complete_structured` at ``background``
priority, so it shares the limits/retry layer with the rest of the system.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import or_, select

from mynotes.db.models import Chunk, DocumentVersion
from mynotes.db.session import session_scope
from mynotes.llm.chat import ChatMessage, ChatModel
from mynotes.llm.structured import complete_structured

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are grading a RAG system's answer.

Question ({language}):
{question}

Reference material (ground truth — the answer must agree with this):
{reference}

Candidate answer:
{answer}

Grade the candidate answer against the reference material only. Reply with a
single JSON object, no other text:
{{"correctness": 0-2, "groundedness": 0-2, "completeness": 0-2,
  "verdict": "pass" | "partial" | "fail", "reason": "<one sentence>"}}

correctness: 2 = factually agrees with reference, 1 = partially, 0 = wrong.
groundedness: 2 = nothing contradicts the reference and no values are invented,
1 = some claims contradict the reference or invent facts, 0 = mostly invented or
answered from outside knowledge. The reference may be a short answer; extra
context in the candidate (names, the document it came from, how it got there)
is NOT a fault as long as it does not contradict the reference.
completeness: 2 = covers what the question asks, 1 = partial, 0 = missing.
"""

_REFERENCE_WORD_CAP = 1200
"""The judge prompt stays sane: reference material capped at 1,200 words."""

_NO_REFERENCE = "(no reference material available)"

_NO_ANSWER_REFERENCE = (
    "The notes do not contain this information. The correct answer says so (declines) and "
    "invents nothing; an answer that supplies a value is wrong."
)
"""Reference for ``category: no-answer`` questions, so a correct decline passes."""


class JudgeVerdict(BaseModel):
    """One parsed judge verdict (the judge's JSON contract)."""

    correctness: int = Field(ge=0, le=2)
    groundedness: int = Field(ge=0, le=2)
    completeness: int = Field(ge=0, le=2)
    verdict: Literal["pass", "partial", "fail"]
    reason: str


def build_reference(question: dict[str, Any]) -> str:
    """Build the judge reference for one question.

    The explicit ``reference_answer`` wins; otherwise the text of the expected
    documents/pages is pulled from Postgres (latest versions only), text-
    deduplicated and capped at 1,200 words.

    Args:
        question: The golden-set question.

    Returns:
        The reference material, or a placeholder when nothing is available.
    """
    if question.get("reference_answer"):
        return str(question["reference_answer"])
    if question.get("category") == "no-answer":
        return _NO_ANSWER_REFERENCE
    doc_paths = list(question.get("expected_docs") or [])
    if not doc_paths:
        return _NO_REFERENCE
    pages = list(question.get("expected_pages") or [])
    parts: list[str] = []
    seen: set[str] = set()
    with session_scope() as session:
        statement = (
            select(Chunk.text)
            .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
            .where(
                DocumentVersion.rel_path_at_version.in_(doc_paths),
                DocumentVersion.is_latest.is_(True),
            )
            .order_by(DocumentVersion.id, Chunk.ordinal)
        )
        if pages:
            statement = statement.where(or_(Chunk.page.in_(pages), Chunk.slide.in_(pages)))
        for (text,) in session.execute(statement):
            clean = (text or "").strip()
            if clean and clean not in seen:
                seen.add(clean)
                parts.append(clean)
    reference = "\n".join(parts)
    words = reference.split()
    if len(words) > _REFERENCE_WORD_CAP:
        reference = " ".join(words[:_REFERENCE_WORD_CAP])
    return reference or _NO_REFERENCE


def judge_answer(
    chat: ChatModel,
    question: dict[str, Any],
    answer_text: str | None,
    reference: str,
) -> dict[str, Any]:
    """Judge one answer against the reference material.

    Args:
        chat: The chat model used for the judge call.
        question: The golden-set question.
        answer_text: The produced answer text (may be empty).
        reference: The ground-truth material from :func:`build_reference`.

    Returns:
        The validated verdict as a dict, or ``{"judge_error": ...}`` on any
        failure (empty answer, unparseable JSON, endpoint error).
    """
    prompt = JUDGE_PROMPT.format(
        language=question.get("language", "pt"),
        question=question["question"],
        reference=reference,
        answer=answer_text or "(no answer produced)",
    )
    try:
        verdict = complete_structured(
            chat, [ChatMessage(role="user", content=prompt)], JudgeVerdict, priority="background"
        )
    except Exception as exc:
        logger.error("judge call failed: %s", exc)
        return {"judge_error": str(exc)}
    return verdict.model_dump()
