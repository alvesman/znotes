"""LLM tag proposals and their review (PLAN.md §7.4 step 4, task 38).

The ``tag_llm`` pipeline job asks the extract LLM for up to five topic tags
per changed document version and stores them as ``document_tag(source='llm')``
rows: ``accepted`` only when the confidence reaches ``tags.llm_auto_accept``
and the tag already exists in the workspace vocabulary, ``proposed``
otherwise. The user settles the rest through :func:`pending_proposals` /
:func:`decide_proposal` / :func:`merge_proposal` behind ``mynotes tags
review|accept|reject``; rejected rows are kept so a tag is never re-proposed.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from mynotes.chunk.tokenizer import TokenCounter, get_token_counter
from mynotes.db.models import Document, DocumentTag, File, Tag
from mynotes.llm.chat import ChatMessage, ChatModel
from mynotes.llm.structured import complete_structured
from mynotes.tags.normalize import normalize_tag
from mynotes.tags.store import merge_tags, resolve_alias

logger = logging.getLogger(__name__)

_EXCERPT_TOKENS = 1500
_MAX_HEADINGS = 30
_VOCABULARY_SIZE = 200
_MINUTES_PER_DAY = 24 * 60

_WINDOW_RE = re.compile(r"(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})")

_SYSTEM_PROMPT = (
    "You assign topic tags to personal project notes. Prefer tags from the "
    "vocabulary when they fit. Propose at most 5 tags that describe what the "
    "document is about (topics, technologies, domains). Don't repeat the "
    "existing tags. Tags are short, lowercase, hyphenated. Reply only with "
    "JSON matching the schema."
)


class TagProposal(BaseModel):
    """One tag proposed by the model for a document version."""

    tag: str = Field(max_length=60)
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(max_length=200)


class TagProposals(BaseModel):
    """The structured answer: at most five proposals."""

    tags: list[TagProposal] = Field(max_length=5)


def cut_excerpt(
    text: str,
    *,
    max_tokens: int = _EXCERPT_TOKENS,
    counter: TokenCounter | None = None,
) -> str:
    """Return the first ``max_tokens`` tokens of ``text``, cut at a paragraph.

    Paragraphs are ``\\n\\n``-separated blocks (the IR plain-text layout). The
    first paragraph is always kept, so a single oversized paragraph is never
    emptied.

    Args:
        text: The plain text to cut.
        max_tokens: Approximate token budget for the excerpt.
        counter: Token counter; defaults to the process-wide one.

    Returns:
        The excerpt, ending at a paragraph boundary whenever one falls within
        the budget; the whole text when it already fits.
    """
    counter = counter if counter is not None else get_token_counter()
    if counter.count(text) <= max_tokens:
        return text
    kept: list[str] = []
    total = 0
    for paragraph in text.split("\n\n"):
        count = counter.count(paragraph)
        if kept and total + count > max_tokens:
            break
        kept.append(paragraph)
        total += count
    return "\n\n".join(kept)


def build_tag_messages(
    *,
    title: str | None,
    rel_path: str,
    headings: Sequence[str],
    excerpt: str,
    existing_tags: Sequence[str],
    vocabulary: Sequence[str],
    language: str | None,
) -> list[ChatMessage]:
    """Build the system and user messages for one auto-tagging call.

    Args:
        title: The resolved document title, or None.
        rel_path: The file path relative to its source root.
        headings: The document's headings; only the first 30 are sent.
        excerpt: The plain-text excerpt (already cut to the token budget).
        existing_tags: Tags the document already has; the model must not
            repeat them.
        vocabulary: Existing tag names the model should prefer when they fit.
        language: Detected document language, or None.

    Returns:
        One system message with the tagging rules and one user message with
        the document context.
    """
    lines = [f"Title: {title or '(untitled)'}", f"Path: {rel_path}"]
    if language:
        lines.append(f"Language: {language}")
    if headings:
        lines.append("Headings: " + "; ".join(list(headings)[:_MAX_HEADINGS]))
    lines.append("Excerpt:\n" + (excerpt or "(no text)"))
    lines.append("Existing tags: " + (", ".join(existing_tags) if existing_tags else "(none)"))
    lines.append("Vocabulary: " + (", ".join(vocabulary) if vocabulary else "(none)"))
    lines.append(
        'Reply with JSON of the shape {"tags": '
        '[{"tag": ..., "confidence": 0.0-1.0, "reason": ...}]}'
    )
    return [
        ChatMessage(role="system", content=_SYSTEM_PROMPT),
        ChatMessage(role="user", content="\n\n".join(lines)),
    ]


def propose_tags(chat: ChatModel, **kwargs: Any) -> list[TagProposal]:
    """Ask the chat model for tag proposals and validate its JSON answer.

    Args:
        chat: The chat model to call (normally ``get_chat("extract")``).
        **kwargs: The keyword arguments of :func:`build_tag_messages`.

    Returns:
        The validated proposals (at most five), in the model's order.
    """
    proposals = complete_structured(
        chat,
        build_tag_messages(**kwargs),
        TagProposals,
        priority="background",
        max_tokens=800,
    )
    return list(proposals.tags)


def tag_vocabulary(session: Session, workspace_id: int, limit: int = _VOCABULARY_SIZE) -> list[str]:
    """Return the top accepted tags by document count for the prompt.

    Args:
        session: An open session/transaction.
        workspace_id: The workspace whose vocabulary to read.
        limit: Maximum number of tag names.

    Returns:
        Up to ``limit`` tag names, most-used first (ties by name), excluding
        ``folder:*`` and ``status:*`` tags.
    """
    documents = func.count(func.distinct(DocumentTag.document_id))
    rows = session.execute(
        select(Tag.name, documents)
        .join(DocumentTag, DocumentTag.tag_id == Tag.id)
        .where(
            Tag.workspace_id == workspace_id,
            DocumentTag.status == "accepted",
            Tag.name.notlike("folder:%"),
            Tag.name.notlike("status:%"),
        )
        .group_by(Tag.id, Tag.name)
        .order_by(documents.desc(), Tag.name)
        .limit(limit)
    ).all()
    return [str(name) for name, _count in rows]


def seconds_until_window_opens(window: str, now: datetime) -> float:
    """Seconds until ``now`` enters a 'HH:MM-HH:MM' local-time window.

    Args:
        window: The window (same format as ``limits.background_window``).
        now: The current local time.

    Returns:
        0.0 when ``now`` is already inside (or the window covers all day),
        otherwise the delay in seconds until the window opens.

    Raises:
        ValueError: If the window format is invalid.
    """
    match = _WINDOW_RE.fullmatch(window.strip())
    if match is None:
        raise ValueError(f"bad window format: {window!r} (expected 'HH:MM-HH:MM')")
    start_h, start_m, end_h, end_m = (int(group) for group in match.groups())
    if start_h > 23 or end_h > 23 or start_m > 59 or end_m > 59:
        raise ValueError(f"bad window time: {window!r}")
    start = start_h * 60 + start_m
    end = end_h * 60 + end_m
    if start == end:
        return 0.0
    minute = now.hour * 60 + now.minute
    if start < end:
        inside = start <= minute < end
        target = start if minute < start else start + _MINUTES_PER_DAY
    else:
        inside = minute >= start or minute < end
        target = start
    if inside:
        return 0.0
    return (target - minute) * 60 - now.second - now.microsecond / 1e6


@dataclass(frozen=True)
class ProposalView:
    """One proposed LLM tag row, for the review CLI."""

    document_id: int
    tag_id: int
    name: str
    confidence: float
    reason: str | None


@dataclass(frozen=True)
class DocumentProposals:
    """A document with its pending LLM tag proposals."""

    document_id: int
    title: str | None
    rel_path: str
    proposals: list[ProposalView]


def pending_proposals(session: Session, limit: int = 100) -> list[DocumentProposals]:
    """Group the proposed LLM tags by document, oldest proposal first.

    Args:
        session: An open session/transaction.
        limit: Maximum number of documents to return.

    Returns:
        One entry per document that still has ``source='llm'`` rows with
        ``status='proposed'``, ordered by the oldest proposal.
    """
    rows = session.execute(
        select(
            DocumentTag.document_id,
            DocumentTag.tag_id,
            Tag.name,
            DocumentTag.confidence,
            DocumentTag.reason,
            Document.title,
            File.rel_path,
        )
        .join(Tag, Tag.id == DocumentTag.tag_id)
        .join(Document, Document.id == DocumentTag.document_id)
        .outerjoin(File, File.id == Document.current_file_id)
        .where(DocumentTag.source == "llm", DocumentTag.status == "proposed")
        .order_by(DocumentTag.created_at, DocumentTag.document_id, Tag.name)
    ).all()
    grouped: dict[int, DocumentProposals] = {}
    for document_id, tag_id, name, confidence, reason, title, rel_path in rows:
        entry = grouped.get(document_id)
        if entry is None:
            entry = DocumentProposals(
                document_id=document_id, title=title, rel_path=rel_path or "", proposals=[]
            )
            grouped[document_id] = entry
        entry.proposals.append(
            ProposalView(
                document_id=document_id,
                tag_id=tag_id,
                name=str(name),
                confidence=float(confidence),
                reason=reason,
            )
        )
    return list(grouped.values())[:limit]


def decide_proposal(session: Session, document_id: int, tag_id: int, decision: str) -> bool:
    """Accept or reject one proposed LLM tag of a document.

    Args:
        session: An open session/transaction; the caller commits.
        document_id: The document the proposal belongs to.
        tag_id: The proposed tag's id.
        decision: ``accepted`` or ``rejected``; rejected rows are kept so the
            tag is not proposed again for the document.

    Returns:
        True when a proposed row changed status.

    Raises:
        ValueError: If ``decision`` is not ``accepted`` or ``rejected``.
    """
    if decision not in ("accepted", "rejected"):
        raise ValueError(f"bad decision: {decision!r} (expected 'accepted' or 'rejected')")
    result = session.execute(
        update(DocumentTag)
        .where(
            DocumentTag.document_id == document_id,
            DocumentTag.tag_id == tag_id,
            DocumentTag.source == "llm",
            DocumentTag.status == "proposed",
        )
        .values(status=decision)
    )
    return bool(result.rowcount)


def merge_proposal(
    session: Session, workspace_id: int, document_id: int, from_name: str, into_name: str
) -> bool:
    """Merge a proposed LLM tag into another tag and accept the merged row.

    Args:
        session: An open session/transaction; the caller commits.
        workspace_id: The workspace owning the vocabulary.
        document_id: The document whose proposal is being merged.
        from_name: The proposed tag's name (normalized, alias-resolved inside).
        into_name: The target tag; created when new, with an alias from
            ``from_name``.

    Returns:
        True when the document's merged row was set to ``accepted``.
    """
    moved = merge_tags(session, workspace_id, from_name, into_name)
    logger.debug(
        "merge_proposal %r -> %r for document %s: %d row(s) moved",
        from_name,
        into_name,
        document_id,
        moved,
    )
    resolved = resolve_alias(session, workspace_id, into_name)
    tag_id = session.execute(
        select(Tag.id).where(Tag.workspace_id == workspace_id, Tag.name == resolved)
    ).scalar_one_or_none()
    if tag_id is None:
        return False
    return decide_proposal(session, document_id, tag_id, "accepted")


def tag_id_by_name(session: Session, workspace_id: int, name: str) -> int | None:
    """Return the id of a (possibly aliased) tag name in the vocabulary.

    Args:
        session: An open session/transaction.
        workspace_id: The workspace owning the vocabulary.
        name: A tag name in any spelling.

    Returns:
        The tag id, or None when the normalized name is empty or unknown.
    """
    resolved = resolve_alias(session, workspace_id, normalize_tag(name))
    if not resolved:
        return None
    return session.execute(
        select(Tag.id).where(Tag.workspace_id == workspace_id, Tag.name == resolved)
    ).scalar_one_or_none()
