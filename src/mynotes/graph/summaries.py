"""Entity summaries ("entity cards") in a debounced background job (task 42).

Entity summaries are marked stale whenever resolution touches an entity
(task 41); the ``refresh_summaries`` job regenerates them in batches with a
10-minute enqueue delay, so bursts of changes coalesce into one rewrite. An
entity only gets a summary once at least two current documents mention it —
for a single mention the extracted description is already the card. The prompt
shows the entity's name, type, aliases, its top active relations and mention
excerpts, and asks for 2-3 factual sentences in the dominant language of the
evidence. Saved summaries become the entity documents' description in the
entities index (outbox upsert).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, aliased

from mynotes.db.models import (
    Chunk,
    Document,
    DocumentVersion,
    KgEntity,
    KgEntityAlias,
    KgMention,
    KgRelation,
)
from mynotes.index.outbox import enqueue_entity_upserts
from mynotes.jobs.queue import enqueue
from mynotes.jobs.registry import Priority
from mynotes.llm.chat import ChatMessage, ChatModel
from mynotes.scan.hashing import hash_text

logger = logging.getLogger(__name__)

_MAX_RELATIONS = 15
"""Active relations shown in the prompt, heaviest first."""

_MAX_MENTIONS = 5
"""Mention excerpts from current documents shown in the prompt."""

_EXCERPT_RADIUS = 120
"""Characters of chunk context kept around each mention."""

_MAX_TOKENS = 300
"""Completion budget for one summary."""

_DEBOUNCE_MINUTES = 10
"""Enqueue delay so bursts of changes rewrite the summary once."""

_LANGUAGE_NAMES = {"pt": "Portuguese", "en": "English"}

_SYSTEM_PROMPT = (
    "You write short summary cards for entities found in personal notes. "
    "Rules: 2-3 sentences, factual, only what the notes say, no source "
    "citations, no markdown. Reply with the summary text only."
)


def enqueue_refresh_summaries(session: Session, entity_ids: Sequence[int]) -> int | None:
    """Queue one debounced ``refresh_summaries`` job for a batch of entities.

    Args:
        session: An open session/transaction; the job row is written but not
            committed.
        entity_ids: The entity ids whose summaries went stale.

    Returns:
        The job id, or None when the batch is empty or deduplicated (an active
        job with the same batch key already exists).
    """
    ids = sorted({int(entity_id) for entity_id in entity_ids})
    if not ids:
        return None
    fingerprint = hash_text(json.dumps(ids))
    return enqueue(
        session,
        "refresh_summaries",
        {"entity_ids": ids},
        dedupe_key=f"refresh_summaries:{fingerprint}",
        priority=Priority.SUMMARIES,
        run_after=datetime.now(UTC) + timedelta(minutes=_DEBOUNCE_MINUTES),
    )


def refresh_entity_summaries(session: Session, entity_ids: Sequence[int], chat: ChatModel) -> int:
    """Regenerate the stale summaries of a batch of entities.

    Entities that are missing, merged, not stale or single-mention are skipped
    without an LLM call. A model failure raises, so the job retries and
    already-rewritten entities are skipped on the re-delivery.

    Args:
        session: An open session/transaction; rows are written but not
            committed.
        entity_ids: The entity ids from the job payload.
        chat: The chat model (normally ``get_chat("extract")``).

    Returns:
        How many summaries were rewritten.
    """
    rewritten = 0
    for entity_id in entity_ids:
        if refresh_entity_summary(session, int(entity_id), chat):
            rewritten += 1
    return rewritten


def refresh_entity_summary(session: Session, entity_id: int, chat: ChatModel) -> bool:
    """Regenerate one entity's summary.

    Args:
        session: An open session/transaction; rows are written but not
            committed.
        entity_id: The entity to summarize.
        chat: The chat model.

    Returns:
        True when a summary was written (and the entity outbox got an upsert).
    """
    entity = session.get(KgEntity, entity_id)
    if entity is None or entity.status != "active":
        return False
    if not entity.summary_stale or entity.mention_count_current < 2:
        return False
    result = chat.complete(
        build_summary_messages(session, entity), priority="background", max_tokens=_MAX_TOKENS
    )
    summary = result.text.strip()
    if not summary:
        logger.warning("refresh_summaries of entity %d: empty answer; keeping it stale", entity_id)
        return False
    entity.summary = summary
    entity.summary_stale = False
    entity.updated_at = datetime.now(UTC)
    enqueue_entity_upserts(session, [entity_id])
    return True


def build_summary_messages(session: Session, entity: KgEntity) -> list[ChatMessage]:
    """Build the system and user messages for one entity summary call.

    The user message carries the entity's name, type, aliases, up to
    :data:`_MAX_RELATIONS` active relations (as ``src -type→ dst
    [attribute=value] (date)``) and up to :data:`_MAX_MENTIONS` excerpts from
    current documents, and asks for the summary in the dominant language of
    the current mention documents (omitted when it is unknown).
    """
    aliases = list(
        session.execute(
            select(KgEntityAlias.alias)
            .where(KgEntityAlias.entity_id == entity.id)
            .order_by(KgEntityAlias.alias)
        ).scalars()
    )
    lines = [
        f"Entity: {entity.canonical_name} ({entity.type})",
    ]
    if entity.description:
        lines.append(f"Description: {entity.description}")
    if aliases:
        lines.append("Aliases: " + ", ".join(aliases))
    relations = _relation_lines(session, entity)
    if relations:
        lines.append("\n".join(["Relations:", *relations]))
    excerpts = _mention_excerpts(session, int(entity.id))
    if excerpts:
        lines.append("\n".join(["Notes mentioning it:", *(f'- "{text}"' for text in excerpts)]))
    language = _dominant_language(session, int(entity.id))
    request = "Write 2-3 sentences, factual, citing no sources"
    if language is not None:
        request += f", in {_LANGUAGE_NAMES.get(language, language)}"
    request += ", summarizing what this entity is and what the notes say about it."
    user = "\n\n".join([*lines, request])
    return [
        ChatMessage(role="system", content=_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user),
    ]


def _relation_lines(session: Session, entity: KgEntity) -> list[str]:
    """Render the entity's top active relations as ``src -type→ dst …`` lines."""
    source = aliased(KgEntity)
    target = aliased(KgEntity)
    rows = session.execute(
        select(KgRelation, source.canonical_name, target.canonical_name)
        .join(source, source.id == KgRelation.src_entity_id)
        .join(target, target.id == KgRelation.dst_entity_id)
        .where(
            or_(
                KgRelation.src_entity_id == entity.id,
                KgRelation.dst_entity_id == entity.id,
            ),
            KgRelation.status == "active",
        )
        .order_by(KgRelation.weight.desc(), KgRelation.id)
        .limit(_MAX_RELATIONS)
    ).all()
    lines = []
    for relation, src_name, dst_name in rows:
        line = f"{src_name} –{relation.type}→ {dst_name}"  # noqa: RUF001
        if relation.attribute is not None:
            line += f" [{relation.attribute}={relation.value}]"
        if relation.valid_from is not None:
            line += f" ({relation.valid_from.date().isoformat()})"
        lines.append(f"- {line}")
    return lines


def _mention_excerpts(session: Session, entity_id: int) -> list[str]:
    """Up to five excerpts around the entity's mentions in current documents."""
    rows = session.execute(
        select(Chunk.text, KgMention.surface_form)
        .join(Chunk, Chunk.id == KgMention.chunk_id)
        .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
        .join(Document, Document.id == DocumentVersion.document_id)
        .where(
            KgMention.entity_id == entity_id,
            DocumentVersion.is_latest.is_(True),
            Document.status == "current",
        )
        .order_by(Document.doc_date.desc().nulls_last(), Document.id, Chunk.ordinal)
        .limit(_MAX_MENTIONS)
    ).all()
    excerpts: list[str] = []
    for text, surface in rows:
        position = text.casefold().find(surface.casefold())
        if position >= 0:
            start = max(0, position - _EXCERPT_RADIUS)
            end = min(len(text), position + len(surface) + _EXCERPT_RADIUS)
            excerpts.append(text[start:end])
        else:
            excerpts.append(text[: 2 * _EXCERPT_RADIUS])
    return excerpts


def _dominant_language(session: Session, entity_id: int) -> str | None:
    """The most common document language among current mention documents."""
    row = session.execute(
        select(Document.language, func.count())
        .select_from(KgMention)
        .join(Chunk, Chunk.id == KgMention.chunk_id)
        .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
        .join(Document, Document.id == DocumentVersion.document_id)
        .where(
            KgMention.entity_id == entity_id,
            DocumentVersion.is_latest.is_(True),
            Document.status == "current",
            Document.language.is_not(None),
        )
        .group_by(Document.language)
        .order_by(func.count().desc(), Document.language)
        .limit(1)
    ).first()
    return str(row[0]) if row is not None else None
