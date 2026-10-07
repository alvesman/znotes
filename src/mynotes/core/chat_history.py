"""Chat history: every streamed chat answer, kept for the History page.

Entries are scoped to the caller's ``Principal.user`` (``"local"`` in local
mode), so in team mode people only see and delete their own chats. The
questions asked on one chat page share a ``conversation_id`` and are shown and
deleted together; entries without one (written before migration 0007) are
one-question conversations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Text, cast, delete, func, literal, select
from sqlalchemy.sql import ColumnElement

from mynotes.core.principal import LOCAL, Principal
from mynotes.db.models import ChatEntry
from mynotes.db.session import session_scope

MAX_LISTED = 200
"""Conversations the History page shows (most recently active first)."""


@dataclass(frozen=True)
class ChatRecord:
    """One stored question and its answer."""

    id: int
    question: str
    answer: str
    citations: list[dict[str, Any]]
    filters: dict[str, Any]
    relevant: bool
    created_at: datetime
    standalone_question: str | None = None


@dataclass(frozen=True)
class Conversation:
    """The turns of one chat, oldest first."""

    key: str
    turns: list[ChatRecord] = field(default_factory=list)

    @property
    def started_at(self) -> datetime:
        return self.turns[0].created_at

    @property
    def last_at(self) -> datetime:
        return self.turns[-1].created_at


def _group_key() -> ColumnElement[str]:
    """The conversation id, or a per-entry key for entries that have none."""
    return func.coalesce(
        ChatEntry.conversation_id,
        literal("entry-") + cast(ChatEntry.id, Text),
    )


def record_chat(
    question: str,
    answer: str,
    citations: list[dict[str, Any]],
    *,
    filters: dict[str, Any] | None = None,
    relevant: bool = True,
    conversation_id: str | None = None,
    standalone_question: str | None = None,
    principal: Principal = LOCAL,
) -> int:
    """Store one finished chat answer and return its id."""
    with session_scope() as session:
        entry = ChatEntry(
            user_name=principal.user,
            conversation_id=conversation_id or None,
            question=question,
            standalone_question=standalone_question
            if standalone_question and standalone_question != question
            else None,
            answer=answer,
            citations=list(citations),
            filters=dict(filters or {}),
            relevant=relevant,
        )
        session.add(entry)
        session.flush()
        return int(entry.id)


def list_conversations(
    *, principal: Principal = LOCAL, limit: int = MAX_LISTED
) -> tuple[list[Conversation], int]:
    """The caller's conversations, most recently active first, and how many there are."""
    key = _group_key().label("key")
    with session_scope() as session:
        grouped = (
            select(key, func.max(ChatEntry.created_at).label("last_at"))
            .where(ChatEntry.user_name == principal.user)
            .group_by(key)
            .subquery()
        )
        total = int(session.scalar(select(func.count()).select_from(grouped)) or 0)
        keys = [
            str(row.key)
            for row in session.execute(
                select(grouped.c.key).order_by(grouped.c.last_at.desc()).limit(limit)
            ).all()
        ]
        if not keys:
            return [], total
        rows = session.execute(
            select(ChatEntry, _group_key().label("key"))
            .where(ChatEntry.user_name == principal.user, _group_key().in_(keys))
            .order_by(ChatEntry.created_at, ChatEntry.id)
        ).all()
        turns: dict[str, list[ChatRecord]] = {k: [] for k in keys}
        for entry, entry_key in rows:
            turns[str(entry_key)].append(
                ChatRecord(
                    id=int(entry.id),
                    question=entry.question,
                    answer=entry.answer,
                    citations=list(entry.citations or []),
                    filters=dict(entry.filters or {}),
                    relevant=bool(entry.relevant),
                    created_at=entry.created_at,
                    standalone_question=entry.standalone_question,
                )
            )
    return [Conversation(key=k, turns=turns[k]) for k in keys if turns[k]], total


def delete_chats(
    *, older_than_days: int | None = None, principal: Principal = LOCAL, now: datetime | None = None
) -> int:
    """Delete the caller's conversations idle for ``older_than_days`` days, or all of them.

    A conversation is deleted as a whole, by its *last* question: one that
    continued yesterday is kept even if it started a month ago.

    Args:
        older_than_days: Keep conversations active in the last N days; None deletes all.
        principal: The caller; only its own chats are deleted.
        now: The reference time (tests).

    Returns:
        The number of conversations deleted.
    """
    with session_scope() as session:
        if older_than_days is None:
            count = int(
                session.scalar(
                    select(func.count(func.distinct(_group_key()))).where(
                        ChatEntry.user_name == principal.user
                    )
                )
                or 0
            )
            session.execute(delete(ChatEntry).where(ChatEntry.user_name == principal.user))
            return count
        cutoff = (now or datetime.now(UTC)) - timedelta(days=older_than_days)
        key = _group_key().label("key")
        stale = [
            str(row.key)
            for row in session.execute(
                select(key)
                .where(ChatEntry.user_name == principal.user)
                .group_by(key)
                .having(func.max(ChatEntry.created_at) < cutoff)
            ).all()
        ]
        if stale:
            session.execute(
                delete(ChatEntry).where(
                    ChatEntry.user_name == principal.user, _group_key().in_(stale)
                )
            )
        return len(stale)


def get_conversation(key: str, *, principal: Principal = LOCAL) -> Conversation | None:
    """One of the caller's conversations by its key (the id, or ``entry-<id>``), or None."""
    with session_scope() as session:
        rows = session.execute(
            select(ChatEntry)
            .where(ChatEntry.user_name == principal.user, _group_key() == key)
            .order_by(ChatEntry.created_at, ChatEntry.id)
        ).scalars()
        turns = [
            ChatRecord(
                id=int(entry.id),
                question=entry.question,
                answer=entry.answer,
                citations=list(entry.citations or []),
                filters=dict(entry.filters or {}),
                relevant=bool(entry.relevant),
                created_at=entry.created_at,
                standalone_question=entry.standalone_question,
            )
            for entry in rows
        ]
    return Conversation(key=key, turns=turns) if turns else None
