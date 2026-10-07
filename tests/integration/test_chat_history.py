"""Chat history storage, conversations, per-user scoping and cleanup (migrations 0006-0007)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete, update

from mynotes.config import Settings
from mynotes.core import chat_history
from mynotes.core.principal import Principal
from mynotes.db.models import ChatEntry
from mynotes.db.session import reset_engine, session_scope

pytestmark = pytest.mark.integration

ANA = Principal(user="ana")
RUI = Principal(user="rui")


@pytest.fixture
def env(migrated_dsn: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    settings = Settings(postgres={"dsn": migrated_dsn})
    for target in ("mynotes.config.get_settings", "mynotes.db.session.get_settings"):
        monkeypatch.setattr(target, lambda: settings)
    reset_engine()
    yield
    with session_scope() as session:
        session.execute(delete(ChatEntry))
    reset_engine()


def _age(entry_id: int, days: int) -> None:
    with session_scope() as session:
        session.execute(
            update(ChatEntry)
            .where(ChatEntry.id == entry_id)
            .values(created_at=datetime.now(UTC) - timedelta(days=days))
        )


def test_turns_group_into_conversations_and_cleanup_goes_by_last_use(env: None) -> None:
    # a conversation that started 40 days ago and continued yesterday
    long_first = chat_history.record_chat("start", "a", [], conversation_id="c-long", principal=ANA)
    long_next = chat_history.record_chat(
        "follow-up", "b", [], conversation_id="c-long", principal=ANA
    )
    # an idle conversation, and an old entry without a conversation id
    idle = chat_history.record_chat(
        "idle",
        "c",
        [{"n": 1, "path": "a.md"}],
        filters={"history": True},
        conversation_id="c-idle",
        principal=ANA,
    )
    legacy = chat_history.record_chat("legacy", "d", [], principal=ANA)
    other = chat_history.record_chat("rui's", "e", [], conversation_id="c-long", principal=RUI)
    _age(long_first, 40)
    _age(long_next, 1)
    _age(idle, 40)
    _age(legacy, 50)
    _age(other, 40)

    conversations, total = chat_history.list_conversations(principal=ANA)
    assert total == 3
    assert conversations[0].key == "c-long"  # most recently active first
    assert [t.question for t in conversations[0].turns] == ["start", "follow-up"]
    idle_conversation = next(c for c in conversations if c.key == "c-idle")
    assert idle_conversation.turns[0].citations == [{"n": 1, "path": "a.md"}]

    assert chat_history.delete_chats(older_than_days=30, principal=ANA) == 2  # idle + legacy
    remaining, _ = chat_history.list_conversations(principal=ANA)
    assert [c.key for c in remaining] == ["c-long"] and len(remaining[0].turns) == 2
    assert chat_history.delete_chats(principal=ANA) == 1
    assert chat_history.list_conversations(principal=ANA) == ([], 0)
    # Rui's chat, even with the same conversation id, is untouched by Ana's cleanup
    assert [t.id for t in chat_history.list_conversations(principal=RUI)[0][0].turns] == [other]


def test_get_conversation_including_entries_without_an_id(env: None) -> None:
    first = chat_history.record_chat("q1", "a1", [], conversation_id="c1", principal=ANA)
    second = chat_history.record_chat("q2", "a2", [], conversation_id="c1", principal=ANA)
    legacy = chat_history.record_chat("old", "a", [], principal=ANA)
    found = chat_history.get_conversation("c1", principal=ANA)
    assert found is not None and [t.id for t in found.turns] == [first, second]
    assert chat_history.get_conversation("c1", principal=RUI) is None
    # an entry saved without an id is continued under its "entry-<id>" key
    key = f"entry-{legacy}"
    assert [t.id for t in chat_history.get_conversation(key, principal=ANA).turns] == [legacy]
    follow = chat_history.record_chat("more", "b", [], conversation_id=key, principal=ANA)
    assert [t.id for t in chat_history.get_conversation(key, principal=ANA).turns] == [
        legacy,
        follow,
    ]
    conversations, _ = chat_history.list_conversations(principal=ANA)
    assert sorted(len(c.turns) for c in conversations) == [2, 2]


def test_standalone_question_is_stored_only_when_it_differs(env: None) -> None:
    same = chat_history.record_chat("q", "a", [], standalone_question="q", principal=ANA)
    other = chat_history.record_chat(
        "e a porta?",
        "9443",
        [],
        conversation_id="c",
        standalone_question="Que porta?",
        principal=ANA,
    )
    turns = {t.id: t for c in chat_history.list_conversations(principal=ANA)[0] for t in c.turns}
    assert turns[same].standalone_question is None
    assert turns[other].standalone_question == "Que porta?"
