"""Unit tests for chat follow-ups: standalone rewrite, earlier turns in the prompt."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from types import SimpleNamespace

import pytest

from fakes import FakeChat
from mynotes.llm.chat import ChatMessage
from mynotes.retrieval import answer as answer_module
from mynotes.retrieval.followup import Turn, condense_question, history_messages

TURNS = [Turn("O que é o Portainer?", "O Portainer gere contentores Docker [1].")]


def test_no_turns_means_no_model_call() -> None:
    chat = FakeChat(["should not be used"])
    assert condense_question("e a porta?", [], chat) == "e a porta?"
    assert chat.calls == []


def test_follow_up_is_rewritten_with_the_earlier_turns() -> None:
    chat = FakeChat(['"Que porta usa o Portainer?"'])
    assert condense_question("e a porta?", TURNS, chat) == "Que porta usa o Portainer?"
    prompt = chat.calls[0][-1].content
    assert "O que é o Portainer?" in prompt and "e a porta?" in prompt
    assert "[1]" not in prompt  # earlier source markers are dropped


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("Standalone question: Que porta usa o Portainer?", "Que porta usa o Portainer?"),
        ("", "e a porta?"),
        ("x" * 1000, "e a porta?"),
    ],
)
def test_odd_replies_fall_back_safely(reply: str, expected: str) -> None:
    assert condense_question("e a porta?", TURNS, FakeChat([reply])) == expected


def test_a_failing_model_keeps_the_question() -> None:
    class Broken:
        def complete(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("down")

    assert condense_question("e a porta?", TURNS, Broken()) == "e a porta?"  # type: ignore[arg-type]


def test_history_messages_alternate_and_drop_markers() -> None:
    long_answer = "texto " * 400 + "[2]"
    messages = history_messages([TURNS[0], Turn("e a porta?", long_answer)])
    assert [m.role for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[1].content == "O Portainer gere contentores Docker."
    assert messages[3].content.endswith("…") and len(messages[3].content) < 1300


def test_answer_stream_searches_the_standalone_question_and_adds_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_retrieve(query: str, filters: object = None) -> object:
        seen["query"] = query
        return SimpleNamespace(
            relevant=True,
            hits=[1],
            timings_ms={},
            analysis=SimpleNamespace(language="pt", as_of=None),
        )

    bundle = SimpleNamespace(citations=[], has_table=False, tokens=10, text="contexto")
    monkeypatch.setattr(answer_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(answer_module, "assemble_context", lambda result: bundle)
    chat = FakeChat(["A porta é 9443."])
    history = history_messages(TURNS)
    events = list(
        answer_module.answer_stream(
            "e a porta?", chat=chat, search_question="Que porta usa o Portainer?", history=history
        )
    )
    assert seen["query"] == "Que porta usa o Portainer?"
    sent: Sequence[ChatMessage] = chat.calls[0]
    assert [m.role for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[1].content == "O que é o Portainer?" and "e a porta?" in sent[-1].content
    assert [e.type for e in events][-2:] == ["sources", "done"]


def test_service_rewrites_only_inside_a_chat_with_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    from mynotes.core import service

    calls: list[dict[str, object]] = []

    def fake_answer_stream(question: str, filters: object = None, **kwargs: object) -> Iterator:
        calls.append({"question": question, **kwargs})
        yield answer_module.AnswerEvent(type="token", data="ok")

    monkeypatch.setattr(service, "answer_stream", fake_answer_stream)
    monkeypatch.setattr(
        "mynotes.llm.chat.get_chat",
        lambda purpose="answer": FakeChat(["Que porta usa o Portainer?"]),
    )
    stored = SimpleNamespace(
        turns=[SimpleNamespace(question=TURNS[0].question, answer=TURNS[0].answer)]
    )
    monkeypatch.setattr(
        "mynotes.core.chat_history.get_conversation",
        lambda key, principal: stored if key == "c1" else None,
    )

    first = list(service.ask_stream("O que é o Portainer?", conversation_id="new"))
    assert [e.type for e in first] == ["token"] and "search_question" not in calls[0]
    follow = list(service.ask_stream("e a porta?", conversation_id="c1"))
    assert follow[0].type == "rewritten" and follow[0].data == "Que porta usa o Portainer?"
    assert calls[1]["search_question"] == "Que porta usa o Portainer?"
    assert [m.role for m in calls[1]["history"]] == ["user", "assistant"]
