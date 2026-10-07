"""Unit-test fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def no_chat_history_writes(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict[str, Any]]]:
    """Unit tests have no database: streamed answers record into a list, never the real DB."""
    saved: list[dict[str, Any]] = []

    def fake_record_chat(question: str, answer: str, citations: list, **kwargs: Any) -> int:
        saved.append({"question": question, "answer": answer, "citations": citations, **kwargs})
        return len(saved)

    monkeypatch.setattr("mynotes.core.chat_history.record_chat", fake_record_chat)
    yield saved
