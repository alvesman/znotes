"""Unit tests for LLM tag proposals: messages, excerpt cut and parsing (task 38)."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from fakes import FakeChat
from mynotes.chunk.tokenizer import WhitespaceCounter
from mynotes.tags.autotag import (
    TagProposal,
    build_tag_messages,
    cut_excerpt,
    propose_tags,
    seconds_until_window_opens,
)

_VALID = json.dumps(
    {"tags": [{"tag": "ml", "confidence": 0.9, "reason": "notes about machine learning"}]}
)


def message_kwargs(**overrides: object) -> dict[str, object]:
    """The build_tag_messages keyword arguments, with optional overrides."""
    kwargs: dict[str, object] = {
        "title": "Nota Alfa",
        "rel_path": "projects/alpha/note.md",
        "headings": ["Ideia", "Detalhes"],
        "excerpt": "Texto sobre machine learning e busca híbrida.",
        "existing_tags": ["project:alpha"],
        "vocabulary": ["ml", "python"],
        "language": "pt",
    }
    kwargs.update(overrides)
    return kwargs


def test_messages_include_vocabulary_and_existing_tags() -> None:
    """The user message carries title, path, headings, excerpt, tags and vocabulary."""
    messages = build_tag_messages(**message_kwargs())

    assert [message.role for message in messages] == ["system", "user"]
    assert messages[0].content == (
        "You assign topic tags to personal project notes. Prefer tags from the "
        "vocabulary when they fit. Propose at most 5 tags that describe what the "
        "document is about (topics, technologies, domains). Don't repeat the "
        "existing tags. Tags are short, lowercase, hyphenated. Reply only with "
        "JSON matching the schema."
    )
    user = messages[1].content
    assert "Nota Alfa" in user
    assert "projects/alpha/note.md" in user
    assert "Ideia; Detalhes" in user
    assert "Texto sobre machine learning e busca híbrida." in user
    assert "project:alpha" in user, "existing tags must reach the prompt"
    assert "ml" in user and "python" in user, "vocabulary must reach the prompt"


def test_messages_cap_headings_at_thirty() -> None:
    """Only the first 30 headings are sent."""
    headings = [f"Heading {number}" for number in range(35)]

    user = build_tag_messages(**message_kwargs(headings=headings))[1].content

    assert "Heading 29" in user
    assert "Heading 30" not in user


def test_excerpt_short_text_is_untouched() -> None:
    """A text within the budget is returned as is."""
    text = "primeiro paragrafo\n\nsegundo paragrafo"

    assert cut_excerpt(text, max_tokens=100, counter=WhitespaceCounter()) == text


def test_excerpt_cut_lands_on_paragraph_boundary() -> None:
    """Whole paragraphs are kept while they fit; the next one starts a new excerpt."""
    counter = WhitespaceCounter()
    paragraphs = [f"paragraph {number} " + "word " * 40 for number in range(6)]
    text = "\n\n".join(paragraphs)

    excerpt = cut_excerpt(text, max_tokens=120, counter=counter)

    assert excerpt == "\n\n".join(paragraphs[:2])
    assert counter.count(excerpt) <= 120


def test_excerpt_keeps_a_single_oversized_paragraph() -> None:
    """The first paragraph is never emptied, even above the budget."""
    paragraph = "word " * 500

    assert cut_excerpt(paragraph, max_tokens=10, counter=WhitespaceCounter()) == paragraph


def test_propose_tags_parses_valid_json() -> None:
    """A valid answer validates on the first background call with 800 tokens."""
    chat = FakeChat([_VALID])

    proposals = propose_tags(chat, **message_kwargs())

    assert proposals == [
        TagProposal(tag="ml", confidence=0.9, reason="notes about machine learning")
    ]
    assert len(chat.calls) == 1
    assert chat.calls[0][0].role == "system"
    assert chat.kwargs[0]["priority"] == "background"
    assert chat.kwargs[0]["max_tokens"] == 800


def test_more_than_five_tags_trigger_one_repair() -> None:
    """Six proposals fail the schema; the repair answer with five wins."""
    tags = [{"tag": f"tag-{number}", "confidence": 0.5, "reason": "r"} for number in range(6)]
    chat = FakeChat([json.dumps({"tags": tags}), json.dumps({"tags": tags[:5]})])

    proposals = propose_tags(chat, **message_kwargs())

    assert [proposal.tag for proposal in proposals] == [f"tag-{number}" for number in range(5)]
    assert len(chat.calls) == 2, "exactly one repair retry"
    assert "not valid JSON" in chat.calls[1][-1].content


@pytest.mark.parametrize(
    ("window", "now", "expected"),
    [
        ("08:00-12:00", datetime(2026, 9, 24, 7, 0, 0), 3600.0),
        ("08:00-12:00", datetime(2026, 9, 24, 13, 0, 0), 19 * 3600.0),
        ("20:00-07:00", datetime(2026, 9, 24, 12, 0, 0), 8 * 3600.0),
        ("08:00-12:00", datetime(2026, 9, 24, 7, 59, 30), 30.0),
        ("08:00-08:00", datetime(2026, 9, 24, 3, 0, 0), 0.0),
        ("08:00-12:00", datetime(2026, 9, 24, 9, 0, 0), 0.0),
        ("20:00-07:00", datetime(2026, 9, 24, 2, 0, 0), 0.0),
    ],
)
def test_seconds_until_window_opens(window: str, now: datetime, expected: float) -> None:
    """Wrapped and straight windows report the delay; inside reports zero."""
    assert seconds_until_window_opens(window, now) == pytest.approx(expected)
