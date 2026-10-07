"""Unit tests for the answer prompt construction (task 30, PLAN.md §9 step 7)."""

from __future__ import annotations

from mynotes.llm.chat import ChatMessage
from mynotes.retrieval.assemble import Citation, ContextBundle
from mynotes.retrieval.prompts import NOT_FOUND, SYSTEM_PROMPT, build_messages


def _bundle(*, has_table: bool = False, tokens: int = 100) -> ContextBundle:
    """A small canned bundle with one citation."""
    citation = Citation(
        n=1,
        document_id=1,
        title="Nota",
        path="notes/nota.md",
        page=None,
        slide=None,
        section=None,
        doc_date=None,
        is_current=True,
    )
    return ContextBundle(
        text="texto do contexto", citations=[citation], tokens=tokens, has_table=has_table
    )


def _messages(
    *, has_table: bool = False, tokens: int = 100, language: str | None = None
) -> list[ChatMessage]:
    return build_messages("pergunta", _bundle(has_table=has_table, tokens=tokens), language)


def test_system_prompt_has_citation_and_historical_rules() -> None:
    assert "[1]" in SYSTEM_PROMPT
    assert "historical" in SYSTEM_PROMPT
    assert "current documents" in SYSTEM_PROMPT


def test_system_prompt_has_the_list_and_identifier_rule() -> None:
    assert "every item the source lists for what is asked" in SYSTEM_PROMPT
    assert "Copy identifiers in full" in SYSTEM_PROMPT


def test_table_rule_only_when_context_has_table() -> None:
    plain = _messages(has_table=False)[0].content
    assert "The context contains tables" not in plain
    with_table = _messages(has_table=True)[0].content
    assert "The context contains tables" in with_table


def test_details_rule_only_for_large_contexts() -> None:
    small = _messages(tokens=200)[0].content
    assert "several details" not in small
    large = _messages(tokens=201)[0].content
    assert "several details" in large


def test_language_directives() -> None:
    assert "Responde em português europeu." in _messages(language="pt")[1].content
    assert "Answer in English." in _messages(language="en")[1].content
    assert "Answer in the same language as the question." in _messages(language=None)[1].content


def test_user_message_layout() -> None:
    bundle = _bundle(has_table=True, tokens=500)
    messages = build_messages("Qual é o prazo?", bundle, "pt")
    assert [message.role for message in messages] == ["system", "user"]
    assert messages[1].content == (
        f"Context:\n{bundle.text}\n\nQuestion: Qual é o prazo?\nResponde em português europeu."
    )
    assert messages[0].content.startswith(SYSTEM_PROMPT)


def test_not_found_texts() -> None:
    assert NOT_FOUND["pt"] == "Não encontrei essa informação nas tuas notas."
    assert NOT_FOUND["en"] == "I couldn't find that information in your notes."


def test_as_of_rule_only_when_the_question_has_a_date() -> None:
    from datetime import UTC, datetime

    bundle = _bundle(has_table=False, tokens=50)
    plain = build_messages("Qual era o prazo?", bundle, "pt")
    assert "as of" not in plain[0].content
    dated = build_messages(
        "Qual era o prazo? as-of: 2025-09-20",
        bundle,
        "pt",
        as_of=datetime(2025, 9, 20, tzinfo=UTC),
    )
    assert "11. The question asks for the state of things as of 2025-09-20." in dated[0].content
    assert dated[1].content == plain[1].content.replace(
        "Qual era o prazo?", "Qual era o prazo? as-of: 2025-09-20"
    )
