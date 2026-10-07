"""Unit tests for the eval judge's reference builder."""

from __future__ import annotations

from eval.judge import build_reference


def test_reference_answer_wins() -> None:
    assert build_reference({"reference_answer": "42", "category": "factual"}) == "42"


def test_no_answer_questions_get_a_decline_reference() -> None:
    reference = build_reference({"category": "no-answer", "expected_docs": []})
    assert "do not contain" in reference and "declines" in reference


def test_other_questions_without_docs_get_the_placeholder() -> None:
    assert build_reference({"category": "factual", "expected_docs": []}).startswith("(no reference")
