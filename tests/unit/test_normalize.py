"""Unit tests for mynotes.text.normalize."""

from __future__ import annotations

import pytest

from mynotes.text.normalize import FOLDING_TEST_STRINGS, normalize_key, normalize_path_key


@pytest.mark.parametrize(("value", "expected"), FOLDING_TEST_STRINGS)
def test_folding_pairs(value: str, expected: str) -> None:
    assert normalize_key(value) == expected


def test_none_becomes_empty() -> None:
    assert normalize_key(None) == ""


@pytest.mark.parametrize(("value", "expected"), FOLDING_TEST_STRINGS)
def test_folding_is_idempotent(value: str, expected: str) -> None:
    once = normalize_key(value)
    assert once == expected
    assert normalize_key(once) == once


def test_unicode_whitespace_runs_collapse_to_single_space() -> None:
    assert normalize_key("a\u00a0\u2003b\tc") == "a b c"
    assert normalize_key("\u00a0 padded \u2009") == "padded"


def test_uppercase_letters_without_decomposition_are_mapped() -> None:
    assert normalize_key("İstanbul") == "istanbul"
    assert normalize_key("Øresund".upper()) == "oresund"


def test_normalize_path_key_folds_each_segment() -> None:
    assert normalize_path_key("Projetos/Ação Nova/Idea.md") == "projetos/acao nova/idea.md"


def test_normalize_path_key_preserves_separator_structure() -> None:
    assert normalize_path_key("Projetos/") == "projetos/"
    assert normalize_path_key("/Mísc/Ação.md") == "/misc/acao.md"
