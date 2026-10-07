"""Unit tests for manual tags from frontmatter and hashtags (task 20)."""

from __future__ import annotations

from mynotes.tags.manual import manual_tags


def test_frontmatter_tags_and_hashtags_are_normalized_in_order() -> None:
    tags = manual_tags({"tags": ["Ideia Alfa", "ml"]}, ["Projeto-X"])
    assert tags == ["ideia-alfa", "ml", "projeto-x"]


def test_keywords_comma_string_is_split() -> None:
    assert manual_tags({"keywords": "a, b"}, []) == ["a", "b"]


def test_duplicates_are_removed_keeping_first_occurrence() -> None:
    tags = manual_tags({"tags": ["ML", "ml ", "Rag"]}, ["rag", "ML"])
    assert tags == ["ml", "rag"]


def test_tags_key_wins_over_keywords_without_duplicating() -> None:
    tags = manual_tags({"tags": ["alpha"], "keywords": "beta, alpha"}, [])
    assert tags == ["alpha", "beta"]


def test_accents_and_case_are_folded() -> None:
    assert manual_tags({"tags": ["Ação Técnica"]}, []) == ["acao-tecnica"]


def test_empty_and_blank_tags_are_dropped() -> None:
    assert manual_tags({"tags": ["", "  "]}, [""]) == []


def test_missing_frontmatter_keys_and_no_hashtags() -> None:
    assert manual_tags({}, []) == []


def test_rel_path_adds_folder_segment_tags() -> None:
    tags = manual_tags({"tags": ["ml"]}, [], rel_path="projetos/x/nota.md")
    assert tags == ["ml", "folder:projetos", "folder:x"]


def test_rel_path_none_keeps_folder_tags_out() -> None:
    assert manual_tags({"tags": ["ml"]}, [], rel_path=None) == ["ml"]
