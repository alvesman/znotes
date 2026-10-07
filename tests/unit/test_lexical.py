"""Unit tests for lexical mention linking (docs/improvement-plan.md step 2)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mynotes.config import GraphConfig, Settings
from mynotes.graph.lexical import (
    EXTRACTOR_MENTION_CONFIDENCE,
    LEXICAL_CONFIDENCE,
    LexicalCandidate,
    LexicalParams,
    find_lexical_mentions,
    usable_alias,
)
from mynotes.text.normalize import normalize_key


def _c(entity_id: int, key: str) -> LexicalCandidate:
    return LexicalCandidate(entity_id, key)


class TestFindLexicalMentions:
    def test_whole_word_match_on_folded_text(self) -> None:
        text = normalize_key("O Projeto Tua assenta na ferramenta Stanza, não em stanzas.")
        found = find_lexical_mentions(text, [_c(1, "stanza"), _c(2, "stanzas")], 8)
        assert found == [_c(2, "stanzas"), _c(1, "stanza")]

    def test_substring_of_a_longer_word_does_not_match(self) -> None:
        text = normalize_key("the keycloakify theme and the wordlist")
        assert find_lexical_mentions(text, [_c(1, "keycloak"), _c(2, "list")], 8) == []

    def test_accents_and_case_fold_through_normalize_key(self) -> None:
        text = normalize_key("Reunião com o João Álvares sobre o Keycloak.")
        found = find_lexical_mentions(text, [_c(1, "joao alvares"), _c(2, "keycloak")], 8)
        assert {c.entity_id for c in found} == {1, 2}

    def test_punctuation_bounds_count_as_word_boundaries(self) -> None:
        text = normalize_key("tags: #dv/where, (Stanza) [[Tagloom]] 'Milvus'.")
        found = find_lexical_mentions(
            text, [_c(1, "#dv/where"), _c(2, "stanza"), _c(3, "tagloom"), _c(4, "milvus")], 8
        )
        assert {c.entity_id for c in found} == {1, 2, 3, 4}

    def test_one_result_per_entity_keeps_its_longest_key(self) -> None:
        text = normalize_key("Marta Quelhas e a Marta falaram.")
        found = find_lexical_mentions(text, [_c(1, "marta"), _c(1, "marta quelhas")], 8)
        assert found == [_c(1, "marta quelhas")]

    def test_per_chunk_cap_prefers_longer_keys_then_lower_ids(self) -> None:
        text = normalize_key("alpha beta gamma delta epsilon")
        candidates = [_c(5, "alpha"), _c(3, "gamma"), _c(9, "beta"), _c(1, "epsilon")]
        found = find_lexical_mentions(text, candidates, 2)
        assert found == [_c(1, "epsilon"), _c(3, "gamma")]

    def test_empty_key_and_empty_text(self) -> None:
        assert find_lexical_mentions("", [_c(1, "stanza")], 8) == []
        assert find_lexical_mentions("stanza", [_c(1, "")], 8) == []


class TestUsableAlias:
    @pytest.mark.parametrize(
        ("alias", "canonical", "expected"),
        [
            ("marta", "marta quelhas", True),
            ("square", "iso/iec 25000", False),
            ("flow", "authentication & authorization (proof of concept)", False),
            ("k8s", "kubernetes", True),
            ("#dv/where", "dataview where", True),
            ("caixa de correio da dst", "mailbox da dst", True),
            ("acme-org/field-app", "field-app", True),
            ("manuel", "manuel@thinkpad-z16-gen-2", False),
        ],
    )
    def test_rule(self, alias: str, canonical: str, expected: bool) -> None:
        assert usable_alias(alias, canonical) is expected


class TestParams:
    def test_defaults_match_the_config_section(self) -> None:
        params = LexicalParams.from_settings(Settings())
        assert params == LexicalParams(
            min_length=4,
            max_mentions=50,
            max_chunk_share=0.02,
            max_ratio=8.0,
            per_chunk=8,
            types=frozenset({"Technology", "Project", "Person", "Organization"}),
        )
        assert LEXICAL_CONFIDENCE < 1.0

    def test_disabled_gives_none(self) -> None:
        settings = Settings()
        settings.graph.lexical_mentions = False
        assert LexicalParams.from_settings(settings) is None

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("lexical_min_length", 1),
            ("lexical_max_mentions", -1),
            ("lexical_max_chunk_share", 0),
            ("lexical_max_chunk_share", 1.5),
            ("lexical_max_ratio", 0),
            ("lexical_per_chunk", 0),
            ("unknown_key", 1),
        ],
    )
    def test_rejects_bad_values(self, key: str, value: object) -> None:
        with pytest.raises(ValidationError):
            GraphConfig(**{key: value})


class TestQueryTimeKnob:
    def test_lexical_mentions_take_part_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mynotes.retrieval import graph_expand

        settings = Settings()
        monkeypatch.setattr(graph_expand, "get_settings", lambda: settings)
        assert graph_expand.min_mention_confidence() == 0.0
        assert graph_expand.mention_confidence_filter() == []
        settings.retrieval.graph_lexical_mentions = False
        assert graph_expand.min_mention_confidence() == EXTRACTOR_MENTION_CONFIDENCE
        assert len(graph_expand.mention_confidence_filter()) == 1
