"""Unit tests for the entity resolution decision rules (task 41)."""

from __future__ import annotations

import pytest

from mynotes.graph.resolve import COSINE_THRESHOLD, TRIGRAM_THRESHOLD, decide_match


def test_exact_match_links_and_ignores_fuzzy_candidates() -> None:
    decision = decide_match(
        exact_id=7,
        best_trgm=(7, 1.0),
        best_knn=(7, 1.0),
        distinct_pairs=set(),
    )
    assert decision.action == "link"
    assert decision.entity_id == 7
    assert decision.candidate_id is None
    assert decision.trigram == 1.0
    assert decision.cosine == 1.0


def test_exact_match_links_even_when_fuzzy_points_elsewhere() -> None:
    decision = decide_match(
        exact_id=7,
        best_trgm=(9, 0.9),
        best_knn=(9, 0.95),
        distinct_pairs=set(),
    )
    assert decision.action == "link"
    assert decision.entity_id == 7


def test_trigram_and_knn_agree_adds_alias() -> None:
    decision = decide_match(
        exact_id=None,
        best_trgm=(4, 0.75),
        best_knn=(4, 0.93),
        distinct_pairs=set(),
    )
    assert decision.action == "link_add_alias"
    assert decision.entity_id == 4
    assert decision.candidate_id is None
    assert decision.trigram == 0.75
    assert decision.cosine == 0.93


def test_only_trigram_passes_adjudicates() -> None:
    decision = decide_match(
        exact_id=None,
        best_trgm=(4, 0.65),
        best_knn=(4, 0.5),
        distinct_pairs=set(),
    )
    assert decision.action == "new_adjudicate"
    assert decision.entity_id is None
    assert decision.candidate_id == 4
    assert decision.cosine == 0.5


def test_only_knn_passes_adjudicates() -> None:
    decision = decide_match(
        exact_id=None,
        best_trgm=(2, 0.4),
        best_knn=(6, 0.9),
        distinct_pairs=set(),
    )
    assert decision.action == "new_adjudicate"
    assert decision.candidate_id == 6
    assert decision.trigram == 0.4


def test_disagreeing_signals_adjudicate_against_trigram() -> None:
    decision = decide_match(
        exact_id=None,
        best_trgm=(4, 0.7),
        best_knn=(6, 0.9),
        distinct_pairs=set(),
    )
    assert decision.action == "new_adjudicate"
    assert decision.candidate_id == 4


def test_disagreement_of_distinct_pair_creates_without_adjudication() -> None:
    decision = decide_match(
        exact_id=None,
        best_trgm=(4, 0.7),
        best_knn=(6, 0.9),
        distinct_pairs={(4, 6)},
    )
    assert decision.action == "new"
    assert decision.candidate_id is None
    assert decision.trigram == 0.7
    assert decision.cosine == 0.9


def test_no_candidates_create_new() -> None:
    decision = decide_match(
        exact_id=None,
        best_trgm=None,
        best_knn=None,
        distinct_pairs=set(),
    )
    assert decision.action == "new"
    assert decision.entity_id is None
    assert decision.candidate_id is None
    assert decision.trigram is None
    assert decision.cosine is None


@pytest.mark.parametrize("score", [0.0, 0.59, TRIGRAM_THRESHOLD - 0.001])
def test_trigram_below_threshold_never_counts(score: float) -> None:
    decision = decide_match(
        exact_id=None,
        best_trgm=(4, score),
        best_knn=None,
        distinct_pairs=set(),
    )
    assert decision.action == "new"
    assert decision.trigram == score


@pytest.mark.parametrize("score", [0.0, 0.5, COSINE_THRESHOLD - 0.001])
def test_knn_below_threshold_never_counts(score: float) -> None:
    decision = decide_match(
        exact_id=None,
        best_trgm=None,
        best_knn=(4, score),
        distinct_pairs=set(),
    )
    assert decision.action == "new"
    assert decision.cosine == score


def test_threshold_values_are_exclusive_boundaries() -> None:
    at_trigram = decide_match(
        exact_id=None, best_trgm=(4, TRIGRAM_THRESHOLD), best_knn=None, distinct_pairs=set()
    )
    at_cosine = decide_match(
        exact_id=None, best_trgm=None, best_knn=(4, COSINE_THRESHOLD), distinct_pairs=set()
    )
    assert at_trigram.action == "new_adjudicate"
    assert at_cosine.action == "new_adjudicate"


def test_distinct_pair_override_is_normalized_pair_order() -> None:
    decision = decide_match(
        exact_id=None,
        best_trgm=(9, 0.8),
        best_knn=(2, 0.92),
        distinct_pairs={(2, 9)},
    )
    assert decision.action == "new"
