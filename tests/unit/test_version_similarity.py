"""Unit tests for the content-similarity helpers (task 34)."""

from __future__ import annotations

import pytest

from mynotes.versioning.similarity import (
    cosine,
    jaccard,
    mean_embedding,
    minhash_of,
    pair_score,
)

_SAME = "o plano de formação cobre python e bases de dados com pesquisa híbrida"
_OTHER = "a filarmónica atuou no teatro da vila com o maestro convidado"


def test_identical_texts_have_jaccard_one() -> None:
    """The same text (up to case and accents) is a perfect match."""
    assert jaccard(_SAME, _SAME) == 1.0
    assert jaccard("O PLANO de Formação", "o plano de formacao") == 1.0


def test_unrelated_texts_have_jaccard_near_zero() -> None:
    """Disjoint vocabularies share no 5-word shingles."""
    assert jaccard(_SAME, _OTHER) < 0.1
    assert jaccard(_SAME, "um texto completamente diferente de todos os outros") < 0.1


def test_short_texts_are_single_shingles() -> None:
    """A text with fewer than five words yields one shingle covering it all."""
    assert jaccard("duas palavras", "duas palavras") == 1.0
    assert jaccard("duas palavras", "outras palavras agora") == 0.0


def test_empty_texts_carry_no_similarity() -> None:
    """Without words there is no evidence: 0.0, not 1.0."""
    assert jaccard("", "") == 0.0
    assert jaccard(_SAME, "") == 0.0


def test_minhash_of_is_deterministic_and_shingle_based() -> None:
    """Equal inputs give equal signatures; num_perm controls the size."""
    assert minhash_of(_SAME).hashvalues.tolist() == minhash_of(_SAME).hashvalues.tolist()
    assert len(minhash_of(_SAME, num_perm=64).hashvalues) == 64
    assert minhash_of(_SAME, num_perm=64).jaccard(minhash_of(_SAME, num_perm=64)) == 1.0


def test_mean_embedding_averages_and_normalizes() -> None:
    """The mean is componentwise and the result is L2-normalized."""
    mean = mean_embedding([[3.0, 0.0], [0.0, 4.0]])
    assert mean == pytest.approx([0.6, 0.8])
    assert sum(value * value for value in mean) == pytest.approx(1.0)
    unit = mean_embedding([[2.0, 0.0]])
    assert unit == pytest.approx([1.0, 0.0])


def test_mean_embedding_rejects_bad_input() -> None:
    """No vectors or mismatched lengths raise ValueError."""
    with pytest.raises(ValueError, match="at least one"):
        mean_embedding([])
    with pytest.raises(ValueError, match="same length"):
        mean_embedding([[1.0, 2.0], [1.0, 2.0, 3.0]])


def test_mean_embedding_of_zero_vector_gets_a_direction() -> None:
    """An all-zero mean becomes a unit vector on index 0."""
    assert mean_embedding([[0.0, 0.0, 0.0]]) == [1.0, 0.0, 0.0]


def test_cosine() -> None:
    """Cosine is 1.0 for identical directions, 0.0 for orthogonal/zero."""
    assert cosine([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 5.0]) == pytest.approx(0.0)
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    with pytest.raises(ValueError, match="same length"):
        cosine([1.0], [1.0, 2.0])


def test_pair_score_formula_and_bonuses() -> None:
    """Half jaccard plus half cosine, +0.1 per bonus, capped at 1.0."""
    assert pair_score(0.6, 0.8, name_signal=False, archive_move=False) == pytest.approx(0.7)
    assert pair_score(0.6, 0.8, name_signal=True, archive_move=False) == pytest.approx(0.8)
    assert pair_score(0.6, 0.8, name_signal=False, archive_move=True) == pytest.approx(0.8)
    assert pair_score(0.6, 0.8, name_signal=True, archive_move=True) == pytest.approx(0.9)
    assert pair_score(1.0, 1.0, name_signal=True, archive_move=True) == 1.0
    assert pair_score(0.0, 0.0, name_signal=False, archive_move=False) == 0.0
