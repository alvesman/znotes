"""Content similarity for successor detection (PLAN.md §7.5 step 3).

Candidate document pairs are confirmed by two content signals: MinHash Jaccard
over 5-word shingles of the normalized text (fast, lexical) and the cosine of
the documents' mean chunk embeddings (semantic). :func:`pair_score` combines
both 50/50 and adds the bonuses for the name signal (same stem) and an
archive-folder move, capped at 1.0.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from functools import lru_cache

import numpy as np

from mynotes.text.normalize import normalize_key

_SHINGLE_WORDS = 5

_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1
_SEED = 1
_BATCH = 4096
"""Shingles hashed per step: bounds the ``shingles by num_perm`` work matrix."""


@lru_cache(maxsize=4)
def _permutations(num_perm: int) -> tuple[np.ndarray, np.ndarray]:
    """The ``a`` and ``b`` of the ``num_perm`` universal hash functions, fixed by seed.

    Both are below 2**32, so ``a * h + b`` with a 32-bit ``h`` never overflows
    an unsigned 64-bit integer.
    """
    generator = np.random.default_rng(_SEED)
    a = generator.integers(1, _MAX_HASH, size=num_perm, dtype=np.uint64)
    b = generator.integers(0, _MAX_HASH, size=num_perm, dtype=np.uint64)
    return a, b


class MinHash:
    """A MinHash signature: the minimum of ``num_perm`` hashes over the items seen.

    Each item is hashed to 32 bits (the first bytes of its SHA-1), then
    permuted by ``(a * h + b) mod (2**61 - 1)``, masked to 32 bits; the
    signature keeps the minimum per permutation. Same scheme as datasketch's
    ``MinHash``, which this replaces: datasketch pulled scipy (139 MB in the
    image) for an LSH index mynotes never used.
    """

    __slots__ = ("_a", "_b", "hashvalues")

    def __init__(self, num_perm: int = 128) -> None:
        self._a, self._b = _permutations(num_perm)
        self.hashvalues = np.full(num_perm, _MAX_HASH, dtype=np.uint64)

    def update_batch(self, items: Iterable[bytes]) -> None:
        """Fold the items into the signature."""
        batch: list[int] = []
        for item in items:
            digest = hashlib.sha1(item, usedforsecurity=False).digest()
            batch.append(int.from_bytes(digest[:4], "little"))
            if len(batch) == _BATCH:
                self._fold(batch)
                batch = []
        if batch:
            self._fold(batch)

    def _fold(self, hashes: list[int]) -> None:
        values = np.asarray(hashes, dtype=np.uint64)
        permuted = ((values[:, None] * self._a + self._b) % _MERSENNE_PRIME) & _MAX_HASH
        self.hashvalues = np.minimum(self.hashvalues, permuted.min(axis=0))

    def jaccard(self, other: MinHash) -> float:
        """Estimate the Jaccard similarity of the two item sets."""
        if len(self.hashvalues) != len(other.hashvalues):
            raise ValueError("MinHash signatures must have the same num_perm")
        matches = int(np.count_nonzero(self.hashvalues == other.hashvalues))
        return matches / len(self.hashvalues)


def _shingles(text: str) -> list[bytes]:
    """Return the 5-word shingles of the normalized text, encoded.

    A text with fewer than five words yields one shingle covering all of it;
    a text without words yields no shingles.
    """
    words = normalize_key(text).split()
    if not words:
        return []
    if len(words) < _SHINGLE_WORDS:
        return [" ".join(words).encode("utf-8")]
    spans = range(len(words) - _SHINGLE_WORDS + 1)
    return [" ".join(words[i : i + _SHINGLE_WORDS]).encode("utf-8") for i in spans]


def minhash_of(text: str, num_perm: int = 128) -> MinHash:
    """Build the MinHash signature of a document text.

    Args:
        text: The document text (plain text of its latest version's chunks).
        num_perm: Number of permutation functions in the signature.

    Returns:
        A :class:`MinHash` over the 5-word shingles of
        :func:`mynotes.text.normalize.normalize_key`-normalized ``text``.
    """
    return _minhash_from(_shingles(text), num_perm)


def _minhash_from(shingles: Sequence[bytes], num_perm: int = 128) -> MinHash:
    """Build a MinHash signature from precomputed shingles."""
    signature = MinHash(num_perm=num_perm)
    signature.update_batch(shingles)
    return signature


def jaccard(a: str, b: str) -> float:
    """Estimate the Jaccard similarity of two texts over their shingles.

    Args:
        a: First document text.
        b: Second document text.

    Returns:
        The MinHash Jaccard estimate in [0.0, 1.0]. Two texts without words
        (empty shingle sets) carry no similarity evidence and score 0.0.
    """
    shingles_a, shingles_b = _shingles(a), _shingles(b)
    if not shingles_a or not shingles_b:
        return 0.0
    return _minhash_from(shingles_a).jaccard(_minhash_from(shingles_b))


def mean_embedding(vectors: Sequence[Sequence[float]]) -> list[float]:
    """Average vectors componentwise and L2-normalize the result.

    Args:
        vectors: One or more embedding vectors of equal length.

    Returns:
        The unit-length mean vector. An all-zero mean becomes a unit vector
        on index 0 (mirroring the test embedder), so downstream cosine keeps
        working.

    Raises:
        ValueError: If ``vectors`` is empty or the lengths differ.
    """
    if not vectors:
        raise ValueError("mean_embedding needs at least one vector")
    length = len(vectors[0])
    if any(len(vector) != length for vector in vectors):
        raise ValueError("all vectors must have the same length")
    mean = [0.0] * length
    for vector in vectors:
        for index, value in enumerate(vector):
            mean[index] += value
    norm = math.sqrt(sum(value * value for value in mean))
    if norm == 0.0:
        if mean:
            mean[0] = 1.0
        return mean
    return [value / norm for value in mean]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute the cosine of two vectors (0.0 when either is all-zero)."""
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for left, right in zip(a, b, strict=True):
        dot += left * right
        norm_a += left * left
        norm_b += right * right
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def pair_score(jac: float, cos: float, *, name_signal: bool, archive_move: bool) -> float:
    """Combine the content signals and bonuses into one score.

    Args:
        jac: Shingle Jaccard of the two documents (0..1).
        cos: Cosine of the documents' mean embeddings (0..1).
        name_signal: True when both files share the same normalized stem.
        archive_move: True when one of the two lives under an archive folder.

    Returns:
        ``min(1.0, 0.5*jac + 0.5*cos + 0.1*name_signal + 0.1*archive_move)``.
    """
    return min(
        1.0,
        0.5 * jac + 0.5 * cos + (0.1 if name_signal else 0.0) + (0.1 if archive_move else 0.0),
    )
