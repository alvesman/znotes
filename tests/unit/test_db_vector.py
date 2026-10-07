"""Unit tests for the float16 vector packing helpers (no database needed)."""

from __future__ import annotations

import pytest

from mynotes.db.models import pack_vector, unpack_vector


def test_round_trip_within_float16_precision() -> None:
    """Values survive pack/unpack within float16 (~2^-11) relative precision."""
    values = [0.0, 1.0, -1.0, 0.5, 1.0 / 3.0, 123.456, -9876.5]
    packed = pack_vector(values)
    assert len(packed) == 2 * len(values)
    restored = unpack_vector(packed)
    assert len(restored) == len(values)
    for original, actual in zip(values, restored, strict=True):
        assert actual == pytest.approx(original, rel=1e-3)


def test_empty_vector_round_trips() -> None:
    """An empty vector packs to zero bytes and back."""
    assert unpack_vector(pack_vector([])) == []
