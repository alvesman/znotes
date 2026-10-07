"""Unit tests for the chunk diff between document versions (task 19)."""

from __future__ import annotations

from mynotes.chunk.chunker import ChunkDraft
from mynotes.chunk.diff import ChunkDiff, diff_chunks


def _draft(ordinal: int, content_hash: str) -> ChunkDraft:
    return ChunkDraft(ordinal=ordinal, text=content_hash, content_hash=content_hash)


def test_multiset_matching_reuses_adds_and_removes() -> None:
    old = ["h1", "h2", "h2", "h3"]
    new = [_draft(0, "h2"), _draft(1, "h3"), _draft(2, "h4")]
    result = diff_chunks(old, new)
    assert result.reused == [(0, "h2"), (1, "h3")]
    assert result.added == [2]
    assert result.removed == ["h1", "h2"]


def test_identical_versions_reuse_everything() -> None:
    old = ["h1", "h2"]
    new = [_draft(0, "h1"), _draft(1, "h2")]
    result = diff_chunks(old, new)
    assert result.reused == [(0, "h1"), (1, "h2")]
    assert result.added == []
    assert result.removed == []


def test_empty_old_version_adds_everything() -> None:
    new = [_draft(0, "h1"), _draft(1, "h2")]
    result = diff_chunks([], new)
    assert result == ChunkDiff(reused=[], added=[0, 1], removed=[])


def test_empty_new_version_removes_everything() -> None:
    result = diff_chunks(["h1", "h2"], [])
    assert result == ChunkDiff(reused=[], added=[], removed=["h1", "h2"])


def test_duplicate_hash_in_new_consumes_one_old_match() -> None:
    old = ["h1", "h1"]
    new = [_draft(0, "h1"), _draft(1, "h1"), _draft(2, "h1")]
    result = diff_chunks(old, new)
    assert result.reused == [(0, "h1"), (1, "h1")]
    assert result.added == [2]
    assert result.removed == []
