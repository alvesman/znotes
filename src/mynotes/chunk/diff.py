"""Chunk diff between two document versions (PLAN.md §7.3).

New chunks are matched to the previous version's chunks by ``content_hash``
(multiset semantics). Matches reuse the stored embedding and KG extraction;
only added and removed chunks cost model calls.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from mynotes.chunk.chunker import ChunkDraft


@dataclass
class ChunkDiff:
    """Difference between the previous and the new version's chunks.

    Attributes:
        reused: ``(new ordinal, old content_hash)`` pairs matched by content
            hash; their embeddings can be reused.
        added: New ordinals with no match in the old version.
        removed: Old content hashes with no match in the new version.
    """

    reused: list[tuple[int, str]] = field(default_factory=list)
    added: list[int] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


def diff_chunks(old_hashes: Sequence[str], new: Sequence[ChunkDraft]) -> ChunkDiff:
    """Match the new version's chunks against the old version's content hashes.

    Args:
        old_hashes: Content hashes of the previous version's chunks, in
            ordinal order (duplicates allowed).
        new: The new version's chunk drafts.

    Returns:
        The reused/added/removed classification. A hash appearing twice in
        the old version and once in the new one yields one reuse and one
        removal.
    """
    remaining = Counter(old_hashes)
    reused: list[tuple[int, str]] = []
    added: list[int] = []
    for draft in new:
        if remaining.get(draft.content_hash, 0) > 0:
            remaining[draft.content_hash] -= 1
            reused.append((draft.ordinal, draft.content_hash))
        else:
            added.append(draft.ordinal)
    removed: list[str] = []
    for content_hash, count in remaining.items():
        removed.extend([content_hash] * count)
    return ChunkDiff(reused=reused, added=added, removed=removed)
