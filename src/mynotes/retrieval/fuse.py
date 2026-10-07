"""Client-side Reciprocal Rank Fusion (PLAN.md §6, §9 step 5).

The same RRF formula as the OpenSearch ``mynotes-hybrid-rrf`` pipeline
(``weight / (rank_constant + rank)``) merges the hybrid ranking with the
rankings contributed by the expanders (versions in task 35, graph in task 43),
so every source shares one score scale and one ``rank_constant``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from mynotes.retrieval.types import SearchHit

_HYBRID_SOURCE = "hybrid"


def rrf_fuse(
    rankings: Mapping[str, Sequence[SearchHit]],
    weights: Mapping[str, float],
    k: int,
) -> list[SearchHit]:
    """Fuse per-source rankings into one list with Reciprocal Rank Fusion.

    Args:
        rankings: Source name to hits in that source's rank order (best
            first); the rank of a hit is its 1-based position.
        weights: Fusion weight per source; sources missing from the mapping
            default to 1.0.
        k: The RRF rank constant, shared with the OpenSearch pipeline.

    Returns:
        The fused hits sorted by fused score descending (ties: better
        ``hybrid`` rank first, then ``chunk_id``) with fresh 1-based ranks.
        One hit per ``chunk_id``: the first-seen object, mutated to carry
        ``sources`` (source name to rank), the fused ``score`` and the first
        ``query_hint`` any source gave it.
    """
    by_chunk: dict[str, SearchHit] = {}
    for source, hits in rankings.items():
        weight = float(weights.get(source, 1.0))
        for rank, hit in enumerate(hits, start=1):
            contribution = weight / (k + rank)
            merged = by_chunk.get(hit.chunk_id)
            if merged is None:
                hit.score = contribution
                hit.sources[source] = rank
                by_chunk[hit.chunk_id] = hit
            else:
                merged.score += contribution
                merged.sources[source] = rank
                # A chunk found by the text search *and* as graph evidence keeps the graph's
                # hint: dropping it reranks the tool note's fact chunk against the bare
                # question (golden set v3 h042: 0.007 instead of a hinted score).
                if merged.query_hint is None and hit.query_hint is not None:
                    merged.query_hint = hit.query_hint
                    merged.query_hint_type = hit.query_hint_type
    fused = sorted(
        by_chunk.values(),
        key=lambda hit: (-hit.score, hit.sources.get(_HYBRID_SOURCE, math.inf), hit.chunk_id),
    )
    for rank, hit in enumerate(fused, start=1):
        hit.rank = rank
    return fused
