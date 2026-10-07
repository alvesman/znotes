"""Unit tests for the client-side RRF fusion (task 28, PLAN.md §9 step 5)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mynotes.retrieval.fuse import rrf_fuse
from mynotes.retrieval.types import SearchHit


def _hit(chunk_id: str, rank: int, *, document_id: int = 1, text: str = "text") -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id=document_id,
        version_id=1,
        rank=rank,
        score=0.0,
        text=text,
        title=None,
        section=None,
        heading_path="",
        path=f"notes/{chunk_id}.md",
        page=None,
        slide=None,
        doc_date=datetime(2026, 1, 1, tzinfo=UTC),
        doc_status="current",
        is_latest=True,
        is_table=False,
        format="md",
        language=None,
        tags=[],
    )


def test_two_rankings_fuse_with_hand_computed_scores() -> None:
    # k = 60; weights: hybrid 1.0, expander 0.5.
    #   score(a) = 1.0/(60+1) + 0.5/(60+3) = 0.01639344 + 0.00793651 = 0.02432995
    #   score(c) = 1.0/(60+3) + 0.5/(60+1) = 0.01587302 + 0.00819672 = 0.02406974
    #   score(b) = 1.0/(60+2)              = 0.01612903
    #   score(d) =           0.5/(60+2)    = 0.00806452
    hybrid = [_hit("a", 1), _hit("b", 2), _hit("c", 3)]
    expander = [_hit("c", 1), _hit("d", 2), _hit("a", 3)]

    fused = rrf_fuse(
        {"hybrid": hybrid, "expander": expander},
        {"hybrid": 1.0, "expander": 0.5},
        k=60,
    )

    assert [hit.chunk_id for hit in fused] == ["a", "c", "b", "d"]
    assert fused[0].score == pytest.approx(1.0 / 61 + 0.5 / 63)
    assert fused[1].score == pytest.approx(1.0 / 63 + 0.5 / 61)
    assert fused[2].score == pytest.approx(1.0 / 62)
    assert fused[3].score == pytest.approx(0.5 / 62)
    assert fused[0].sources == {"hybrid": 1, "expander": 3}
    assert fused[1].sources == {"hybrid": 3, "expander": 1}
    assert fused[2].sources == {"hybrid": 2}
    assert fused[3].sources == {"expander": 2}
    assert [hit.rank for hit in fused] == [1, 2, 3, 4]


def test_first_seen_hit_object_is_kept() -> None:
    hybrid = [_hit("a", 1)]
    expander = [_hit("a", 1)]

    fused = rrf_fuse({"hybrid": hybrid, "expander": expander}, {"hybrid": 1.0}, k=60)

    assert len(fused) == 1
    assert fused[0] is hybrid[0]
    assert fused[0].score == pytest.approx(1.0 / 61 + 1.0 / 61)


def test_ties_prefer_the_better_hybrid_rank_then_chunk_id() -> None:
    # Without weights for "expander" it defaults to 1.0, so a and c get the
    # same total (1/61 + 1/63) and the hybrid rank breaks the tie. b and d
    # both score 1/62; b has a hybrid rank, d has none, so b comes first.
    hybrid = [_hit("a", 1), _hit("b", 2), _hit("c", 3)]
    expander = [_hit("c", 1), _hit("d", 2), _hit("a", 3)]

    fused = rrf_fuse({"hybrid": hybrid, "expander": expander}, {"hybrid": 1.0}, k=60)

    assert [hit.chunk_id for hit in fused] == ["a", "c", "b", "d"]
    assert fused[0].score == pytest.approx(fused[1].score)
    assert fused[2].score == pytest.approx(fused[3].score)


def test_chunk_id_breaks_ties_without_hybrid_ranks() -> None:
    # Two equal-weight sources, both hits at rank 1 with no hybrid rank.
    left = [_hit("zzz", 1)]
    right = [_hit("aaa", 1)]

    fused = rrf_fuse(
        {"left": left, "right": right},
        {"left": 0.5, "right": 0.5},
        k=60,
    )

    assert [hit.chunk_id for hit in fused] == ["aaa", "zzz"]
    assert fused[0].sources == {"right": 1}


def test_weights_are_respected() -> None:
    # With weight 2.0 the expander dominates:
    #   score(c) = 1.0/63 + 2.0/61 = 0.04866 > score(a) = 1.0/61 + 2.0/63 = 0.04814
    #   score(d) = 2.0/62 = 0.03226 > score(b) = 1.0/62 = 0.01613
    hybrid = [_hit("a", 1), _hit("b", 2), _hit("c", 3)]
    expander = [_hit("c", 1), _hit("d", 2), _hit("a", 3)]

    fused = rrf_fuse(
        {"hybrid": hybrid, "expander": expander},
        {"hybrid": 1.0, "expander": 2.0},
        k=60,
    )

    assert [hit.chunk_id for hit in fused] == ["c", "a", "d", "b"]


def test_empty_rankings_fuse_to_nothing() -> None:
    assert rrf_fuse({}, {}, k=60) == []
    assert rrf_fuse({"hybrid": []}, {"hybrid": 1.0}, k=60) == []


def test_a_hint_from_any_source_survives_the_merge() -> None:
    hybrid = [_hit("c1", 1), _hit("c2", 2)]
    graph_copy = _hit("c2", 1)
    graph_copy.query_hint, graph_copy.query_hint_type = "Memoranda", "Technology"
    fused = rrf_fuse({"hybrid": hybrid, "graph": [graph_copy]}, {}, 20)
    merged = next(h for h in fused if h.chunk_id == "c2")
    assert merged is hybrid[1]  # first-seen object kept
    assert (merged.query_hint, merged.query_hint_type) == ("Memoranda", "Technology")
    assert next(h for h in fused if h.chunk_id == "c1").query_hint is None
