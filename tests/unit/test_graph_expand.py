"""Unit tests for query-time graph expansion (task 43).

Pure logic only: :func:`score_entities` (one-hop and PageRank), the Python
filter mirror :func:`hit_matches_filters`, the graph-facts line format and the
entity-linking text helpers. Database and OpenSearch behaviour is covered by
``tests/integration/test_graph_retrieval.py``.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from mynotes.graph.expand import Subgraph, score_entities
from mynotes.retrieval.entity_link import trigram_similarity, word_ngrams
from mynotes.retrieval.graph_expand import format_relation_line
from mynotes.retrieval.hybrid import hit_matches_filters
from mynotes.retrieval.types import SearchFilters, SearchHit


def _hit(**overrides: object) -> SearchHit:
    """A current, latest, unfiltered default hit."""
    values: dict[str, object] = {
        "chunk_id": "00000000-0000-0000-0000-000000000001",
        "document_id": 1,
        "version_id": 1,
        "rank": 1,
        "score": 0.0,
        "text": "text",
        "title": None,
        "section": None,
        "heading_path": "",
        "path": "notes/a.md",
        "page": None,
        "slide": None,
        "doc_date": None,
        "doc_status": "current",
        "is_latest": True,
        "is_table": False,
        "format": "md",
        "language": None,
        "tags": [],
    }
    values.update(overrides)
    return SearchHit(**values)  # type: ignore[arg-type]


def _chain_subgraph() -> Subgraph:
    """Seed 1 —decided→ 2 and seed 1 —related_to→ 3 (unit-test weights)."""
    return Subgraph(
        nodes={1: "Seed", 2: "Decided", 3: "Related"},
        edges=[(1, 2, 10, 1.0), (1, 3, 11, 0.4)],
    )


class TestScoreEntitiesOneHop:
    def test_strong_relation_outranks_weak_one(self) -> None:
        """A seed's direct neighbour via `decided` scores above one via `related_to`."""
        scores = score_entities(_chain_subgraph(), {1: 1.0}, method="one_hop")

        assert scores[2] > scores[3]
        assert scores[2] == pytest.approx(1.0 / (1 + math.log1p(1)))  # degree-1 neighbour
        assert scores[3] == pytest.approx(0.4 / (1 + math.log1p(1)))

    def test_seeds_keep_their_score_and_neighbours_accumulate(self) -> None:
        """Two seeds feeding one neighbour sum their contributions."""
        sub = Subgraph(
            nodes={1: "A", 2: "B", 3: "C"},
            edges=[(1, 3, 10, 1.0), (2, 3, 11, 1.0)],
        )
        scores = score_entities(sub, {1: 1.0, 2: 0.5}, method="one_hop")

        # Flow 1.0 + 0.5 into a node of degree 2, damped by 1 + ln(1 + 2); the seeds
        # (max 1.0) stay as they are.
        assert scores[3] == pytest.approx(1.5 / (1 + math.log1p(2)))
        assert scores[1] == 1.0 and scores[2] == 0.5

    def test_empty_subgraph_yields_no_scores(self) -> None:
        assert score_entities(Subgraph(), {1: 1.0}, method="one_hop") == {}


class TestScoreEntitiesPpr:
    def test_nearby_nodes_score_higher_and_values_are_finite(self) -> None:
        """PageRank decays with distance; every value is finite."""
        sub = Subgraph(
            nodes={1: "A", 2: "B", 3: "C", 4: "D", 5: "E"},
            edges=[(1, 2, 1, 1.0), (2, 3, 2, 1.0), (3, 4, 3, 1.0), (4, 5, 4, 1.0)],
        )
        scores = score_entities(sub, {1: 1.0}, method="ppr")

        assert scores[2] > scores[4]
        assert all(math.isfinite(value) for value in scores.values())

    def test_zero_seed_falls_back_to_uniform(self) -> None:
        """No seed in the graph still produces a valid distribution."""
        sub = Subgraph(nodes={2: "B", 3: "C"}, edges=[(2, 3, 1, 1.0)])
        scores = score_entities(sub, {99: 1.0}, method="ppr")

        assert math.isclose(sum(scores.values()), 1.0)


class TestHitMatchesFilters:
    def test_acl_prefixes_are_enforced_exactly(self) -> None:
        """A hit outside the allowed prefixes never passes; empty means no access.

        Like the index clauses, the prefix matches a containing folder (exact
        term on the ancestors or the ``prefix/`` path prefix); an exact file
        path is not its own ancestor and never matches.
        """
        hit = _hit(path="privado/historia.md")

        assert hit_matches_filters(hit, SearchFilters(acl_prefixes=["privado"]))
        assert not hit_matches_filters(hit, SearchFilters(acl_prefixes=["projetos"]))
        assert not hit_matches_filters(hit, SearchFilters(acl_prefixes=["privado/notas"]))
        assert not hit_matches_filters(hit, SearchFilters(acl_prefixes=["privado/historia.md"]))
        assert not hit_matches_filters(hit, SearchFilters(acl_prefixes=[]))
        assert hit_matches_filters(hit, SearchFilters(acl_prefixes=None))

    def test_tag_filters_require_every_tag(self) -> None:
        """All filter tags must be present on the hit."""
        hit = _hit(tags=["trabalho", "projeto-x"])

        assert hit_matches_filters(hit, SearchFilters(tags=["trabalho"]))
        assert hit_matches_filters(hit, SearchFilters(tags=["trabalho", "projeto-x"]))
        assert not hit_matches_filters(hit, SearchFilters(tags=["trabalho", "outro"]))

    def test_status_and_recency_gates_mirror_the_index_clauses(self) -> None:
        """Superseded needs history intent; old versions need include_old_versions."""
        superseded = _hit(doc_status="superseded")
        assert not hit_matches_filters(superseded, SearchFilters())
        assert hit_matches_filters(superseded, SearchFilters(include_superseded=True))

        old = _hit(is_latest=False)
        assert not hit_matches_filters(old, SearchFilters())
        assert hit_matches_filters(old, SearchFilters(include_old_versions=True))

    def test_folders_formats_dates_and_documents(self) -> None:
        """The remaining clauses match the OpenSearch semantics."""
        hit = _hit(
            path="Projetos/Alfa/nota.md", format="md", doc_date=datetime(2026, 6, 1, tzinfo=UTC)
        )

        assert hit_matches_filters(hit, SearchFilters(folders=["projetos"]))
        assert not hit_matches_filters(hit, SearchFilters(folders=["arquivo"]))
        assert hit_matches_filters(hit, SearchFilters(formats=["md"]))
        assert not hit_matches_filters(hit, SearchFilters(formats=["pdf"]))
        assert hit_matches_filters(hit, SearchFilters(since=datetime(2026, 1, 1, tzinfo=UTC)))
        assert not hit_matches_filters(hit, SearchFilters(since=datetime(2026, 7, 1, tzinfo=UTC)))
        assert hit_matches_filters(hit, SearchFilters(document_ids=[1]))
        assert not hit_matches_filters(hit, SearchFilters(document_ids=[2]))


class TestGraphFactsLine:
    def test_full_line_with_attribute_date_and_historical(self) -> None:
        """Every optional piece renders in the specified order."""
        line = format_relation_line(
            src="Ana",
            dst="Rust",
            relation_type="proposes",
            attribute="tier",
            value="paid",
            status="historical",
            valid_from=datetime(2026, 6, 15, tzinfo=UTC),
        )

        assert line == "Ana —proposes→ Rust [tier = paid] (2026-06-15) [historical]"

    def test_active_line_without_optional_parts(self) -> None:
        line = format_relation_line(
            src="Projeto Alfa",
            dst="Rust",
            relation_type="uses",
            attribute=None,
            value=None,
            status="active",
            valid_from=None,
        )

        assert line == "Projeto Alfa —uses→ Rust"


class TestEntityLinkHelpers:
    def test_word_ngrams_cover_one_to_four_words(self) -> None:
        assert word_ngrams("quem propos a tecnologia") == [
            "quem",
            "propos",
            "a",
            "tecnologia",
            "quem propos",
            "propos a",
            "a tecnologia",
            "quem propos a",
            "propos a tecnologia",
            "quem propos a tecnologia",
        ]

    def test_trigram_similarity_bounds(self) -> None:
        assert trigram_similarity("projeto alfa", "projeto alfa") == 1.0
        assert trigram_similarity("rust", "alfa") < 0.2
        assert trigram_similarity("", "") == 1.0
        assert trigram_similarity("", "rust") == 0.0


def test_cap_per_entity_limits_one_entity_and_keeps_unowned_chunks() -> None:
    from uuid import uuid4

    from mynotes.graph.expand import cap_per_entity

    ids = [uuid4() for _ in range(8)]
    ordered = [(ids[i], 1.0 - 0.01 * i) for i in range(8)]
    owners = {ids[0]: 1, ids[1]: 1, ids[2]: 1, ids[3]: 1, ids[4]: 2, ids[5]: 1}
    kept = cap_per_entity(ordered, owners, cap=2, limit=10)
    assert [c for c, _ in kept] == [ids[0], ids[1], ids[4], ids[6], ids[7]]
    assert len(cap_per_entity(ordered, {}, cap=1, limit=3)) == 3
