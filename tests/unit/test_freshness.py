"""Unit tests for the freshness weighting (graph edge decay and recency boost)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from fakes import FakeReranker
from mynotes.config import RetrievalConfig, Settings
from mynotes.graph.expand import freshness_factor
from mynotes.retrieval.analyze import QueryAnalysis
from mynotes.retrieval.pipeline import recency_params
from mynotes.retrieval.rerank import RecencyParams, rerank_hits
from mynotes.retrieval.types import SearchFilters, SearchHit

NOW = datetime(2026, 10, 1, tzinfo=UTC)


class TestFreshnessFactor:
    def test_halves_every_half_life(self) -> None:
        assert freshness_factor(NOW - timedelta(days=365), NOW, 365) == pytest.approx(0.5)
        assert freshness_factor(NOW - timedelta(days=730), NOW, 365) == pytest.approx(0.25)
        assert freshness_factor(NOW - timedelta(days=30), NOW, 30) == pytest.approx(0.5)

    def test_now_future_and_unknown_dates_are_not_penalized(self) -> None:
        assert freshness_factor(NOW, NOW, 365) == 1.0
        assert freshness_factor(NOW + timedelta(days=10), NOW, 365) == 1.0
        assert freshness_factor(None, NOW, 365) == 1.0

    def test_is_monotonic_in_age(self) -> None:
        ages = [1, 10, 100, 1000]
        factors = [freshness_factor(NOW - timedelta(days=a), NOW, 200) for a in ages]
        assert factors == sorted(factors, reverse=True)
        assert all(0 < f < 1 for f in factors)


class TestRecencyParams:
    def test_multiplier_is_floored(self) -> None:
        params = RecencyParams(half_life_days=365, min_multiplier=0.8, reference=NOW)
        assert params.multiplier(NOW) == pytest.approx(1.0)
        assert params.multiplier(NOW - timedelta(days=365)) == pytest.approx(0.9)
        assert params.multiplier(NOW - timedelta(days=365 * 50)) == pytest.approx(0.8, abs=1e-6)
        assert params.multiplier(None) == 1.0


def _hit(chunk_id: str, rank: int, doc_date: datetime | None, document_id: int) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id=document_id,
        version_id=document_id,
        rank=rank,
        score=0.01,
        text="prazo do projeto alfa",
        title=None,
        section=None,
        heading_path="",
        path=f"notes/{chunk_id}.md",
        page=None,
        slide=None,
        doc_date=doc_date,
        doc_status="current",
        is_latest=True,
        is_table=False,
        format="md",
        language=None,
        tags=[],
    )


class TestRerankRecencyBoost:
    """Identical texts tie on the fake reranker, so only the boost can reorder them."""

    def test_without_boost_the_older_hit_keeps_its_place(self) -> None:
        old = _hit("old", 1, NOW - timedelta(days=800), document_id=1)
        new = _hit("new", 2, NOW - timedelta(days=5), document_id=2)
        ranked = rerank_hits("prazo projeto alfa", [old, new], FakeReranker(), top=10)
        assert [hit.chunk_id for hit in ranked] == ["old", "new"]

    def test_with_boost_the_newer_hit_wins_and_raw_scores_are_kept(self) -> None:
        old = _hit("old", 1, NOW - timedelta(days=800), document_id=1)
        new = _hit("new", 2, NOW - timedelta(days=5), document_id=2)
        params = RecencyParams(half_life_days=365, min_multiplier=0.8, reference=NOW)
        ranked = rerank_hits(
            "prazo projeto alfa", [old, new], FakeReranker(), top=10, recency=params
        )
        assert [hit.chunk_id for hit in ranked] == ["new", "old"]
        assert ranked[0].rank == 1 and ranked[1].rank == 2
        assert old.rerank_score == new.rerank_score  # the reranker's own score is untouched
        assert old.score == new.score == 0.01  # the fused score is untouched

    def test_undated_hit_is_not_penalized(self) -> None:
        undated = _hit("undated", 1, None, document_id=1)
        new = _hit("new", 2, NOW, document_id=2)
        params = RecencyParams(half_life_days=365, min_multiplier=0.8, reference=NOW)
        ranked = rerank_hits(
            "prazo projeto alfa", [undated, new], FakeReranker(), top=10, recency=params
        )
        assert [hit.chunk_id for hit in ranked] == ["undated", "new"]


def _analysis(intent: str, as_of: datetime | None) -> QueryAnalysis:
    return QueryAnalysis(
        raw="q",
        text="q",
        filters=SearchFilters(),
        intent=intent,  # type: ignore[arg-type]
        as_of=as_of,
        language=None,
    )


class TestRecencyParamsFromSettings:
    def test_on_by_default_with_365_day_half_life(self) -> None:
        params = recency_params(Settings(), _analysis("current", None), now=NOW)
        assert params == RecencyParams(half_life_days=365.0, min_multiplier=0.8, reference=NOW)

    def test_disabled_gives_none(self) -> None:
        settings = Settings()
        settings.retrieval.recency_boost = False
        assert recency_params(settings, _analysis("current", None), now=NOW) is None

    def test_current_intent_measures_from_now(self) -> None:
        settings = Settings()
        settings.retrieval.recency_boost = True
        settings.retrieval.recency_half_life_days = 90
        settings.retrieval.recency_min_multiplier = 0.5
        params = recency_params(settings, _analysis("current", None), now=NOW)
        assert params == RecencyParams(half_life_days=90, min_multiplier=0.5, reference=NOW)

    def test_as_of_is_the_reference_and_plain_history_gets_no_boost(self) -> None:
        settings = Settings()
        settings.retrieval.recency_boost = True
        as_of = datetime(2025, 6, 1, tzinfo=UTC)
        params = recency_params(settings, _analysis("history", as_of), now=NOW)
        assert params is not None and params.reference == as_of
        assert recency_params(settings, _analysis("history", None), now=NOW) is None


class TestConfigValidation:
    def test_defaults(self) -> None:
        retrieval = RetrievalConfig()
        assert retrieval.graph_freshness is False and retrieval.recency_boost is True
        assert retrieval.graph_freshness_half_life_days == 365.0
        assert retrieval.recency_half_life_days == 365.0
        assert retrieval.recency_min_multiplier == 0.8

    @pytest.mark.parametrize(
        "field, value",
        [
            ("graph_freshness_half_life_days", 0),
            ("recency_half_life_days", -1),
            ("recency_min_multiplier", 0),
            ("recency_min_multiplier", 1.5),
        ],
    )
    def test_rejects_non_positive_half_lives_and_bad_floors(self, field: str, value: float) -> None:
        with pytest.raises(ValidationError):
            RetrievalConfig(**{field: value})
