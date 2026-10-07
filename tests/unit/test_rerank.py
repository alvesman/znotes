"""Unit tests for the cross-encoder rerank step (task 28, PLAN.md §9 step 5)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from fakes import FakeReranker
from mynotes.llm.errors import EndpointUnavailable
from mynotes.retrieval.rerank import rerank_document_text, rerank_hits
from mynotes.retrieval.rerank_client import RerankResult
from mynotes.retrieval.types import SearchHit


def _hit(
    chunk_id: str,
    rank: int,
    text: str,
    *,
    title: str | None = None,
    heading_path: str = "",
    doc_status: str = "current",
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id=1,
        version_id=1,
        rank=rank,
        score=0.01,
        text=text,
        title=title,
        section=None,
        heading_path=heading_path,
        path=f"notes/{chunk_id}.md",
        page=None,
        slide=None,
        doc_date=datetime(2026, 1, 1, tzinfo=UTC),
        doc_status=doc_status,
        is_latest=True,
        is_table=False,
        format="md",
        language=None,
        tags=[],
    )


class RecordingReranker(FakeReranker):
    """A :class:`FakeReranker` that records the documents it receives."""

    def __init__(self) -> None:
        super().__init__()
        self.documents: list[list[str]] = []

    def rerank(
        self, query: str, documents: Sequence[str], top_n: int | None = None
    ) -> list[RerankResult]:
        self.documents.append(list(documents))
        return super().rerank(query, documents, top_n)


def test_title_and_heading_path_are_prepended_when_missing() -> None:
    hit = _hit(
        "1",
        1,
        "o termostato está avariado",
        title="Frigorífico",
        heading_path="Cozinha > Manutenção",
    )

    assert (
        rerank_document_text(hit) == "Frigorífico\nCozinha > Manutenção\no termostato está avariado"
    )


def test_title_is_not_duplicated_when_already_in_the_text() -> None:
    hit = _hit("1", 1, "Frigorífico\no termostato está avariado", title="Frigorífico")

    assert rerank_document_text(hit) == "Frigorífico\no termostato está avariado"


def test_containment_check_is_case_insensitive() -> None:
    hit = _hit("1", 1, "the FRIGORÍFICO manual", title="Frigorífico")

    assert rerank_document_text(hit) == "the FRIGORÍFICO manual"


def test_only_the_top_hits_are_reranked_and_order_follows_scores() -> None:
    reranker = RecordingReranker()
    hits = [
        _hit("a", 1, "totally unrelated words here"),
        _hit("b", 2, "frigorifico termostato avariado revisao"),
        _hit("c", 3, "another unrelated paragraph indeed"),
    ]

    out = rerank_hits("frigorifico termostato", hits, reranker, top=2)

    assert [hit.chunk_id for hit in out] == ["b", "a"]
    assert [hit.rank for hit in out] == [1, 2]
    assert len(reranker.documents) == 1
    assert len(reranker.documents[0]) == 2
    assert out[0].rerank_score is not None and out[0].rerank_score > 0
    assert out[1].rerank_score == 0


def test_archived_document_gets_the_ninety_percent_prior() -> None:
    hits = [
        _hit("b", 2, "frigorifico", doc_status="archived"),
        _hit("a", 1, "frigorifico"),
    ]

    out = rerank_hits("frigorifico", hits, FakeReranker(), top=5)

    assert [hit.chunk_id for hit in out] == ["a", "b"]
    assert out[0].rerank_score == out[1].rerank_score
    assert [hit.rank for hit in out] == [1, 2]


def test_reranker_failure_degrades_to_the_input_order() -> None:
    class ExplodingReranker:
        def rerank(
            self, query: str, documents: Sequence[str], top_n: int | None = None
        ) -> list[RerankResult]:
            raise EndpointUnavailable("reranker")

    hits = [_hit("a", 1, "one"), _hit("b", 2, "two"), _hit("c", 3, "three")]

    out = rerank_hits("q", hits, ExplodingReranker(), top=2)

    assert [hit.chunk_id for hit in out] == ["a", "b", "c"]
    assert [hit.rank for hit in out] == [1, 2, 3]
    assert all(hit.rerank_score is None for hit in out)


def test_empty_input_returns_empty() -> None:
    assert rerank_hits("q", [], FakeReranker(), top=5) == []
