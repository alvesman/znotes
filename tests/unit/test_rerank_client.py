"""Unit tests for the rerank client and the deterministic fakes."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from fakes import FakeEmbedder, FakeReranker, mock_transport
from mynotes.config import RerankerConfig, RetryConfig, Settings
from mynotes.llm.errors import EndpointError
from mynotes.llm.limits import EndpointGuard
from mynotes.retrieval.rerank_client import RerankClient, RerankResult

type Call = dict[str, Any]


def rerank_response(results: list[dict[str, float | int]]) -> httpx.Response:
    """A 200 response with a ``results`` list of {index, relevance_score}."""
    return httpx.Response(200, json={"results": results})


def make_reranker(
    settings: Settings,
    handler: Callable[[httpx.Request], httpx.Response],
    **cfg_updates: object,
) -> tuple[RerankClient, list[Call]]:
    """Build a client on a MockTransport; ``calls`` records every request."""
    cfg: RerankerConfig = (
        settings.models.reranker.model_copy(update=cfg_updates)
        if cfg_updates
        else settings.models.reranker
    )
    guard = EndpointGuard("reranker", cfg.limits, RetryConfig(max_attempts=1))
    calls: list[Call] = []

    def record(request: httpx.Request) -> httpx.Response:
        calls.append({"request": request, "body": json.loads(request.content)})
        return handler(request)

    http = httpx.Client(transport=mock_transport({("POST", "/rerank"): record}))
    return RerankClient(cfg, guard=guard, http=http), calls


def score_of(doc_index: int) -> float:
    """Deterministic pseudo-shuffled score so batches interleave when merged."""
    return ((doc_index * 37) % 97) / 97


def scoring_handler(request: httpx.Request) -> httpx.Response:
    """Score each document by its number ('doc N'), returned in reverse order."""
    documents = json.loads(request.content)["documents"]
    results = [
        {"index": local, "relevance_score": score_of(int(doc.split()[-1]))}
        for local, doc in enumerate(documents)
    ]
    return rerank_response(list(reversed(results)))


def test_rerank_parses_and_sorts_results(test_settings: Settings) -> None:
    """Results come back sorted by score regardless of the server order."""
    client, calls = make_reranker(
        test_settings,
        lambda request: rerank_response(
            [
                {"index": 0, "relevance_score": 0.1},
                {"index": 1, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.5},
            ]
        ),
    )

    results = client.rerank("pergunta", ["d0", "d1", "d2"])

    assert results == [
        RerankResult(index=1, score=0.9),
        RerankResult(index=2, score=0.5),
        RerankResult(index=0, score=0.1),
    ]
    body = calls[0]["body"]
    assert body["query"] == "pergunta"
    assert body["documents"] == ["d0", "d1", "d2"]
    assert body["top_n"] == 3, "cfg.top_n=0 must resolve to len(documents)"
    assert isinstance(body["model"], str) and body["model"]


def test_rerank_batches_ninety_docs_and_respects_top_n(test_settings: Settings) -> None:
    """90 docs with max 40/request: 3 calls, global indices, top 5 kept."""
    documents = [f"doc {i}" for i in range(90)]
    client, calls = make_reranker(test_settings, scoring_handler, max_documents_per_request=40)

    results = client.rerank("pergunta", documents, top_n=5)

    assert len(calls) == 3
    assert [len(call["body"]["documents"]) for call in calls] == [40, 40, 10]
    assert [call["body"]["top_n"] for call in calls] == [5, 5, 5]
    expected = sorted(range(90), key=lambda i: (-score_of(i), i))[:5]
    assert [result.index for result in results] == expected
    assert all(result.score == pytest.approx(score_of(result.index)) for result in results)


def test_rerank_cfg_top_n_zero_returns_all_sorted(test_settings: Settings) -> None:
    """With no top_n argument and cfg.top_n=0 every document is returned."""
    documents = [f"doc {i}" for i in range(90)]
    client, _ = make_reranker(test_settings, scoring_handler, max_documents_per_request=40)

    results = client.rerank("pergunta", documents)

    assert len(results) == 90
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)
    assert sorted(result.index for result in results) == list(range(90))


def test_rerank_empty_documents_returns_empty_list(test_settings: Settings) -> None:
    """No documents: [] without touching the endpoint."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request must be made")

    client, calls = make_reranker(test_settings, handler)

    assert client.rerank("pergunta", []) == []
    assert calls == []


def test_rerank_malformed_response_raises_endpoint_error(test_settings: Settings) -> None:
    """A body without results[].index/relevance_score fails clearly."""
    client, _ = make_reranker(
        test_settings, lambda request: httpx.Response(200, json={"results": [{"nope": 1}]})
    )

    with pytest.raises(EndpointError) as exc_info:
        client.rerank("pergunta", ["d0"])

    assert exc_info.value.endpoint == "reranker"
    assert "malformed response" in exc_info.value.message


def test_fake_embedder_is_deterministic_and_normalized() -> None:
    """Identical texts give identical unit vectors; shared words score higher."""
    embedder = FakeEmbedder(dim=8)
    question = embedder.embed_query("Como limpar o filtro do exaustor")
    again = embedder.embed_query("Como limpar o filtro do exaustor", task="tag_lookup")

    assert question == again, "embed_query must ignore the task"
    assert len(question) == 8
    assert math.isclose(math.sqrt(sum(v * v for v in question)), 1.0, rel_tol=1e-9)

    def cosine(u: list[float], v: list[float]) -> float:
        return sum(a * b for a, b in zip(u, v, strict=True))

    related = embedder.embed_query("limpar o filtro")
    unrelated = embedder.embed_query("orçamento aprovado pela direção")

    assert cosine(question, related) > cosine(question, unrelated)


def test_fake_embedder_records_calls() -> None:
    """Every embed call is recorded so tests can assert re-embedding."""
    embedder = FakeEmbedder()

    embedder.embed_documents(["a", "b"])
    embedder.embed_query("c", task="tag_lookup")

    assert embedder.calls == [["a", "b"], ["c"]]


def test_fake_embedder_empty_text_becomes_unit_vector_on_index_0() -> None:
    """An all-zero token count falls back to a unit vector on index 0."""
    vector = FakeEmbedder(dim=8).embed_query("   ")

    assert vector == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_fake_reranker_jaccard_scores_and_ties() -> None:
    """Jaccard overlap ordering, ties broken by original index."""
    reranker = FakeReranker()

    results = reranker.rerank(
        "limpar o filtro",
        ["orçamento aprovado", "Limpar o filtro com água", "filtro entupido"],
    )

    assert [result.index for result in results] == [1, 2, 0]
    assert results[0].score == pytest.approx(0.6)
    assert results[1].score == pytest.approx(0.25)
    assert results[2].score == pytest.approx(0.0)

    tied = reranker.rerank("filtro", ["sem relação", "também sem"])
    assert [result.index for result in tied] == [0, 1], "ties by original index"
    assert reranker.rerank("q", [], top_n=3) == []
    assert len(reranker.rerank("q", ["a", "b", "c"], top_n=2)) == 2
