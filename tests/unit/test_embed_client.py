"""Unit tests for the embedding client (MockTransport, no network)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from fakes import mock_transport
from mynotes.config import EmbeddingConfig, RetryConfig, Settings
from mynotes.embed.client import EmbeddingClient
from mynotes.llm.errors import EndpointError
from mynotes.llm.limits import EndpointGuard

type Call = dict[str, Any]


def embeddings_response(vectors: list[list[float]]) -> httpx.Response:
    """A 200 response with ``data`` in the given order (index = position)."""
    return httpx.Response(
        200, json={"data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]}
    )


def make_embedder(
    settings: Settings,
    handler: Callable[[httpx.Request], httpx.Response],
    **cfg_updates: object,
) -> tuple[EmbeddingClient, list[Call]]:
    """Build a client on a MockTransport; ``calls`` records every request."""
    cfg: EmbeddingConfig = (
        settings.models.embedding.model_copy(update=cfg_updates)
        if cfg_updates
        else settings.models.embedding
    )
    guard = EndpointGuard("embedding", cfg.limits, RetryConfig(max_attempts=1))
    calls: list[Call] = []

    def record(request: httpx.Request) -> httpx.Response:
        calls.append({"request": request, "body": json.loads(request.content)})
        return handler(request)

    http = httpx.Client(transport=mock_transport({("POST", "/embeddings"): record}))
    return EmbeddingClient(cfg, guard=guard, http=http), calls


def test_format_query_builds_instruction(test_settings: Settings) -> None:
    """The default template sends the query as typed; an instruction template uses the task."""
    client, _ = make_embedder(test_settings, lambda request: embeddings_response([]))
    assert client.format_query("olá") == "olá"

    client, _ = make_embedder(
        test_settings,
        lambda request: embeddings_response([]),
        query_template="Instruct: {task}\nQuery: {text}",
    )
    assert client.format_query("olá") == (
        "Instruct: Given a question, retrieve passages that answer the question\nQuery: olá"
    )


def test_format_query_unknown_task_raises_clear_keyerror(test_settings: Settings) -> None:
    """An unknown task raises KeyError naming the configured tasks."""
    client, _ = make_embedder(test_settings, lambda request: embeddings_response([]))

    with pytest.raises(KeyError, match=r"unknown embedding task 'nope'.*retrieval"):
        client.format_query("olá", task="nope")


def test_embed_documents_batches_in_order_without_prefix(test_settings: Settings) -> None:
    """20 texts with batch=8: 3 requests, input order kept, no passage prefix."""
    texts = [f"documento número {i}" for i in range(20)]
    dim = test_settings.models.embedding.dim

    def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        # Return each batch with its indices reversed: the client must
        # reorder by data[].index, not trust the response order.
        data = []
        for local_index in reversed(range(len(inputs))):
            text_number = int(inputs[local_index].split()[-1])
            data.append(
                {
                    "index": local_index,
                    "embedding": [float(text_number)] + [0.0] * (dim - 1),
                }
            )
        return httpx.Response(200, json={"data": data})

    client, calls = make_embedder(test_settings, handler, normalize="never")
    vectors = client.embed_documents(texts)

    assert len(calls) == 3
    assert sorted(len(call["body"]["input"]) for call in calls) == [4, 8, 8]
    sent_inputs = [text for call in calls for text in call["body"]["input"]]
    assert sorted(sent_inputs) == sorted(texts), "bodies must contain the plain texts"
    assert [vector[0] for vector in vectors] == [float(i) for i in range(20)]


def test_dimension_mismatch_raises_endpoint_error(test_settings: Settings) -> None:
    """A vector with the wrong length fails with the exact spec message."""
    client, calls = make_embedder(test_settings, lambda request: embeddings_response([[0.5, 0.5]]))

    with pytest.raises(EndpointError) as exc_info:
        client.embed_documents(["olá"])

    assert exc_info.value.endpoint == "embedding"
    assert exc_info.value.status is None
    assert exc_info.value.message == "dimension mismatch: got 2, expected 8"
    assert len(calls) == 1


def test_missing_vectors_in_response_raise_endpoint_error(test_settings: Settings) -> None:
    """A response with fewer vectors than inputs is rejected, not misaligned."""
    dim = test_settings.models.embedding.dim
    client, _ = make_embedder(
        test_settings,
        lambda request: embeddings_response([[0.0] * dim for _ in range(2)]),
    )

    with pytest.raises(EndpointError, match="vector count mismatch: got 2, expected 3"):
        client.embed_documents(["a", "b", "c"])


def test_normalize_auto_normalizes_non_unit_vectors(test_settings: Settings) -> None:
    """auto: a norm-2 vector is normalized, a unit vector is left untouched."""
    unit = [0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    doubled = [value * 2 for value in unit]
    client, _ = make_embedder(test_settings, lambda request: embeddings_response([doubled, unit]))

    vectors = client.embed_documents(["a", "b"])

    assert vectors[0] == pytest.approx(unit)
    assert vectors[1] == unit, "a unit vector must be returned unchanged"


def test_normalize_always_and_never(test_settings: Settings) -> None:
    """always normalizes everything; never returns the server vector as-is."""
    doubled_unit = [1.2, 1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    always_client, _ = make_embedder(
        test_settings,
        lambda request: embeddings_response([doubled_unit]),
        normalize="always",
    )
    assert always_client.embed_documents(["a"])[0] == pytest.approx([0.6, 0.8, 0, 0, 0, 0, 0, 0])

    never_client, _ = make_embedder(
        test_settings,
        lambda request: embeddings_response([doubled_unit]),
        normalize="never",
    )
    assert never_client.embed_documents(["a"])[0] == doubled_unit


def test_authorization_header_sent_only_with_api_key(test_settings: Settings) -> None:
    """The Bearer header appears with a key and disappears without one."""

    def handler(request: httpx.Request) -> httpx.Response:
        return embeddings_response([[0.0] * test_settings.models.embedding.dim])

    keyed_client, keyed_calls = make_embedder(
        test_settings, handler, api_key=SecretStr("unit-test-key")
    )
    keyed_client.embed_query("pergunta")

    assert keyed_calls[0]["request"].headers["Authorization"] == "Bearer unit-test-key"

    anonymous_client, anonymous_calls = make_embedder(test_settings, handler, api_key=None)
    anonymous_client.embed_query("pergunta")

    assert "Authorization" not in anonymous_calls[0]["request"].headers


def test_empty_and_whitespace_texts_are_rejected(test_settings: Settings) -> None:
    """Empty or whitespace-only inputs raise ValueError before any request."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request must be made")

    client, calls = make_embedder(test_settings, handler)

    with pytest.raises(ValueError):
        client.embed_query("")
    with pytest.raises(ValueError):
        client.embed_query("   \n\t")
    with pytest.raises(ValueError):
        client.embed_documents(["ok", "  "])
    assert calls == []
    assert client.embed_documents([]) == []


def test_embed_query_formats_then_embeds(test_settings: Settings) -> None:
    """The query body carries the formatted instruction, not the raw text."""
    client, calls = make_embedder(
        test_settings,
        lambda request: embeddings_response([[0.0] * test_settings.models.embedding.dim]),
        query_template="Instruct: {task}\nQuery: {text}",
    )

    client.embed_query("olá")

    assert calls[0]["body"]["input"] == [
        "Instruct: Given a question, retrieve passages that answer the question\nQuery: olá"
    ]


def test_passage_prefix_is_applied_to_documents(test_settings: Settings) -> None:
    """A configured prefix is prepended to every document, never to queries."""
    dim = test_settings.models.embedding.dim

    def echo_handler(request: httpx.Request) -> httpx.Response:
        count = len(json.loads(request.content)["input"])
        return embeddings_response([[0.0] * dim for _ in range(count)])

    client, calls = make_embedder(test_settings, echo_handler, passage_prefix="passage: ")

    client.embed_documents(["primeiro", "segundo"])
    client.embed_query("pergunta")

    assert calls[0]["body"]["input"] == ["passage: primeiro", "passage: segundo"]
    assert "passage: " not in calls[1]["body"]["input"][0]
