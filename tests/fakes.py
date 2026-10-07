"""Deterministic, network-free fakes for the embedder, reranker and chat (tasks 06-07).

Unit tests combine these with ``httpx.MockTransport`` (see
:func:`mock_transport`) so no test touches the network or a real model
endpoint.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterator, Sequence
from typing import Literal

import httpx

from mynotes.llm.chat import ChatMessage, ChatResult
from mynotes.retrieval.rerank_client import RerankResult
from mynotes.text.normalize import normalize_key

type Handler = Callable[[httpx.Request], httpx.Response]


def _unit_vector(vector: list[float]) -> list[float]:
    """L2-normalize; an all-zero vector becomes a unit vector on index 0."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        unit = [0.0] * len(vector)
        if unit:
            unit[0] = 1.0
        return unit
    return [value / norm for value in vector]


class FakeEmbedder:
    """Hashing-trick embedder: same words, similar vectors, no network.

    Tokens come from :func:`mynotes.text.normalize.normalize_key`; each token
    adds 1.0 at ``blake2b(token) % dim`` and the result is L2-normalized.
    Every call is recorded in ``calls`` so tests can assert what was (not)
    re-embedded.
    """

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.model = "fake-embedder"
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        self.calls.append(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str, task: str = "retrieval") -> list[float]:
        self.calls.append([text])
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in normalize_key(text).split():
            digest = hashlib.blake2b(token.encode("utf-8")).hexdigest()
            vector[int(digest, 16) % self.dim] += 1.0
        return _unit_vector(vector)


class FakeReranker:
    """Jaccard-overlap reranker: deterministic scores, no network.

    Score = |query tokens ∩ document tokens| / |union| over
    :func:`normalize_key` token sets, sorted descending, ties by original
    index.
    """

    def rerank(
        self, query: str, documents: Sequence[str], top_n: int | None = None
    ) -> list[RerankResult]:
        docs = list(documents)
        if not docs:
            return []
        query_tokens = set(normalize_key(query).split())
        scored: list[RerankResult] = []
        for index, doc in enumerate(docs):
            doc_tokens = set(normalize_key(doc).split())
            union = query_tokens | doc_tokens
            score = (len(query_tokens & doc_tokens) / len(union)) if union else 0.0
            scored.append(RerankResult(index=index, score=score))
        scored.sort(key=lambda result: (-result.score, result.index))
        if top_n is not None and top_n > 0:
            scored = scored[:top_n]
        return scored


class FakeChat:
    """Scripted chat model: canned answers in order, every call recorded.

    ``responses`` is either a list (one answer per call, in order) or a
    callable receiving the message list and returning the answer text.
    :meth:`stream` yields the same answer split into three pieces.
    ``calls`` holds the message list of every complete/stream call and
    ``kwargs`` the matching keyword arguments (priority, max_tokens,
    temperature, response_format).
    """

    def __init__(self, responses: list[str] | Callable[[Sequence[ChatMessage]], str]) -> None:
        self._responses = responses
        self.calls: list[list[ChatMessage]] = []
        self.kwargs: list[dict[str, object]] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        priority: Literal["interactive", "background"] = "interactive",
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict | None = None,
    ) -> ChatResult:
        self._record(messages, priority, max_tokens, temperature, response_format)
        return ChatResult(text=self._next_response(), finish_reason="stop", usage={})

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        priority: Literal["interactive", "background"] = "interactive",
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        self._record(messages, priority, max_tokens, None, None)
        text = self._next_response()
        third = max(1, len(text) // 3)
        for piece in (text[:third], text[third : 2 * third], text[2 * third :]):
            if piece:
                yield piece

    def _record(
        self,
        messages: Sequence[ChatMessage],
        priority: str,
        max_tokens: int | None,
        temperature: float | None,
        response_format: dict | None,
    ) -> None:
        self.calls.append(list(messages))
        self.kwargs.append(
            {
                "priority": priority,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "response_format": response_format,
            }
        )

    def _next_response(self) -> str:
        if callable(self._responses):
            return self._responses(self.calls[-1])
        return self._responses[len(self.calls) - 1]


def mock_transport(handler_map: dict[tuple[str, str], Handler]) -> httpx.MockTransport:
    """Route ``(METHOD, path-suffix)`` to handlers; unknown routes return 404.

    Args:
        handler_map: Maps e.g. ``("POST", "/embeddings")`` to a handler that
            receives the :class:`httpx.Request` and returns the response.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        for (method, suffix), respond in handler_map.items():
            if request.method == method and request.url.path.endswith(suffix):
                return respond(request)
        return httpx.Response(404, text=f"no mock route for {request.method} {request.url.path}")

    return httpx.MockTransport(handler)
