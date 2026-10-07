"""Rerank client for the Cohere-style reranker endpoint (``/v1/rerank``).

Sends ``{"model", "query", "documents", "top_n"}`` to ``cfg.url`` and reads
``results[].index`` / ``results[].relevance_score``. Larger document lists are
split into consecutive batches sent in parallel through the shared
:class:`~mynotes.llm.limits.EndpointGuard`; batch-local indices are remapped to
the original positions before merging.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import httpx
from pydantic import SecretStr

from mynotes.config import ModelsNotConfigured, RerankerConfig, get_settings
from mynotes.llm.errors import EndpointError
from mynotes.llm.limits import EndpointGuard, get_guard


@dataclass(frozen=True)
class RerankResult:
    """One reranked document: position in the input list and relevance score."""

    index: int
    score: float


class Reranker(Protocol):
    """Anything that can rerank documents against a query."""

    def rerank(
        self, query: str, documents: Sequence[str], top_n: int | None = None
    ) -> list[RerankResult]: ...


def _auth_headers(api_key: SecretStr | None) -> dict[str, str]:
    """Build the request headers; ``Authorization`` only when a key is set."""
    if api_key is None:
        return {}
    key = api_key.get_secret_value()
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


class RerankClient:
    """Client for one reranker endpoint; implements :class:`Reranker`."""

    def __init__(
        self,
        cfg: RerankerConfig,
        guard: EndpointGuard | None = None,
        http: httpx.Client | None = None,
    ) -> None:
        """Build a client for one endpoint configuration.

        Args:
            cfg: Reranker endpoint, model and limits.
            guard: Shared endpoint guard; defaults to ``get_guard("reranker")``.
            http: HTTP client to use; a plain ``httpx.Client`` is created when None.
        """
        self.cfg = cfg
        self.guard = guard if guard is not None else get_guard("reranker")
        self._owns_http = http is None
        self._http = http if http is not None else httpx.Client()

    def close(self) -> None:
        """Close the owned HTTP client; an injected client is left open."""
        if self._owns_http:
            self._http.close()

    def rerank(
        self, query: str, documents: Sequence[str], top_n: int | None = None
    ) -> list[RerankResult]:
        """Score documents against ``query`` and return the best ones.

        Args:
            query: The query text.
            documents: Candidate documents, in their original order.
            top_n: How many results to keep; defaults to ``cfg.top_n``, where
                ``0`` (or any non-positive value) means ``len(documents)``.

        Returns:
            Results sorted by score descending (ties by original index), cut
            to ``top_n``. Empty input returns ``[]`` without a request.

        Raises:
            EndpointError: On HTTP failure or a malformed response.
        """
        docs = list(documents)
        if not docs:
            return []
        limit = top_n if top_n is not None else self.cfg.top_n
        if limit <= 0:
            limit = len(docs)
        max_docs = self.cfg.max_documents_per_request
        jobs = [
            (offset, docs[offset : offset + max_docs]) for offset in range(0, len(docs), max_docs)
        ]
        workers = max(1, min(self.cfg.limits.concurrency, len(jobs)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            batch_results = list(pool.map(lambda job: self._rerank_batch(query, *job, limit), jobs))
        merged = [result for batch in batch_results for result in batch]
        merged.sort(key=lambda result: (-result.score, result.index))
        return merged[:limit]

    def _rerank_batch(
        self, query: str, offset: int, batch: list[str], limit: int
    ) -> list[RerankResult]:
        """Send one request for a batch and remap its indices to global positions."""
        response = self.guard.request(
            self._http,
            "POST",
            self.cfg.url,
            json={
                "model": self.cfg.model,
                "query": query,
                "documents": batch,
                "top_n": min(limit, len(batch)),
            },
            headers=_auth_headers(self.cfg.api_key),
        )
        try:
            items = response.json()["results"]
        except (ValueError, KeyError, TypeError) as exc:
            msg = f"malformed response body: {type(exc).__name__}: {exc}"
            raise EndpointError("reranker", response.status_code, msg) from exc
        results: list[RerankResult] = []
        for item in items:
            try:
                index = int(item["index"]) + offset
                score = float(item["relevance_score"])
            except (KeyError, TypeError, ValueError) as exc:
                msg = f"malformed response item: {type(exc).__name__}: {exc}"
                raise EndpointError("reranker", response.status_code, msg) from exc
            results.append(RerankResult(index=index, score=score))
        return results


@lru_cache(maxsize=1)
def reranker_configured() -> bool:
    """Whether a reranker endpoint is set; without one, results keep their fused order."""
    return bool(get_settings().models.reranker.url)


def get_reranker() -> Reranker:
    """Return the process-wide reranker built from :func:`get_settings`.

    Raises:
        ModelsNotConfigured: When no reranker url (or model) is set; callers check
            :func:`reranker_configured` first.
    """
    cfg = get_settings().models.reranker
    missing = [f"models.reranker.{k}" for k, v in (("url", cfg.url), ("model", cfg.model)) if not v]
    if missing:
        raise ModelsNotConfigured(missing)
    return RerankClient(cfg)
