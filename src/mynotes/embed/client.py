"""Embedding client for the OpenAI-compatible ``/embeddings`` endpoint.

Queries are formatted with the configured instruction template (``Instruct:
{task}\\nQuery: {text}`` for ``nvidia/llama-embed-nemotron-8b``); documents are
sent as ``passage_prefix + text`` (empty by default). Every HTTP call goes
through the shared :class:`~mynotes.llm.limits.EndpointGuard` from task 05,
which bounds concurrency, retries transient failures and records stats.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Protocol

import httpx
from pydantic import SecretStr

from mynotes.config import EmbeddingConfig, ModelsNotConfigured, get_settings
from mynotes.llm.errors import EndpointError
from mynotes.llm.limits import EndpointGuard, get_guard

_NORM_TOLERANCE = 1e-3


class Embedder(Protocol):
    """Anything that can embed queries and documents."""

    dim: int
    model: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str, task: str = "retrieval") -> list[float]: ...


def _embeddings_url(base_url: str) -> str:
    """Join ``base_url`` (with or without a trailing slash) with ``/embeddings``."""
    return f"{base_url.rstrip('/')}/embeddings"


def _auth_headers(api_key: SecretStr | None) -> dict[str, str]:
    """Build the request headers; ``Authorization`` only when a key is set."""
    if api_key is None:
        return {}
    key = api_key.get_secret_value()
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


def _l2_normalize(vector: list[float]) -> list[float]:
    """Return the unit vector in the same direction (zero vectors pass through)."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _require_text(text: str) -> None:
    """Reject empty or whitespace-only inputs before anything is sent."""
    if not text or not text.strip():
        raise ValueError("embedding input must be a non-empty, non-whitespace string")


class EmbeddingClient:
    """Client for one embedding endpoint; implements :class:`Embedder`.

    Documents are batched (``cfg.batch`` inputs per request) and batches are
    sent in parallel through the endpoint guard; results keep the input order.
    Responses are ordered by ``data[].index``, validated against ``cfg.dim``
    and L2-normalized according to ``cfg.normalize``.
    """

    def __init__(
        self,
        cfg: EmbeddingConfig,
        guard: EndpointGuard | None = None,
        http: httpx.Client | None = None,
    ) -> None:
        """Build a client for one endpoint configuration.

        Args:
            cfg: Embedding endpoint, model, query format and limits.
            guard: Shared endpoint guard; defaults to ``get_guard("embedding")``.
            http: HTTP client to use; a plain ``httpx.Client`` is created when None.
        """
        missing = [
            f"models.embedding.{key}"
            for key, value in (("base_url", cfg.base_url), ("model", cfg.model), ("dim", cfg.dim))
            if not value
        ]
        if missing:
            raise ModelsNotConfigured(missing)
        self.cfg = cfg
        self.dim = cfg.dim
        self.model = cfg.model
        self.guard = guard if guard is not None else get_guard("embedding")
        self._owns_http = http is None
        self._http = http if http is not None else httpx.Client()

    @property
    def passage_prefix(self) -> str:
        """The prefix prepended to every document text before it is sent."""
        return self.cfg.passage_prefix

    def close(self) -> None:
        """Close the owned HTTP client; an injected client is left open."""
        if self._owns_http:
            self._http.close()

    def format_query(self, text: str, task: str = "retrieval") -> str:
        """Format ``text`` with the instruction template configured for ``task``.

        Args:
            text: The raw query text.
            task: Key into ``cfg.tasks``; defaults to the retrieval instruction.

        Returns:
            The formatted query string (e.g. ``"Instruct: ...\\nQuery: ..."``).

        Raises:
            KeyError: If ``task`` is not configured, with the known task names.
        """
        try:
            instruction = self.cfg.tasks[task]
        except KeyError:
            known = ", ".join(sorted(self.cfg.tasks))
            msg = f"unknown embedding task {task!r}; configured tasks: {known}"
            raise KeyError(msg) from None
        return self.cfg.query_template.format(task=instruction, text=text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents, preserving the input order.

        Args:
            texts: Document texts; each is sent as ``passage_prefix + text``.

        Returns:
            One vector per input, in input order.

        Raises:
            ValueError: If any text is empty or whitespace-only.
            EndpointError: On HTTP failure or a dimension mismatch.
        """
        batch_texts = list(texts)
        for text in batch_texts:
            _require_text(text)
        if not batch_texts:
            return []
        inputs = [self.cfg.passage_prefix + text for text in batch_texts]
        batches = [inputs[i : i + self.cfg.batch] for i in range(0, len(inputs), self.cfg.batch)]
        workers = max(1, min(self.cfg.limits.concurrency, len(batches)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            batch_vectors = list(pool.map(self._embed_batch, batches))
        return [vector for batch in batch_vectors for vector in batch]

    def embed_query(self, text: str, task: str = "retrieval") -> list[float]:
        """Embed one query formatted with the instruction for ``task``.

        Args:
            text: The raw query text.
            task: Key into ``cfg.tasks``.

        Returns:
            The query vector.

        Raises:
            ValueError: If ``text`` is empty or whitespace-only.
            KeyError: If ``task`` is not configured.
            EndpointError: On HTTP failure or a dimension mismatch.
        """
        _require_text(text)
        return self._embed_batch([self.format_query(text, task)])[0]

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """Send one request for ``batch`` and return its parsed vectors."""
        response = self.guard.request(
            self._http,
            "POST",
            _embeddings_url(self.cfg.base_url),
            json={"model": self.cfg.model, "input": batch},
            headers=_auth_headers(self.cfg.api_key),
        )
        return self._parse_vectors(response, expected=len(batch))

    def _parse_vectors(self, response: httpx.Response, expected: int) -> list[list[float]]:
        """Extract, order by index, dimension-check and normalize the vectors."""
        try:
            items = response.json()["data"]
        except (ValueError, KeyError, TypeError) as exc:
            msg = f"malformed response body: {type(exc).__name__}: {exc}"
            raise EndpointError("embedding", response.status_code, msg) from exc
        if not isinstance(items, list):
            msg = "malformed response body: 'data' is not a list"
            raise EndpointError("embedding", response.status_code, msg)
        positioned: list[tuple[int, list[float]]] = []
        for item in items:
            try:
                index = int(item["index"])
                vector = item["embedding"]
            except (KeyError, TypeError, ValueError) as exc:
                msg = f"malformed response item: {type(exc).__name__}: {exc}"
                raise EndpointError("embedding", response.status_code, msg) from exc
            positioned.append((index, vector))
        positioned.sort(key=lambda pair: pair[0])
        if len(positioned) != expected:
            msg = f"vector count mismatch: got {len(positioned)}, expected {expected}"
            raise EndpointError("embedding", response.status_code, msg)
        return [self._normalize(self._check_dim(vector)) for _, vector in positioned]

    def _check_dim(self, vector: list[float]) -> list[float]:
        """Require the vector length to equal ``cfg.dim``."""
        if len(vector) != self.cfg.dim:
            msg = f"dimension mismatch: got {len(vector)}, expected {self.cfg.dim}"
            raise EndpointError("embedding", None, msg)
        return vector

    def _normalize(self, vector: list[float]) -> list[float]:
        """Apply ``cfg.normalize``: ``auto`` only fixes norms off by > 1e-3."""
        if self.cfg.normalize == "never":
            return vector
        if self.cfg.normalize == "always":
            return _l2_normalize(vector)
        norm = math.sqrt(sum(value * value for value in vector))
        if abs(norm - 1.0) > _NORM_TOLERANCE:
            return _l2_normalize(vector)
        return vector


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Return the process-wide embedder built from :func:`get_settings`."""
    return EmbeddingClient(get_settings().models.embedding)
