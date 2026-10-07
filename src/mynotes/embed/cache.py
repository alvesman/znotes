"""Postgres-backed embedding cache (PLAN.md §5.1, §7.6).

Vectors live in ``embedding_cache`` as little-endian float16, keyed by
``(model, content_hash)``. Note: the column is named ``content_hash``
(migration 0001), but it stores :func:`embed_key` values — a blake3 digest of
the exact text sent to the model (``passage_prefix + embed_text``) — and not
the chunk's ``content_hash``. Two chunks that produce the same model input
therefore share one cache row.

:func:`ensure_embeddings` is the entry point used by the pipeline: it returns
one vector per input text in input order and calls the embedder only for
texts that are not cached yet, so unchanged text is never re-embedded.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from mynotes.db.models import EmbeddingCache, pack_vector, unpack_vector
from mynotes.embed.client import Embedder
from mynotes.scan.hashing import hash_text

logger = logging.getLogger(__name__)


def embed_key(embedder: Embedder, embed_text: str) -> str:
    """Return the cache key for one document text.

    The key covers exactly what is sent to the model: the embedder's
    ``passage_prefix`` (empty for embedders that do not define one, such as
    the test fake) followed by ``embed_text``.

    Args:
        embedder: The embedder whose document inputs are being cached.
        embed_text: The contextualized chunk text, without the prefix.

    Returns:
        The blake3 hex digest of the full model input.
    """
    prefix = getattr(embedder, "passage_prefix", "")
    return hash_text(prefix + embed_text)


def get_cached(session: Session, model: str, keys: Sequence[str]) -> dict[str, list[float]]:
    """Read the cached vectors for ``keys``.

    Args:
        session: An open session/transaction.
        model: The embedding model name (first cache key part).
        keys: :func:`embed_key` values; duplicates are harmless.

    Returns:
        A mapping from key to the unpacked vector, containing only the keys
        found in the cache.
    """
    unique = sorted(set(keys))
    if not unique:
        return {}
    rows = session.execute(
        select(EmbeddingCache.content_hash, EmbeddingCache.vector).where(
            EmbeddingCache.model == model,
            EmbeddingCache.content_hash.in_(unique),
        )
    ).all()
    return {content_hash: unpack_vector(vector) for content_hash, vector in rows}


def put_cached(
    session: Session, model: str, dim: int, items: Mapping[str, Sequence[float]]
) -> None:
    """Insert cache rows, keeping any vector that is already stored.

    Uses ``INSERT ... ON CONFLICT (model, content_hash) DO NOTHING`` so a
    concurrent worker that embedded the same input first wins; either vector
    is a valid embedding of the same model input.

    Args:
        session: An open session/transaction; rows are written but not
            committed.
        model: The embedding model name (first cache key part).
        dim: The expected vector dimension, stored alongside the vector.
        items: Mapping from :func:`embed_key` value to its vector.

    Raises:
        ValueError: If a vector's length differs from ``dim``.
    """
    values = []
    for key, vector in items.items():
        components = list(vector)
        if len(components) != dim:
            msg = f"vector {key!r} has {len(components)} components, expected {dim}"
            raise ValueError(msg)
        values.append(
            {
                "model": model,
                "content_hash": key,
                "dim": dim,
                "vector": pack_vector(components),
            }
        )
    if not values:
        return
    statement = pg_insert(EmbeddingCache).values(values)
    statement = statement.on_conflict_do_nothing(
        index_elements=[EmbeddingCache.model, EmbeddingCache.content_hash]
    )
    session.execute(statement)


def ensure_embeddings(
    session: Session, embedder: Embedder, texts: Sequence[str]
) -> list[list[float]]:
    """Return one vector per text, embedding and caching only the misses.

    The texts are keyed with :func:`embed_key`; hits are read from
    ``embedding_cache``, the unique missing texts are embedded with
    :meth:`Embedder.embed_documents` and stored via :func:`put_cached`.

    Args:
        session: An open session/transaction; cache writes commit with it.
        embedder: The embedder to use for cache misses.
        texts: The document texts, in the order the vectors are needed.

    Returns:
        One vector per input text, in input order.
    """
    inputs = list(texts)
    if not inputs:
        return []
    keys = [embed_key(embedder, text) for text in inputs]
    cached = get_cached(session, embedder.model, keys)
    missing: dict[str, str] = {}
    for key, text in zip(keys, inputs, strict=True):
        if key not in cached:
            missing.setdefault(key, text)
    fresh: dict[str, list[float]] = {}
    if missing:
        ordered_keys = list(missing)
        vectors = embedder.embed_documents([missing[key] for key in ordered_keys])
        fresh = dict(zip(ordered_keys, vectors, strict=True))
        put_cached(session, embedder.model, embedder.dim, fresh)
        logger.info("embedded %d new text(s) with %d cache hit(s)", len(fresh), len(cached))
    by_key = {**cached, **fresh}
    return [by_key[key] for key in keys]
