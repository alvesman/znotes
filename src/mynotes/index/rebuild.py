"""Zero-downtime rebuild of the chunks and entities indices (PLAN.md §6, §7.6).

:func:`rebuild_chunks_index` and :func:`rebuild_entities_index` build a
brand-new physical index (``_v2``, then ``_v3``, ...) straight from Postgres
plus the ``embedding_cache``, verify the document count against Postgres, and
only then swap the alias atomically. The old physical index is kept for
inspection; deleting it is a manual step.

The worker may keep running during a rebuild: outbox rows written while the new
index is being filled are applied later, to whatever physical index the alias
points at after the swap, which is correct because every upsert re-renders the
chunk from current Postgres state. The verification count is the safety net:
any mismatch leaves the alias untouched on the old index.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk as bulk_helper
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from mynotes.config import Settings, get_settings
from mynotes.db.models import Chunk, EmbeddingCache, KgEntity
from mynotes.db.session import session_scope
from mynotes.embed.cache import ensure_embeddings
from mynotes.embed.client import Embedder, get_embedder
from mynotes.graph.resolve import entity_text
from mynotes.index.documents import build_chunk_docs, build_entity_docs
from mynotes.index.mappings import chunks_index_body, entities_index_body
from mynotes.index.opensearch import (
    create_index,
    current_physical_index,
    get_client,
    physical_name,
    swap_alias,
)
from mynotes.index.reconcile import expected_chunk_ids

logger = logging.getLogger(__name__)

_REEMBED_PROGRESS_STEP = 500


def rebuild_chunks_index(*, reembed: bool = False, batch: int = 200) -> dict[str, Any]:
    """Rebuild the chunks index from Postgres and swap the alias when verified.

    Creates ``<alias>_v<N+1>`` with the current mapping, streams every indexable
    chunk into it in batches (chunks without a cached embedding are embedded on
    the fly with :func:`~mynotes.embed.cache.ensure_embeddings`), refreshes and
    counts the new index, and swaps the alias only when the count matches
    Postgres. On a mismatch a ``RuntimeError`` is raised and the alias stays on
    the old index; the new index is kept for inspection.

    Args:
        reembed: When True, first delete the ``embedding_cache`` rows of the
            current model and re-embed every indexable chunk in batches (slow;
            progress is logged every 500 chunks). Use after a model change.
        batch: Chunks per build/bulk batch.

    Returns:
        ``{"old": ..., "new": ..., "count": ..., "reembedded": ...}``.
    """
    settings = get_settings()
    client = get_client()
    alias = settings.opensearch.chunks_alias
    old = current_physical_index(client, alias)
    new = physical_name(alias, _next_version(client, alias, old))
    logger.info("rebuild: creating '%s' (alias '%s' currently on '%s')", new, alias, old)
    body = chunks_index_body(settings.models.embedding.dim, settings.text_matching.folding)
    create_index(client, new, body)
    embedder = get_embedder()
    if reembed:
        _reembed_all(settings, embedder, batch)
    expected = _rebuild_into(settings, client, embedder, new, batch)
    count = _count_after_refresh(client, new)
    if count != expected:
        raise RuntimeError(
            f"rebuild of '{new}' incomplete: {count}/{expected} documents; "
            f"alias '{alias}' left on '{old}'"
        )
    swap_alias(client, alias, new)
    if old is not None and old != new:
        logger.info(
            "rebuild: old index '%s' kept; drop it manually "
            "('mynotes index drop-old' is not implemented)",
            old,
        )
    logger.info("rebuild: '%s' now serves '%s' with %d document(s)", new, alias, count)
    return {"old": old, "new": new, "count": count, "reembedded": reembed}


def rebuild_entities_index(*, batch: int = 200) -> dict[str, Any]:
    """Rebuild the entities index from Postgres and swap the alias when verified.

    Mirrors :func:`rebuild_chunks_index` for the entities alias: it creates
    ``<alias>_v<N+1>`` with the entities mapping, streams every active entity
    into it in batches (entity texts without a cached vector are embedded on
    the fly with :func:`~mynotes.embed.cache.ensure_embeddings`), refreshes and
    counts the new index, and swaps the alias only when the count matches the
    number of active entities. On a mismatch a ``RuntimeError`` is raised and
    the alias stays on the old index; the new index is kept for inspection.
    Merged and stale entities are never placed in the new index.

    Args:
        batch: Entities per build/bulk batch.

    Returns:
        ``{"old": ..., "new": ..., "count": ...}``.
    """
    settings = get_settings()
    client = get_client()
    alias = settings.opensearch.entities_alias
    old = current_physical_index(client, alias)
    new = physical_name(alias, _next_version(client, alias, old))
    logger.info("rebuild: creating '%s' (alias '%s' currently on '%s')", new, alias, old)
    create_index(
        client,
        new,
        entities_index_body(settings.models.embedding.dim, settings.text_matching.folding),
    )
    embedder = get_embedder()
    expected = _rebuild_entities_into(settings, client, embedder, new, batch)
    count = _count_after_refresh(client, new)
    if count != expected:
        raise RuntimeError(
            f"rebuild of '{new}' incomplete: {count}/{expected} documents; "
            f"alias '{alias}' left on '{old}'"
        )
    swap_alias(client, alias, new)
    if old is not None and old != new:
        logger.info(
            "rebuild: old index '%s' kept; drop it manually "
            "('mynotes index drop-old' is not implemented)",
            old,
        )
    logger.info("rebuild: '%s' now serves '%s' with %d document(s)", new, alias, count)
    return {"old": old, "new": new, "count": count}


def _rebuild_entities_into(
    settings: Settings, client: OpenSearch, embedder: Embedder, new: str, batch: int
) -> int:
    """Bulk-index every active entity into ``new``; return the expected count.

    Every batch fills missing cache rows via :func:`ensure_embeddings` first
    and renders its documents through :func:`build_entity_docs` in one
    session, so the documents never miss a vector.
    """
    ids = _all_entity_ids()
    expected = len(ids)
    indexed = 0
    for start in range(0, expected, batch):
        entity_ids = ids[start : start + batch]
        with session_scope() as session:
            ensure_embeddings(session, embedder, _entity_texts(session, entity_ids))
            docs = build_entity_docs(session, entity_ids)
        actions = [
            {"_op_type": "index", "_index": new, "_id": entity_id, "_source": source}
            for entity_id, source in docs.items()
            if source is not None
        ]
        if actions:
            bulk_helper(client, actions, raise_on_error=True, raise_on_exception=True)
        indexed += len(actions)
        logger.info("rebuild: %d/%d entity(ies) indexed into '%s'", indexed, expected, new)
    return expected


def _all_entity_ids() -> list[int]:
    """The active KG entity ids the rebuild must place in the new index."""
    with session_scope() as session:
        return list(
            session.execute(
                select(KgEntity.id).where(KgEntity.status == "active").order_by(KgEntity.id)
            ).scalars()
        )


def _entity_texts(session: Session, entity_ids: list[int]) -> list[str]:
    """The ``entity_text`` values of the given entities (existing ones only)."""
    rows = session.execute(
        select(KgEntity.canonical_name, KgEntity.type, KgEntity.description).where(
            KgEntity.id.in_(entity_ids)
        )
    ).all()
    return [entity_text(name, type, description or "") for name, type, description in rows]


def _next_version(client: OpenSearch, alias: str, old: str | None) -> int:
    """Return the next free ``_vN`` version for ``alias``."""
    version = 1
    if old is not None and old.startswith(f"{alias}_v"):
        suffix = old[len(alias) + 2 :]
        if suffix.isdigit():
            version = int(suffix) + 1
    while client.indices.exists(index=physical_name(alias, version)):
        version += 1
    return version


def _rebuild_into(
    settings: Settings, client: OpenSearch, embedder: Embedder, new: str, batch: int
) -> int:
    """Bulk-index every indexable chunk into ``new``; return the expected count.

    Every batch renders its documents through :func:`build_chunk_docs` in one
    session after :func:`ensure_embeddings` filled missing cache rows, so the
    documents never miss a vector. The expected count is the number of
    indexable chunk ids in Postgres: anything less in the new index fails the
    verification in :func:`rebuild_chunks_index`.
    """
    model = settings.models.embedding.model
    passage_prefix = settings.models.embedding.passage_prefix
    ids = _all_indexable_ids(settings)
    expected = len(ids)
    indexed = 0
    for start in range(0, expected, batch):
        chunk_ids = ids[start : start + batch]
        with session_scope() as session:
            ensure_embeddings(session, embedder, _embed_texts(session, chunk_ids))
            docs = build_chunk_docs(session, chunk_ids, model, passage_prefix)
        actions = [
            {"_op_type": "index", "_index": new, "_id": chunk_id, "_source": source}
            for chunk_id, source in docs.items()
            if source is not None
        ]
        if actions:
            bulk_helper(client, actions, raise_on_error=True, raise_on_exception=True)
        indexed += len(actions)
        logger.info("rebuild: %d/%d chunk(s) indexed into '%s'", indexed, expected, new)
    return expected


def _all_indexable_ids(settings: Settings) -> list[UUID]:
    """The indexable chunk ids the rebuild must place in the new index."""
    with session_scope() as session:
        return expected_chunk_ids(session, settings)


def _embed_texts(session: Session, chunk_ids: list[UUID]) -> list[str]:
    """The ``embed_text`` values of the given chunks (existing ones only)."""
    return list(session.execute(select(Chunk.embed_text).where(Chunk.id.in_(chunk_ids))).scalars())


def _reembed_all(settings: Settings, embedder: Embedder, batch: int) -> None:
    """Drop the current model's cache rows and re-embed every indexable chunk."""
    model = settings.models.embedding.model
    with session_scope() as session:
        deleted = session.execute(delete(EmbeddingCache).where(EmbeddingCache.model == model))
    logger.info("rebuild: deleted %d cache row(s) of model '%s'", deleted.rowcount, model)
    ids = _all_indexable_ids(settings)
    done = 0
    next_log = _REEMBED_PROGRESS_STEP
    for start in range(0, len(ids), batch):
        chunk_ids = ids[start : start + batch]
        with session_scope() as session:
            ensure_embeddings(session, embedder, _embed_texts(session, chunk_ids))
        done += len(chunk_ids)
        if done >= next_log or done == len(ids):
            logger.info("rebuild: re-embedded %d/%d chunk(s)", done, len(ids))
            next_log = done + _REEMBED_PROGRESS_STEP


def _count_after_refresh(client: OpenSearch, index: str) -> int:
    """Refresh ``index`` and return its document count (verification hook)."""
    client.indices.refresh(index=index)
    return int(client.count(index=index)["count"])
