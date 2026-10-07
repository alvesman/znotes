"""Builders that turn Postgres rows into OpenSearch documents (PLAN.md §6, §7.6).

:func:`indexable_chunk_ids` decides which chunks of a document belong in the
search index, :func:`build_chunk_docs` renders chunks as ``_source`` payloads
(reading vectors from the Postgres embedding cache, so building documents never
calls a model endpoint) and :func:`document_index_tags` reads the accepted tags
stored on a document (task 37). The syncer (task 22) uses all three.
:func:`build_entity_docs` does the same for KG entities (task 41) and is
installed as the syncer's ``ENTITY_DOC_BUILDER`` by :mod:`mynotes.index.sync`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from mynotes.config import Settings, get_settings
from mynotes.db.models import (
    Chunk,
    Document,
    DocumentTag,
    DocumentVersion,
    File,
    KgEntity,
    KgEntityAlias,
    SourceRoot,
    Tag,
)
from mynotes.embed.cache import get_cached
from mynotes.graph.resolve import entity_text
from mynotes.platform.paths import path_ancestors, path_ancestors_folded
from mynotes.scan.hashing import hash_text
from mynotes.tags.store import accepted_tags


def heading_path_text(heading_path: Sequence[str]) -> str:
    """Join a chunk's heading path for the ``heading_path_text`` index field.

    Args:
        heading_path: The heading titles from the document root to the chunk.

    Returns:
        The titles joined with ``" > "``; an empty path becomes ``""``.
    """
    return " > ".join(heading_path)


def chunk_source(
    *,
    chunk_id: UUID | int | str,
    document_id: int | str,
    version_id: int | str,
    source_root_id: int | str,
    workspace_id: int | str,
    text: str,
    title: str | None,
    heading_path: Sequence[str],
    section: str | None,
    tags: Sequence[str],
    format: str,
    language: str | None,
    doc_status: str,
    rel_path: str,
    doc_date: datetime | None,
    is_latest: bool,
    is_table: bool,
    page: int | None,
    slide: int | None,
    ordinal: int,
    embedding: Sequence[float],
) -> dict[str, Any]:
    """Build one chunk ``_source`` payload for the ``mynotes_chunks`` mapping.

    Pure helper (no database access) so the field rules — string ids, the
    ``" > "`` heading join, exact ``path_ancestors`` next to folded keys and
    ISO ``doc_date`` — are testable without Postgres.

    Args:
        chunk_id: The chunk id (stored as a string).
        document_id: The owning document id (stored as a string).
        version_id: The owning document version id (stored as a string).
        source_root_id: The document's source root id (stored as a string).
        workspace_id: The document's workspace id (stored as a string).
        text: The chunk text.
        title: The document title, or None.
        heading_path: The chunk's heading path.
        section: The chunk's section label, or None.
        tags: The document's index tags.
        format: The document format (``md``, ``pdf``, ...).
        language: The resolved language (chunk language or document language).
        doc_status: The document status (``current``, ...).
        rel_path: The current file path relative to the source root.
        doc_date: The document date, or None.
        is_latest: Whether the chunk belongs to the document's latest version.
        is_table: Whether the chunk is a table chunk.
        page: The PDF page number, or None.
        slide: The PPTX slide number, or None.
        ordinal: The chunk position within its version.
        embedding: The chunk's cached embedding vector.

    Returns:
        The ``_source`` dict; field names match the strict mapping exactly.
    """
    return {
        "chunk_id": str(chunk_id),
        "document_id": str(document_id),
        "version_id": str(version_id),
        "source_root_id": str(source_root_id),
        "workspace_id": str(workspace_id),
        "text": text,
        "title": title,
        "heading_path_text": heading_path_text(heading_path),
        "section": section,
        "tags": list(tags),
        "format": format,
        "language": language,
        "doc_status": doc_status,
        "path": rel_path,
        "path_ancestors": path_ancestors(rel_path),
        "path_ancestors_folded": path_ancestors_folded(rel_path),
        "doc_date": doc_date.isoformat() if doc_date is not None else None,
        "is_latest": is_latest,
        "is_table": is_table,
        "page": page,
        "slide": slide,
        "ordinal": ordinal,
        "embedding": list(embedding),
    }


def chunk_meta_hash(source: dict[str, Any]) -> str:
    """Fingerprint the index-relevant part of a chunk ``_source`` (task 23).

    The reconciler compares this hash between the index and a fresh build from
    Postgres to find stale documents. The embedding and the hash field itself
    are excluded, so only metadata/content changes count as drift.

    Args:
        source: A ``chunk_source`` payload (``meta_hash`` may already be set).

    Returns:
        :func:`mynotes.scan.hashing.hash_text` of the remaining fields as JSON
        with sorted keys.
    """
    payload = {key: value for key, value in source.items() if key not in ("embedding", "meta_hash")}
    return hash_text(json.dumps(payload, sort_keys=True))


def indexable_chunk_ids(session: Session, document_id: int, settings: Settings) -> list[UUID]:
    """List the chunks of a document that should be present in the index.

    Final rule (task 33). For a non-deleted document:

    - ``versions.index_old_versions`` off (default): the chunks of the latest
      version only;
    - on: the chunks of every retained version — the latest with
      ``is_latest=true``, older ones with ``is_latest=false``. Purged versions
      no longer exist, so "all versions" and "all retained versions" coincide.

    Changing ``versions.index_old_versions`` requires ``mynotes index rebuild``;
    the regular embed/sync flow only keeps the chunks it writes consistent.
    Deleted (and missing) documents, and documents of a disabled source root
    (removed from the configuration), have no indexable chunks.

    Args:
        session: An open session/transaction.
        document_id: The document to collect chunks for.
        settings: Application settings (``versions.index_old_versions``).

    Returns:
        The chunk ids ordered by ordinal.
    """
    document = session.get(Document, document_id)
    if document is None or document.status == "deleted":
        return []
    root = session.get(SourceRoot, document.source_root_id)
    if root is not None and not root.enabled:
        return []
    stmt = (
        select(Chunk.id)
        .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
        .where(DocumentVersion.document_id == document_id)
    )
    if not settings.versions.index_old_versions:
        stmt = stmt.where(DocumentVersion.is_latest.is_(True))
    return list(session.execute(stmt.order_by(Chunk.ordinal)).scalars())


def build_chunk_docs(
    session: Session,
    chunk_ids: Sequence[UUID],
    embedder_model: str,
    passage_prefix: str,
) -> dict[str, dict[str, Any] | None]:
    """Build the OpenSearch ``_source`` payloads for the given chunks.

    Rows are loaded with a few set-based queries; embeddings are read from the
    ``embedding_cache`` (keyed by ``hash_text(passage_prefix + embed_text)``),
    so this never calls a model endpoint.

    Args:
        session: An open session/transaction.
        chunk_ids: The chunk ids to render.
        embedder_model: The embedding model name (first cache key part).
        passage_prefix: The embedder's passage prefix (part of the cache key).

    Returns:
        A mapping from chunk id (as string) to the ``_source`` dict, or None
        when the chunk no longer exists, its document is deleted (or has no
        current file), or its embedding is not cached yet.
    """
    ids = list(chunk_ids)
    if not ids:
        return {}
    chunks = list(session.execute(select(Chunk).where(Chunk.id.in_(ids))).scalars())
    documents = _by_id(session, Document, {chunk.document_id for chunk in chunks})
    versions = _by_id(session, DocumentVersion, {chunk.document_version_id for chunk in chunks})
    file_ids = {
        document.current_file_id
        for document in documents.values()
        if document.current_file_id is not None
    }
    files = _by_id(session, File, file_ids)
    tags_by_document = _tags_by_document(session, list(documents))
    vectors = get_cached(
        session, embedder_model, [hash_text(passage_prefix + chunk.embed_text) for chunk in chunks]
    )
    result: dict[str, dict[str, Any] | None] = {str(chunk_id): None for chunk_id in ids}
    for chunk in chunks:
        document = documents.get(chunk.document_id)
        version = versions.get(chunk.document_version_id)
        file_row = (
            files.get(document.current_file_id)
            if document is not None and document.current_file_id is not None
            else None
        )
        vector = vectors.get(hash_text(passage_prefix + chunk.embed_text))
        if document is None or version is None or file_row is None or vector is None:
            continue
        if document.status == "deleted":
            continue
        source = chunk_source(
            chunk_id=chunk.id,
            document_id=document.id,
            version_id=version.id,
            source_root_id=document.source_root_id,
            workspace_id=document.workspace_id,
            text=chunk.text,
            title=document.title,
            heading_path=chunk.heading_path,
            section=chunk.section,
            tags=tags_by_document.get(document.id, []),
            format=document.format,
            language=chunk.language or document.language,
            doc_status=document.status,
            rel_path=file_row.rel_path,
            doc_date=document.doc_date,
            is_latest=version.is_latest,
            is_table=chunk.is_table,
            page=chunk.page,
            slide=chunk.slide,
            ordinal=chunk.ordinal,
            embedding=vector,
        )
        source["meta_hash"] = chunk_meta_hash(source)
        result[str(chunk.id)] = source
    return result


def build_entity_docs(
    session: Session, ids: Sequence[int | str]
) -> dict[str, dict[str, Any] | None]:
    """Build the OpenSearch ``_source`` payloads for the given KG entities.

    Active entities render with their aliases and the cached embedding of
    :func:`mynotes.graph.resolve.entity_text` (read from the Postgres
    ``embedding_cache`` under ``models.embedding.model``/``passage_prefix``,
    so building documents never calls a model endpoint). Merged or stale
    entities, unknown ids and entities without a cached vector map to None —
    the syncer turns that into a delete, so merged entities leave the index.

    Args:
        session: An open session/transaction.
        ids: The KG entity ids (ints or their string form).

    Returns:
        A mapping from entity id (as string) to the ``_source`` dict, or None
        when the entity must not be in the index.
    """
    settings = get_settings()
    id_list = [int(entity_id) for entity_id in ids]
    if not id_list:
        return {}
    entities = list(session.execute(select(KgEntity).where(KgEntity.id.in_(id_list))).scalars())
    aliases_by_entity: dict[int, list[str]] = {}
    if entities:
        for entity_id, alias in session.execute(
            select(KgEntityAlias.entity_id, KgEntityAlias.alias)
            .where(KgEntityAlias.entity_id.in_([entity.id for entity in entities]))
            .order_by(KgEntityAlias.entity_id, KgEntityAlias.alias)
        ).all():
            aliases_by_entity.setdefault(int(entity_id), []).append(str(alias))
    vectors = get_cached(
        session,
        settings.models.embedding.model,
        [
            hash_text(
                settings.models.embedding.passage_prefix
                + entity_text(entity.canonical_name, entity.type, entity.description or "")
            )
            for entity in entities
        ],
    )
    result: dict[str, dict[str, Any] | None] = {str(entity_id): None for entity_id in id_list}
    for entity in entities:
        if entity.status != "active":
            continue
        vector = vectors.get(
            hash_text(
                settings.models.embedding.passage_prefix
                + entity_text(entity.canonical_name, entity.type, entity.description or "")
            )
        )
        if vector is None:
            continue
        result[str(entity.id)] = {
            "entity_id": str(entity.id),
            "workspace_id": str(entity.workspace_id),
            "type": entity.type,
            "status": entity.status,
            "canonical_name": entity.canonical_name,
            "aliases": aliases_by_entity.get(entity.id, []),
            "description": entity.summary if entity.summary else (entity.description or ""),
            "embedding": list(vector),
        }
    return result


def document_index_tags(session: Session, document_id: int) -> list[str]:
    """Collect the tags of a document for the index ``tags`` field.

    The accepted ``document_tag`` rows (all sources: folder rules, frontmatter,
    hashtags and later ``llm``/``manual``), so aliases apply in the index.

    Args:
        session: An open session/transaction.
        document_id: The document whose tags to collect.

    Returns:
        The accepted tag names, ordered by source then name; empty when the
        document has none.
    """
    return accepted_tags(session, document_id)


def _tags_by_document(session: Session, document_ids: Sequence[int]) -> dict[int, list[str]]:
    """Read the accepted tags of several documents with one query."""
    if not document_ids:
        return {}
    rows = session.execute(
        select(DocumentTag.document_id, DocumentTag.source, Tag.name)
        .join(Tag, Tag.id == DocumentTag.tag_id)
        .where(
            DocumentTag.document_id.in_(document_ids),
            DocumentTag.status == "accepted",
        )
        .order_by(DocumentTag.document_id, DocumentTag.source, Tag.name)
    ).all()
    result: dict[int, list[str]] = {}
    for document_id, _source, name in rows:
        result.setdefault(int(document_id), []).append(str(name))
    return result


def _by_id(session: Session, model: type, ids: Sequence[int | None]) -> dict[int, Any]:
    """Load rows of ``model`` by primary key in one query (None ids dropped)."""
    present = {row_id for row_id in ids if row_id is not None}
    if not present:
        return {}
    rows = session.execute(select(model).where(model.id.in_(present))).scalars()
    return {row.id: row for row in rows}
