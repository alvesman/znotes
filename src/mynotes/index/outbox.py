"""Write side of the index outbox (PLAN.md §5.5, §7.6).

Every chunk or entity change that must reach OpenSearch gets an
``index_outbox`` row in the same transaction as the change. This module only
writes rows; the syncer (task 22) drains them with ``_bulk`` and collapses
repeated rows for the same document to the latest op, so nothing here
deduplicates.
"""

from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID

from sqlalchemy import insert
from sqlalchemy.orm import Session

from mynotes.db.models import IndexOutbox

_CHUNKS_INDEX = "chunks"
_ENTITIES_INDEX = "entities"


def _enqueue(
    session: Session, index_name: str, doc_kind: str, op: str, doc_ids: Iterable[UUID | int | str]
) -> int:
    """Append one outbox row per id and return the row count.

    Args:
        session: An open session/transaction; rows are written but not
            committed.
        index_name: Target OpenSearch alias (``chunks`` or ``entities``).
        doc_kind: The outbox document kind (``chunk`` or ``entity``).
        op: ``upsert`` or ``delete``.
        doc_ids: The document ids; stored as strings.

    Returns:
        The number of rows appended.
    """
    ids = list(doc_ids)
    if not ids:
        return 0
    # Core executemany insert: an initial 20k-file ingest enqueues hundreds of
    # thousands of rows and the ORM unit-of-work path is pure overhead here.
    session.execute(
        insert(IndexOutbox),
        [
            {
                "index_name": index_name,
                "doc_kind": doc_kind,
                "doc_id": str(doc_id),
                "op": op,
            }
            for doc_id in ids
        ],
    )
    return len(ids)


def enqueue_chunk_upserts(session: Session, chunk_ids: Iterable[UUID | str]) -> int:
    """Queue upserts for chunk documents.

    Args:
        session: An open session/transaction.
        chunk_ids: The chunk ids to (re)index.

    Returns:
        The number of outbox rows appended.
    """
    return _enqueue(session, _CHUNKS_INDEX, "chunk", "upsert", chunk_ids)


def enqueue_chunk_deletes(session: Session, chunk_ids: Iterable[UUID | str]) -> int:
    """Queue deletes for chunk documents.

    Args:
        session: An open session/transaction.
        chunk_ids: The chunk ids to remove from the index.

    Returns:
        The number of outbox rows appended.
    """
    return _enqueue(session, _CHUNKS_INDEX, "chunk", "delete", chunk_ids)


def enqueue_entity_upserts(session: Session, entity_ids: Iterable[int]) -> int:
    """Queue upserts for entity documents (used by task 41).

    Args:
        session: An open session/transaction.
        entity_ids: The KG entity ids to (re)index.

    Returns:
        The number of outbox rows appended.
    """
    return _enqueue(session, _ENTITIES_INDEX, "entity", "upsert", entity_ids)


def enqueue_entity_deletes(session: Session, entity_ids: Iterable[int]) -> int:
    """Queue deletes for entity documents.

    Args:
        session: An open session/transaction.
        entity_ids: The KG entity ids to remove from the index.

    Returns:
        The number of outbox rows appended.
    """
    return _enqueue(session, _ENTITIES_INDEX, "entity", "delete", entity_ids)
