"""Supersedes chains and their effects (PLAN.md §7.5 step 6).

An active (``auto``/``confirmed``) supersedes relation makes the older
document ``superseded`` and points it at the head of its chain (relations are
followed transitively). The effect is metadata-only: a reindex job refreshes
``doc_status`` on the older document's indexed chunks, and its KG relation
evidence is marked non-current. When a relation is rejected, the effects are
undone — the document returns to ``current`` unless another active relation
still supersedes it — and the index is refreshed again.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from mynotes.db.models import DocRelation, Document, DocumentVersion, KgRelationEvidence
from mynotes.graph.temporal import enqueue_temporal_refresh
from mynotes.jobs.queue import enqueue
from mynotes.jobs.registry import Priority

_ACTIVE_STATUSES = ("auto", "confirmed")


def _has_active_supersedes(session: Session, older_id: int) -> bool:
    """Return True when an active supersedes relation points at ``older_id``."""
    return (
        session.execute(
            select(DocRelation.id)
            .where(
                DocRelation.dst_document_id == older_id,
                DocRelation.type == "supersedes",
                DocRelation.status.in_(_ACTIVE_STATUSES),
            )
            .limit(1)
        ).scalar_one_or_none()
        is not None
    )


def head_of_chain(session: Session, document_id: int) -> int:
    """Follow active supersedes relations newer-ward to the chain head.

    Args:
        session: An open session/transaction.
        document_id: Any document in the chain.

    Returns:
        The id of the last document reachable through active (``auto`` or
        ``confirmed``) supersedes relations; cycles are broken on the first
        revisit, and when several relations supersede the same document the
        oldest one (lowest id) is followed.
    """
    current = document_id
    visited = {current}
    while True:
        nxt = session.execute(
            select(DocRelation.src_document_id)
            .where(
                DocRelation.dst_document_id == current,
                DocRelation.type == "supersedes",
                DocRelation.status.in_(_ACTIVE_STATUSES),
            )
            .order_by(DocRelation.id)
            .limit(1)
        ).scalar_one_or_none()
        if nxt is None or nxt in visited:
            return current
        visited.add(nxt)
        current = nxt


def apply_supersedes_effects(session: Session, older_id: int) -> None:
    """Apply the effects of an active supersedes relation to the older document.

    Idempotent. When no active supersedes relation points at ``older_id``
    (for example it was rejected) nothing happens. Otherwise the document
    becomes ``superseded`` with ``superseded_by_document_id`` set to its
    chain head, a ``reindex_metadata`` job refreshes ``doc_status`` in the
    index, its KG relation evidence rows are marked non-current and a
    ``kg_temporal_refresh`` job re-derives the affected relations' validity
    (task 42).

    Args:
        session: An open session/transaction; rows are written but not
            committed.
        older_id: The superseded (dst) document id.
    """
    active = _has_active_supersedes(session, older_id)
    if not active:
        return
    older = session.get(Document, older_id)
    if older is None:
        return
    older.status = "superseded"
    older.superseded_by_document_id = head_of_chain(session, older_id)
    older.updated_at = datetime.now(UTC)
    enqueue(
        session,
        "reindex_metadata",
        {"document_id": older_id},
        dedupe_key=f"reindex_metadata:{older_id}",
        priority=Priority.EMBED,
    )
    version_ids = select(DocumentVersion.id).where(DocumentVersion.document_id == older_id)
    session.execute(
        update(KgRelationEvidence)
        .where(KgRelationEvidence.document_version_id.in_(version_ids))
        .values(is_current=False)
    )
    enqueue_temporal_refresh(session, older_id)


def revert_supersedes_effects(session: Session, older_id: int) -> None:
    """Undo the effects after a supersedes relation was rejected or deleted.

    When no other active relation supersedes the document anymore, its status
    returns to ``current`` and ``superseded_by_document_id`` is cleared, the
    latest version's KG evidence becomes current again (task 42), and a
    ``kg_temporal_refresh`` job re-derives the affected relations' validity;
    the index is refreshed in either case so ``doc_status`` stays correct.

    Args:
        session: An open session/transaction.
        older_id: The previously superseded (dst) document id.
    """
    if _has_active_supersedes(session, older_id):
        return
    older = session.get(Document, older_id)
    if older is None:
        return
    if older.status == "superseded":
        older.status = "current"
        older.superseded_by_document_id = None
        older.updated_at = datetime.now(UTC)
        latest_ids = select(DocumentVersion.id).where(
            DocumentVersion.document_id == older_id, DocumentVersion.is_latest.is_(True)
        )
        session.execute(
            update(KgRelationEvidence)
            .where(KgRelationEvidence.document_version_id.in_(latest_ids))
            .values(is_current=True)
        )
        enqueue_temporal_refresh(session, older_id)
    enqueue(
        session,
        "reindex_metadata",
        {"document_id": older_id},
        dedupe_key=f"reindex_metadata:{older_id}",
        priority=Priority.EMBED,
    )
