"""User review of proposed supersedes relations (PLAN.md §10.1, task 35).

The detector (task 34) writes relations with status ``auto`` (applied
immediately) or ``proposed`` (waiting for a decision). This module is the
decision surface behind the ``mynotes versions`` CLI: it lists relations as
view objects with both documents resolved, confirms or rejects them (applying
or undoing the chain effects from :mod:`mynotes.versioning.chains`), swaps a
wrong direction and records manual relations. Like the handlers, the functions
write through the given session and leave the commit to the caller
(``session_scope``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from mynotes.db.models import Chunk, DocRelation, Document, DocumentVersion, File
from mynotes.text.normalize import normalize_path_key
from mynotes.versioning.chains import apply_supersedes_effects, revert_supersedes_effects

_ACTIVE_STATUSES = ("auto", "confirmed")


@dataclass
class DocRef:
    """One endpoint of a relation, with its display fields.

    Attributes:
        document_id: The document id.
        title: The extracted title, or None.
        path: The current file path relative to its source root (empty when
            the document has no file row anymore).
        doc_date: The document date, or None.
        status: The document status (``current``, ``superseded``, ...).
    """

    document_id: int
    title: str | None
    path: str
    doc_date: datetime | None
    status: str


@dataclass
class RelationView:
    """One supersedes relation prepared for display or JSON output.

    Attributes:
        id: The relation id.
        status: ``proposed``, ``auto``, ``confirmed`` or ``rejected``.
        confidence: The detector's confidence in [0, 1].
        method: ``name_content``, ``frontmatter``, ``hash`` or ``manual``.
        newer: The ``src`` document (the successor).
        older: The ``dst`` document (the superseded one).
        evidence: The detector's evidence dict (``jaccard``, ``cosine``,
            ``name_signal``, ``archive_move``, ``order_reason`` for
            ``name_content`` relations).
    """

    id: int
    status: str
    confidence: float
    method: str
    newer: DocRef
    older: DocRef
    evidence: dict


def list_relations(
    session: Session, status: Sequence[str] | None = None, limit: int = 50
) -> list[RelationView]:
    """List supersedes relations, oldest first, with both documents resolved.

    Args:
        session: An open session/transaction.
        status: Only relations with one of these statuses; None means all.
        limit: Maximum number of relations to return.

    Returns:
        One :class:`RelationView` per relation, ordered by relation id.
    """
    statement = select(DocRelation).where(DocRelation.type == "supersedes")
    if status:
        statement = statement.where(DocRelation.status.in_(list(status)))
    statement = statement.order_by(DocRelation.id).limit(limit)
    views: list[RelationView] = []
    for relation in session.execute(statement).scalars():
        views.append(
            RelationView(
                id=relation.id,
                status=relation.status,
                confidence=relation.confidence,
                method=relation.method,
                newer=_doc_ref(session, relation.src_document_id),
                older=_doc_ref(session, relation.dst_document_id),
                evidence=dict(relation.evidence or {}),
            )
        )
    return views


def confirm(session: Session, relation_id: int) -> None:
    """Mark a relation ``confirmed`` and apply its chain effects.

    The older document becomes ``superseded`` (metadata-only, reindex queued)
    and its KG evidence is marked non-current.

    Args:
        session: An open session/transaction; rows are written but not
            committed.
        relation_id: The supersedes relation to confirm.

    Raises:
        ValueError: When the id is not a supersedes relation.
    """
    relation = _get_supersedes(session, relation_id)
    relation.status = "confirmed"
    relation.updated_at = datetime.now(UTC)
    session.flush()
    apply_supersedes_effects(session, relation.dst_document_id)


def reject(session: Session, relation_id: int) -> None:
    """Mark a relation ``rejected`` and undo its chain effects.

    The older document returns to ``current`` unless another active relation
    still supersedes it; the index is refreshed either way.

    Args:
        session: An open session/transaction.
        relation_id: The supersedes relation to reject.

    Raises:
        ValueError: When the id is not a supersedes relation.
    """
    relation = _get_supersedes(session, relation_id)
    relation.status = "rejected"
    relation.updated_at = datetime.now(UTC)
    session.flush()
    revert_supersedes_effects(session, relation.dst_document_id)


def swap(session: Session, relation_id: int) -> None:
    """Reverse a relation's direction (the user says the order is wrong).

    The relation flips to ``(src=old dst, dst=old src)`` and is confirmed, so
    the previous successor becomes the superseded document and the previous
    predecessor returns to ``current`` (unless another active relation keeps
    it superseded).

    Args:
        session: An open session/transaction.
        relation_id: The supersedes relation to swap.

    Raises:
        ValueError: When the id is not a supersedes relation or the reverse
            direction is already taken by another relation.
    """
    relation = _get_supersedes(session, relation_id)
    old_src, old_dst = relation.src_document_id, relation.dst_document_id
    reverse = session.execute(
        select(DocRelation.id).where(
            DocRelation.src_document_id == old_dst,
            DocRelation.dst_document_id == old_src,
            DocRelation.type == "supersedes",
            DocRelation.id != relation.id,
        )
    ).scalar_one_or_none()
    if reverse is not None:
        raise ValueError(
            f"cannot swap relation {relation_id}: the reverse relation {reverse} already exists"
        )
    relation.src_document_id, relation.dst_document_id = old_dst, old_src
    relation.status = "confirmed"
    relation.updated_at = datetime.now(UTC)
    session.flush()
    revert_supersedes_effects(session, old_dst)
    apply_supersedes_effects(session, old_src)


def add_manual(session: Session, newer_path: str, older_path: str) -> int:
    """Record a manual ``newer supersedes older`` relation, confirmed.

    Both paths are resolved with :func:`find_document_by_path`. An existing
    relation between the two documents in this direction is upgraded to a
    confirmed manual one.

    Args:
        session: An open session/transaction.
        newer_path: Path of the successor document.
        older_path: Path of the superseded document.

    Returns:
        The relation id.

    Raises:
        ValueError: When a path resolves to no (or several) documents, when
            both paths are the same document, or when the reverse relation
            already exists.
    """
    newer_id, _ = find_document_by_path(session, newer_path)
    older_id, _ = find_document_by_path(session, older_path)
    if newer_id == older_id:
        raise ValueError("newer and older paths resolve to the same document")
    reverse = session.execute(
        select(DocRelation).where(
            DocRelation.src_document_id == older_id,
            DocRelation.dst_document_id == newer_id,
            DocRelation.type == "supersedes",
        )
    ).scalar_one_or_none()
    if reverse is not None:
        raise ValueError(
            f"the reverse relation {reverse.id} already exists; swap it instead"
            " (mynotes versions swap)"
        )
    relation = session.execute(
        select(DocRelation).where(
            DocRelation.src_document_id == newer_id,
            DocRelation.dst_document_id == older_id,
            DocRelation.type == "supersedes",
        )
    ).scalar_one_or_none()
    now = datetime.now(UTC)
    if relation is None:
        relation = DocRelation(
            src_document_id=newer_id,
            dst_document_id=older_id,
            type="supersedes",
            method="manual",
            confidence=1.0,
            status="confirmed",
            evidence={"manual": True},
        )
        session.add(relation)
    else:
        relation.method = "manual"
        relation.confidence = 1.0
        relation.status = "confirmed"
        relation.updated_at = now
    session.flush()
    apply_supersedes_effects(session, older_id)
    return relation.id


def find_document_by_path(session: Session, path: str) -> tuple[int, str | None]:
    """Resolve a file path to its document id and title.

    The path may be relative to a source root (as shown by ``mynotes search``)
    or any deeper/absolute path ending in the same way. Matching tries the
    exact rel path, then the accent/case-folded one, then a path suffix.

    Args:
        session: An open session/transaction.
        path: The path to resolve.

    Returns:
        The document id and its title (None when untitled).

    Raises:
        ValueError: When no document matches, or several documents do.
    """
    cleaned = str(path).strip().strip("/")
    if not cleaned:
        raise ValueError("empty path")
    escaped = cleaned.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    rows = (
        session.execute(
            select(File)
            .where(
                File.status != "deleted",
                File.document_id.is_not(None),
                or_(
                    File.rel_path == cleaned,
                    File.rel_path_norm == normalize_path_key(cleaned),
                    File.rel_path.like("%/" + escaped),
                ),
            )
            .order_by(File.id)
        )
        .scalars()
        .all()
    )
    active = [row for row in rows if row.status == "active"] or rows
    document_ids: list[int] = []
    for row in active:
        if row.document_id not in document_ids:
            document_ids.append(row.document_id)
    if not document_ids:
        raise ValueError(f"no document found for path {path!r}")
    if len(document_ids) > 1:
        raise ValueError(f"path {path!r} is ambiguous: documents {document_ids}")
    document = session.get(Document, document_ids[0])
    assert document is not None
    return document.id, document.title


def document_preview(session: Session, document_id: int, limit: int = 300) -> str:
    """The first ``limit`` characters of a document's latest version text.

    Args:
        session: An open session/transaction.
        document_id: The document to preview.
        limit: Maximum number of characters to return.

    Returns:
        The joined chunk texts of the latest version, truncated; empty when
        the document has no (chunked) version.
    """
    version_id = session.execute(
        select(Document.latest_version_id).where(Document.id == document_id)
    ).scalar_one_or_none()
    if version_id is None:
        return ""
    texts = list(
        session.execute(
            select(Chunk.text)
            .where(Chunk.document_version_id == version_id)
            .order_by(Chunk.ordinal)
        ).scalars()
    )
    return "\n".join(texts).strip()[:limit]


def _get_supersedes(session: Session, relation_id: int) -> DocRelation:
    """Load one supersedes relation or raise."""
    relation = session.get(DocRelation, relation_id)
    if relation is None or relation.type != "supersedes":
        raise ValueError(f"relation {relation_id} is not a supersedes relation")
    return relation


def _doc_ref(session: Session, document_id: int) -> DocRef:
    """Build the :class:`DocRef` of one relation endpoint."""
    row = session.execute(
        select(Document, File.rel_path)
        .outerjoin(File, File.id == Document.current_file_id)
        .where(Document.id == document_id)
    ).first()
    if row is None:
        return DocRef(document_id=document_id, title=None, path="", doc_date=None, status="deleted")
    document, rel_path = row
    path = rel_path
    if not path and document.latest_version_id is not None:
        path = (
            session.execute(
                select(DocumentVersion.rel_path_at_version).where(
                    DocumentVersion.id == document.latest_version_id
                )
            ).scalar_one_or_none()
            or ""
        )
    return DocRef(
        document_id=document.id,
        title=document.title,
        path=path or "",
        doc_date=document.doc_date,
        status=document.status,
    )
