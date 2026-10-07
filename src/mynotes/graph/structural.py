"""Structural graph queries over documents, folders, tags and relations.

The structural layer (PLAN.md §8.1) is deterministic and cheap: nodes are
documents, folders and tags; edges are the ``doc_relation`` rows
(``links_to``, ``supersedes``, ``duplicate_of``) plus the implicit
folder/tag membership from ``file.rel_path`` and ``document_tag``. Task 43
uses :func:`linked_documents` for structural expansion in retrieval and the
``mynotes graph doc`` command renders :class:`StructuralNeighbours`.
"""

from __future__ import annotations

import posixpath
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import case, func, or_, select, text
from sqlalchemy.orm import Session

from mynotes.db.models import DocRelation, Document, DocumentTag, File, Tag
from mynotes.tags.store import accepted_tags
from mynotes.text.normalize import normalize_key, normalize_path_key

_ACTIVE_STATUSES = ("auto", "confirmed")
_SAME_FOLDER_LIMIT = 20
_SAME_PROJECT_LIMIT = 20


@dataclass
class DocNode:
    """One document as a graph node.

    Attributes:
        document_id: The document id.
        title: The document title, or None when untitled.
        path: The current file path (POSIX separators), or ``""`` when the
            document has no current file.
        status: The document status (``current``, ``superseded``, ...).
        doc_date: The document date, or None.
    """

    document_id: int
    title: str | None
    path: str
    status: str
    doc_date: datetime | None


@dataclass
class StructuralNeighbours:
    """Everything structurally connected to one document.

    Attributes:
        document: The document itself as a node.
        links_out: Documents this one links to (active ``links_to``).
        links_in: Documents linking to this one (active ``links_to``).
        supersedes: Documents this one supersedes (active relations).
        superseded_by: Documents superseding this one (active relations).
        duplicates: Documents related by ``duplicate_of`` in either direction
            (active relations).
        same_folder: Up to 20 other documents in the same folder, most recent
            first (by doc_date, then id).
        same_project: Up to 20 other documents sharing an accepted
            ``project:*`` tag, most recent first.
        tags: The document's accepted tag names.
    """

    document: DocNode
    links_out: list[DocNode]
    links_in: list[DocNode]
    supersedes: list[DocNode]
    superseded_by: list[DocNode]
    duplicates: list[DocNode]
    same_folder: list[DocNode]
    same_project: list[DocNode]
    tags: list[str]


def neighbours(session: Session, document_id: int) -> StructuralNeighbours:
    """Collect the structural neighbourhood of one document.

    Args:
        session: An open session/transaction.
        document_id: The document whose neighbours to collect.

    Returns:
        The :class:`StructuralNeighbours` of the document.

    Raises:
        LookupError: When the document does not exist.
    """
    document = session.get(Document, document_id)
    if document is None:
        raise LookupError(f"document {document_id} does not exist")
    related: dict[str, list[int]] = {
        "links_out": _related_document_ids(session, document_id, "links_to", outgoing=True),
        "links_in": _related_document_ids(session, document_id, "links_to", outgoing=False),
        "supersedes": _related_document_ids(session, document_id, "supersedes", outgoing=True),
        "superseded_by": _related_document_ids(session, document_id, "supersedes", outgoing=False),
        "duplicates": _duplicate_ids(session, document_id),
    }
    related_ids = {document_id}
    for ids in related.values():
        related_ids.update(ids)
    nodes = _load_nodes(session, related_ids)
    path = nodes[document_id].path
    return StructuralNeighbours(
        document=nodes[document_id],
        links_out=_nodes(nodes, related["links_out"]),
        links_in=_nodes(nodes, related["links_in"]),
        supersedes=_nodes(nodes, related["supersedes"]),
        superseded_by=_nodes(nodes, related["superseded_by"]),
        duplicates=_nodes(nodes, related["duplicates"]),
        same_folder=_nodes(nodes, _same_folder_ids(session, document, path)),
        same_project=_nodes(nodes, _same_project_ids(session, document)),
        tags=accepted_tags(session, document_id),
    )


def find_document(session: Session, ref: str) -> list[DocNode]:
    """Resolve a path-or-title reference to candidate documents.

    Matching follows PLAN.md §6.1: exact ``rel_path``, then the folded
    ``rel_path`` (case/accent-insensitive), then the folded title. The first
    step with hits wins.

    Args:
        session: An open session/transaction.
        ref: A file path relative to a source root, or a document title.

    Returns:
        The matching non-deleted documents (may be empty or several).
    """
    cleaned = ref.strip()
    if not cleaned:
        return []
    candidates: list[DocNode] = []
    seen: set[int] = set()

    def add(rows: Sequence[Any]) -> None:
        for document, rel_path in rows:
            if document.id not in seen:
                seen.add(document.id)
                candidates.append(_make_node(document, rel_path))

    add(
        session.execute(
            select(Document, File.rel_path)
            .join(File, File.id == Document.current_file_id)
            .where(
                File.rel_path == cleaned,
                File.status != "deleted",
                Document.status != "deleted",
            )
            .order_by(Document.id)
        ).all()
    )
    if candidates:
        return candidates
    add(
        session.execute(
            select(Document, File.rel_path)
            .join(File, File.id == Document.current_file_id)
            .where(
                File.rel_path_norm == normalize_path_key(cleaned),
                File.status != "deleted",
                Document.status != "deleted",
            )
            .order_by(Document.id)
        ).all()
    )
    if candidates:
        return candidates
    add(
        session.execute(
            select(Document, File.rel_path)
            .outerjoin(File, File.id == Document.current_file_id)
            .where(
                Document.title_norm == normalize_key(cleaned),
                Document.status != "deleted",
            )
            .order_by(Document.id)
        ).all()
    )
    return candidates


def linked_documents(
    session: Session, document_ids: Sequence[int], hops: int = 1
) -> dict[int, int]:
    """Map document id to hop distance over active ``links_to`` relations.

    Follows relations in both directions from every input, restricted to
    non-deleted documents and up to ``hops`` steps; the inputs themselves are
    never in the result. The recursive CTE is cycle-safe: the depth grows
    monotonically and ``UNION`` collapses repeated (node, depth) pairs.

    Args:
        session: An open session/transaction.
        document_ids: The starting document ids.
        hops: Maximum number of relation steps to follow (>= 1).

    Returns:
        A dict of reachable document id to the smallest hop distance.
    """
    starts = sorted({int(document_id) for document_id in document_ids})
    if not starts or hops < 1:
        return {}
    query = text(
        """
        WITH RECURSIVE walk(node, depth) AS (
            SELECT inp.node, 0
            FROM unnest(CAST(:ids AS bigint[])) AS inp(node)
            JOIN document d ON d.id = inp.node AND d.status <> 'deleted'
          UNION
            SELECT CASE WHEN r.src_document_id = w.node
                        THEN r.dst_document_id ELSE r.src_document_id END,
                   w.depth + 1
            FROM walk w
            JOIN doc_relation r
              ON r.type = 'links_to'
             AND r.status IN ('auto', 'confirmed')
             AND (r.src_document_id = w.node OR r.dst_document_id = w.node)
            JOIN document d
              ON d.id = (CASE WHEN r.src_document_id = w.node
                              THEN r.dst_document_id ELSE r.src_document_id END)
             AND d.status <> 'deleted'
            WHERE w.depth < :hops
        )
        SELECT node, min(depth)
        FROM walk
        WHERE node <> ALL(CAST(:ids AS bigint[]))
        GROUP BY node
        """
    )
    rows = session.execute(query, {"ids": starts, "hops": hops}).all()
    return {int(node): int(depth) for node, depth in rows}


def _make_node(document: Document, rel_path: str | None) -> DocNode:
    """Build a :class:`DocNode` from a document row and its file path."""
    return DocNode(
        document_id=document.id,
        title=document.title,
        path=rel_path or "",
        status=document.status,
        doc_date=document.doc_date,
    )


def _load_nodes(session: Session, document_ids: Sequence[int]) -> dict[int, DocNode]:
    """Load documents with their current file path keyed by id."""
    unique = sorted(set(document_ids))
    if not unique:
        return {}
    rows = session.execute(
        select(Document, File.rel_path)
        .outerjoin(File, File.id == Document.current_file_id)
        .where(Document.id.in_(unique))
    ).all()
    return {document.id: _make_node(document, rel_path) for document, rel_path in rows}


def _nodes(nodes: dict[int, DocNode], ids: Sequence[int]) -> list[DocNode]:
    """Project ids onto loaded nodes, keeping the id order."""
    return [nodes[document_id] for document_id in ids if document_id in nodes]


def _related_document_ids(
    session: Session, document_id: int, relation_type: str, *, outgoing: bool
) -> list[int]:
    """The ids on the other end of the document's active relations of a type."""
    column = DocRelation.src_document_id if outgoing else DocRelation.dst_document_id
    other = DocRelation.dst_document_id if outgoing else DocRelation.src_document_id
    return list(
        session.execute(
            select(other)
            .where(
                column == document_id,
                DocRelation.type == relation_type,
                DocRelation.status.in_(_ACTIVE_STATUSES),
            )
            .order_by(other)
        ).scalars()
    )


def _duplicate_ids(session: Session, document_id: int) -> list[int]:
    """The ids of documents related by ``duplicate_of`` in either direction."""
    rows = session.execute(
        select(DocRelation.src_document_id, DocRelation.dst_document_id).where(
            DocRelation.type == "duplicate_of",
            DocRelation.status.in_(_ACTIVE_STATUSES),
            or_(
                DocRelation.src_document_id == document_id,
                DocRelation.dst_document_id == document_id,
            ),
        )
    ).all()
    return sorted(dst if src == document_id else src for src, dst in rows)


def _same_folder_ids(session: Session, document: Document, path: str) -> list[int]:
    """Up to 20 other workspace documents in the same folder, most recent first."""
    folder = posixpath.dirname(path) if path else ""
    folder_expr = case(
        (File.rel_path.contains("/"), func.regexp_replace(File.rel_path, "/[^/]*$", "")),
        else_="",
    )
    return list(
        session.execute(
            select(Document.id)
            .join(File, File.id == Document.current_file_id)
            .where(
                Document.workspace_id == document.workspace_id,
                Document.status != "deleted",
                Document.id != document.id,
                File.status != "deleted",
                folder_expr == folder,
            )
            .order_by(Document.doc_date.desc().nulls_last(), Document.id.desc())
            .limit(_SAME_FOLDER_LIMIT)
        ).scalars()
    )


def _same_project_ids(session: Session, document: Document) -> list[int]:
    """Up to 20 other workspace documents sharing an accepted ``project:*`` tag."""
    project_tags = (
        select(DocumentTag.tag_id)
        .join(Tag, Tag.id == DocumentTag.tag_id)
        .where(
            DocumentTag.document_id == document.id,
            Tag.name.like("project:%"),
            DocumentTag.status == "accepted",
        )
    )
    rows = session.execute(
        select(Document.id, Document.doc_date)
        .join(DocumentTag, DocumentTag.document_id == Document.id)
        .where(
            Document.workspace_id == document.workspace_id,
            Document.status != "deleted",
            Document.id != document.id,
            DocumentTag.status == "accepted",
            DocumentTag.tag_id.in_(project_tags),
        )
        .distinct()
        .order_by(Document.doc_date.desc().nulls_last(), Document.id.desc())
        .limit(_SAME_PROJECT_LIMIT)
    ).all()
    return [int(document_id) for document_id, _doc_date in rows]
