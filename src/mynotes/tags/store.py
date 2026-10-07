"""Tag persistence: vocabulary, aliases and per-document tags (PLAN.md §5.2).

Manual tags (folder rules, frontmatter, hashtags) are stored as ``accepted``
``document_tag`` rows; the index and the query-time tag filters read the same
accepted set, so aliases (``tag_alias``) apply everywhere consistently.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from sqlalchemy import and_, delete, exists, func, literal, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, aliased

from mynotes.db.models import Document, DocumentTag, Tag, TagAlias, Workspace
from mynotes.tags.normalize import normalize_tag, tag_kind

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TagSummary:
    """One row of the tag vocabulary listing (``mynotes tags list``).

    Attributes:
        name: The normalized tag name.
        kind: The tag kind (``project``, ``topic``, ``status``, ``folder`` or
            ``other``).
        document_count: Number of documents with accepted rows of the tag.
        sources: The sorted sources the tag is applied with.
    """

    name: str
    kind: str
    document_count: int
    sources: list[str]


@dataclass(frozen=True)
class TagDetail:
    """One ``document_tag`` row of a document (``mynotes tags show``)."""

    name: str
    kind: str
    source: str
    status: str
    confidence: float


def workspace_id_by_name(session: Session, name: str) -> int | None:
    """Return the id of the named workspace, or None when it does not exist.

    Args:
        session: An open session/transaction.
        name: The workspace name (``settings.workspace``).

    Returns:
        The workspace id, or None.
    """
    return session.execute(select(Workspace.id).where(Workspace.name == name)).scalar_one_or_none()


def resolve_alias(session: Session, workspace_id: int, name: str) -> str:
    """Resolve a tag name through the workspace's ``tag_alias`` vocabulary.

    Args:
        session: An open session/transaction.
        workspace_id: The workspace the vocabulary belongs to.
        name: A (normalized) tag name.

    Returns:
        The canonical tag name the alias points at, or the normalized input
        name when no alias exists.
    """
    key = normalize_tag(name)
    row = session.execute(
        select(Tag.name)
        .join(TagAlias, TagAlias.tag_id == Tag.id)
        .where(TagAlias.workspace_id == workspace_id, TagAlias.alias_norm == key)
    ).first()
    return str(row[0]) if row is not None else key


def ensure_tag(session: Session, workspace_id: int, name: str) -> int:
    """Return the id of the named tag, resolving aliases and creating it if new.

    Args:
        session: An open session/transaction; the caller commits.
        workspace_id: The workspace the vocabulary belongs to.
        name: A tag name in any spelling; normalized, then alias-resolved.

    Returns:
        The tag id (an existing row, the alias target, or a freshly inserted
        tag with ``display_name = name`` and ``kind`` from the prefix).

    Raises:
        ValueError: When the name is empty after normalization.
    """
    resolved = resolve_alias(session, workspace_id, name)
    if not resolved:
        raise ValueError("tag name is empty after normalization")
    tag_id = session.execute(
        select(Tag.id).where(Tag.workspace_id == workspace_id, Tag.name == resolved)
    ).scalar_one_or_none()
    if tag_id is not None:
        return int(tag_id)
    # ON CONFLICT instead of a plain insert: two workers chunking documents of
    # the same new folder race here, and a UniqueViolation would fail (and
    # back off) whole chunk jobs.
    session.execute(
        pg_insert(Tag)
        .values(
            workspace_id=workspace_id,
            name=resolved,
            display_name=resolved,
            kind=tag_kind(resolved),
        )
        .on_conflict_do_nothing(index_elements=["workspace_id", "name"])
    )
    tag_id = int(
        session.execute(
            select(Tag.id).where(Tag.workspace_id == workspace_id, Tag.name == resolved)
        ).scalar_one()
    )
    logger.debug(
        "created tag %r (kind %s) in workspace %s", resolved, tag_kind(resolved), workspace_id
    )
    return tag_id


def accepted_tags(session: Session, document_id: int) -> list[str]:
    """Return the accepted tag names of a document across all sources.

    Args:
        session: An open session/transaction.
        document_id: The document whose tags to collect.

    Returns:
        The names of the ``status='accepted'`` tags, ordered by source
        (``folder`` first) then name.
    """
    rows = session.execute(
        select(DocumentTag.source, Tag.name)
        .join(Tag, Tag.id == DocumentTag.tag_id)
        .where(DocumentTag.document_id == document_id, DocumentTag.status == "accepted")
        .order_by(DocumentTag.source, Tag.name)
    ).all()
    return [str(name) for _, name in rows]


def set_document_tags(
    session: Session,
    document_id: int,
    version_id: int | None,
    tags_by_source: Mapping[str, Sequence[str]],
) -> bool:
    """Replace a document's tag rows of the given sources and report changes.

    Args:
        session: An open session/transaction; the caller commits.
        document_id: The document to tag.
        version_id: The version the tags were observed on (stored on the rows).
        tags_by_source: Mapping from source (``folder``, ``frontmatter`` or
            ``hashtag``) to tag names. Rows of exactly these sources are
            replaced; ``llm``/``manual`` rows are never touched. Names are
            normalized and alias-resolved before insertion.

    Returns:
        True when the document's accepted tag set changed.

    Raises:
        ValueError: When the document does not exist.
    """
    document = session.get(Document, document_id)
    if document is None:
        raise ValueError(f"document {document_id} does not exist")
    workspace_id = document.workspace_id
    before = set(accepted_tags(session, document_id))
    sources = [str(source) for source in tags_by_source]
    if sources:
        session.execute(
            delete(DocumentTag).where(
                DocumentTag.document_id == document_id, DocumentTag.source.in_(sources)
            )
        )
    for source, raw_tags in tags_by_source.items():
        seen: set[int] = set()
        for raw in raw_tags:
            name = normalize_tag(raw)
            if not name:
                continue
            tag_id = ensure_tag(session, workspace_id, name)
            if tag_id in seen:
                continue
            seen.add(tag_id)
            session.execute(
                pg_insert(DocumentTag)
                .values(
                    document_id=document_id,
                    tag_id=tag_id,
                    source=str(source),
                    status="accepted",
                    document_version_id=version_id,
                )
                .on_conflict_do_nothing(index_elements=["document_id", "tag_id", "source"])
            )
    after = set(accepted_tags(session, document_id))
    return before != after


def merge_tags(session: Session, workspace_id: int, from_name: str, into_name: str) -> int:
    """Merge one tag into another: move rows, alias the name, drop the tag.

    Args:
        session: An open session/transaction; the caller commits.
        workspace_id: The workspace the vocabulary belongs to.
        from_name: The tag to merge away. Its ``document_tag`` rows move to the
            target (rows that would collide with an existing target
            ``(document, tag, source)`` row are skipped), a ``tag_alias`` from
            it to the target is created and the tag is deleted.
        into_name: The surviving tag; created when it does not exist yet.

    Returns:
        The number of ``document_tag`` rows moved to the target.
    """
    from_key = normalize_tag(from_name)
    into_key = normalize_tag(into_name)
    if not from_key or not into_key:
        raise ValueError("tag names must not be empty after normalization")
    from_resolved = resolve_alias(session, workspace_id, from_key)
    into_resolved = resolve_alias(session, workspace_id, into_key)
    if from_resolved == into_resolved:
        return 0
    from_tag = session.execute(
        select(Tag).where(Tag.workspace_id == workspace_id, Tag.name == from_resolved)
    ).scalar_one_or_none()
    into_id = ensure_tag(session, workspace_id, into_resolved)
    if from_tag is None:
        _upsert_alias(session, workspace_id, from_key, into_id)
        logger.info("merge %r -> %r: source tag did not exist; alias created", from_key, into_key)
        return 0
    target = aliased(DocumentTag)
    moved = int(
        session.execute(
            select(func.count())
            .select_from(DocumentTag)
            .where(
                DocumentTag.tag_id == from_tag.id,
                ~exists(
                    select(literal(1)).where(
                        target.document_id == DocumentTag.document_id,
                        target.source == DocumentTag.source,
                        target.tag_id == into_id,
                    )
                ),
            )
        ).scalar_one()
    )
    session.execute(
        pg_insert(DocumentTag)
        .from_select(
            [
                "document_id",
                "tag_id",
                "source",
                "confidence",
                "status",
                "document_version_id",
                "reason",
                "created_at",
            ],
            select(
                DocumentTag.document_id,
                literal(into_id),
                DocumentTag.source,
                DocumentTag.confidence,
                DocumentTag.status,
                DocumentTag.document_version_id,
                DocumentTag.reason,
                DocumentTag.created_at,
            ).where(DocumentTag.tag_id == from_tag.id),
        )
        .on_conflict_do_nothing(index_elements=["document_id", "tag_id", "source"])
    )
    _upsert_alias(session, workspace_id, from_key, into_id)
    session.execute(delete(Tag).where(Tag.id == from_tag.id))
    logger.info("merge %r -> %r: moved %d document tag row(s)", from_key, into_key, moved)
    return moved


def resolve_tag_names(session: Session, workspace_id: int, names: Sequence[str]) -> list[str]:
    """Normalize tag filter values and resolve aliases (query-time filters).

    Args:
        session: An open session/transaction.
        workspace_id: The workspace the vocabulary belongs to; ``None`` values
            are handled by the caller.
        names: Raw filter values (``--tag``, inline ``#tag``).

    Returns:
        The normalized, de-duplicated names (first occurrence kept) with every
        alias replaced by its canonical tag name.
    """
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in names:
        name = normalize_tag(raw)
        if name and name not in seen:
            seen.add(name)
            normalized.append(name)
    if not normalized:
        return []
    alias_rows = session.execute(
        select(TagAlias.alias_norm, Tag.name)
        .join(Tag, Tag.id == TagAlias.tag_id)
        .where(TagAlias.workspace_id == workspace_id, TagAlias.alias_norm.in_(normalized))
    ).all()
    targets = {str(alias): str(name) for alias, name in alias_rows}
    return [targets.get(name, name) for name in normalized]


def document_ids_with_tag(session: Session, workspace_id: int, name: str) -> list[int]:
    """Return the ids of the documents tagged with a (possibly aliased) tag.

    Args:
        session: An open session/transaction.
        workspace_id: The workspace the vocabulary belongs to.
        name: A tag name in any spelling; normalized, then alias-resolved.

    Returns:
        The document ids with a ``document_tag`` row of the resolved tag.
    """
    resolved = resolve_alias(session, workspace_id, name)
    return list(
        session.execute(
            select(DocumentTag.document_id)
            .join(Tag, Tag.id == DocumentTag.tag_id)
            .where(Tag.workspace_id == workspace_id, Tag.name == resolved)
            .order_by(DocumentTag.document_id)
        ).scalars()
    )


def tag_summaries(
    session: Session,
    workspace_id: int,
    prefix: str | None = None,
    source: str | None = None,
) -> list[TagSummary]:
    """Summarize the tag vocabulary with accepted-usage counts.

    Args:
        session: An open session/transaction.
        workspace_id: The workspace the vocabulary belongs to.
        prefix: Only tags whose name starts with this prefix (escaped literal).
        source: Only tags applied with this source (accepted on at least one
            document); the counts and sources then reflect the filter.
            Without it, unused tags are listed too (count 0).

    Returns:
        The summaries ordered by tag name.
    """
    join_conditions = [
        DocumentTag.tag_id == Tag.id,
        DocumentTag.status == "accepted",
    ]
    if source is not None:
        join_conditions.append(DocumentTag.source == source)
    stmt = (
        select(
            Tag.name,
            Tag.kind,
            func.count(func.distinct(DocumentTag.document_id)),
            func.array_agg(func.distinct(DocumentTag.source)),
        )
        .join(DocumentTag, and_(*join_conditions), isouter=source is None)
        .where(Tag.workspace_id == workspace_id)
        .group_by(Tag.id, Tag.name, Tag.kind)
        .order_by(Tag.name)
    )
    if prefix:
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        stmt = stmt.where(Tag.name.like(escaped + "%", escape="\\"))
    summaries: list[TagSummary] = []
    for name, kind, count, sources in session.execute(stmt).all():
        summaries.append(
            TagSummary(
                name=str(name),
                kind=str(kind),
                document_count=int(count or 0),
                sources=sorted(str(item) for item in (sources or []) if item),
            )
        )
    return summaries


def document_tag_details(session: Session, document_id: int) -> list[TagDetail]:
    """List a document's tag rows with source and review status.

    Args:
        session: An open session/transaction.
        document_id: The document whose rows to list.

    Returns:
        The rows ordered by source then tag name.
    """
    rows = session.execute(
        select(Tag.name, Tag.kind, DocumentTag.source, DocumentTag.status, DocumentTag.confidence)
        .join(DocumentTag, DocumentTag.tag_id == Tag.id)
        .where(DocumentTag.document_id == document_id)
        .order_by(DocumentTag.source, Tag.name)
    ).all()
    return [
        TagDetail(
            name=str(name),
            kind=str(kind),
            source=str(source),
            status=str(status),
            confidence=float(confidence),
        )
        for name, kind, source, status, confidence in rows
    ]


def _upsert_alias(session: Session, workspace_id: int, alias_norm: str, tag_id: int) -> None:
    """Point ``alias_norm`` at ``tag_id``, replacing an existing mapping."""
    session.execute(
        pg_insert(TagAlias)
        .values(workspace_id=workspace_id, alias_norm=alias_norm, tag_id=tag_id)
        .on_conflict_do_update(
            index_elements=["workspace_id", "alias_norm"], set_={"tag_id": tag_id}
        )
    )
