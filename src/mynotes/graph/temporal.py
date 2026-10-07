"""Temporal validity of KG relations and entity timelines (PLAN.md §8.4, task 42).

An evidence row is *current* when its chunk belongs to the latest version of a
document with ``status == "current"``. :func:`relation_status` turns the
evidence facts of one relation into its ``(status, valid_from, valid_to)``:
current evidence keeps a relation ``active``; evidence that only survives in
superseded or older versions makes it ``historical`` (the successor's date
bounds it); evidence only from deleted documents leaves it ``stale`` (kept for
history, excluded by default).

:func:`refresh_document_relations` recomputes those columns for every relation
with evidence from one document (the ``kg_temporal_refresh`` job),
:func:`current_attribute_values` resolves attribute supersession for one source
entity (the newest current value wins and older values become ``historical``),
and :func:`timeline` renders an entity's or a project tag's temporal view.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import Session

from mynotes.db.models import (
    Chunk,
    DocRelation,
    Document,
    DocumentVersion,
    File,
    KgEntity,
    KgEntityAlias,
    KgMention,
    KgRelation,
    KgRelationEvidence,
)
from mynotes.jobs.queue import enqueue
from mynotes.jobs.registry import Priority
from mynotes.text.normalize import normalize_key

logger = logging.getLogger(__name__)

_ACTIVE_SUPERSEDES = ("auto", "confirmed")

_EPOCH = datetime.min.replace(tzinfo=UTC)


@dataclass(frozen=True)
class EvidenceFact:
    """The temporal inputs of one evidence row.

    Attributes:
        doc_date: The evidence's document date, or None when unknown.
        is_current: The stored ``kg_relation_evidence.is_current`` value.
        deleted: True when the owning document's status is ``deleted``.
    """

    doc_date: datetime | None
    is_current: bool
    deleted: bool


@dataclass(frozen=True)
class AttributeFact:
    """One candidate value for an ``(entity, attribute)`` pair.

    Attributes:
        relation_id: The relation holding the value.
        value: The attribute value.
        doc_date: The latest doc_date among the relation's *current* evidence
            (None when it has none); the race key for the current value.
        confidence: The evidence confidence used to break date ties.
        evidence_date: The latest doc_date among all evidence of any currency;
            the display date for history entries.
    """

    relation_id: int
    value: str
    doc_date: datetime | None
    confidence: float
    evidence_date: datetime | None = None


@dataclass
class AttributeValue:
    """The current value of one attribute plus its superseded history."""

    attribute: str
    value: str
    doc_date: datetime | None
    relation_id: int
    history: list[tuple[str, datetime | None]] = field(default_factory=list)


@dataclass
class TimelineItem:
    """One row of an entity or tag timeline."""

    date: datetime | None
    kind: Literal["relation", "attribute", "document"]
    text: str
    status: str
    document_path: str | None


def relation_status(
    evidence: Sequence[EvidenceFact], successor_dates: Sequence[datetime]
) -> tuple[str, datetime | None, datetime | None]:
    """Compute ``(status, valid_from, valid_to)`` from evidence facts.

    Args:
        evidence: Every evidence row of the relation. Evidence of deleted
            documents never counts as current, matching the stored
            ``is_current`` rule.
        successor_dates: The doc_dates of the successors that superseded the
            evidence: newer versions of the same documents (edited in place)
            and active supersedes successors.

    Returns:
        ``valid_from`` is the earliest evidence ``doc_date`` (None when no
        evidence is dated). Any current evidence from a living document means
        ``active`` with ``valid_to=None``. Otherwise evidence from living
        documents means ``historical``, with ``valid_to`` the earliest
        successor date when known and the latest evidence date otherwise. Only
        deleted (or no) evidence means ``stale`` with ``valid_to=None``.
    """
    dated = [fact.doc_date for fact in evidence if fact.doc_date is not None]
    valid_from = min(dated) if dated else None
    if any(fact.is_current and not fact.deleted for fact in evidence):
        return "active", valid_from, None
    if any(not fact.deleted for fact in evidence):
        if successor_dates:
            return "historical", valid_from, min(successor_dates)
        return "historical", valid_from, (max(dated) if dated else None)
    return "stale", valid_from, None


def pick_current_attribute(facts: Sequence[AttributeFact]) -> AttributeFact | None:
    """The fact whose current evidence is the newest; ties go to confidence.

    Only facts that still have current evidence (a non-None ``doc_date``) can
    win; facts without any are history candidates. A full tie is broken by the
    lowest relation id so the choice is deterministic.

    Args:
        facts: The candidate facts of one ``(entity, attribute)`` pair.

    Returns:
        The winning fact, or None when no fact has current evidence.
    """
    dated = [fact for fact in facts if fact.doc_date is not None]
    if not dated:
        return None

    def sort_key(fact: AttributeFact) -> tuple[datetime, float, int]:
        return (fact.doc_date, fact.confidence, -fact.relation_id)

    return max(dated, key=sort_key)


def enqueue_temporal_refresh(session: Session, document_id: int) -> None:
    """Queue a deduplicated ``kg_temporal_refresh`` job for one document.

    Args:
        session: An open session/transaction; the job row is written but not
            committed.
        document_id: The document whose relations need a temporal refresh.
    """
    enqueue(
        session,
        "kg_temporal_refresh",
        {"document_id": int(document_id)},
        dedupe_key=f"kg_temporal_refresh:{int(document_id)}",
        priority=Priority.MAINTENANCE,
    )


def evidence_relation_ids(session: Session, document_id: int) -> list[int]:
    """The ids of every relation evidenced by any version of a document, ascending."""
    return sorted(
        int(relation_id)
        for relation_id in session.execute(
            select(KgRelationEvidence.relation_id)
            .join(DocumentVersion, DocumentVersion.id == KgRelationEvidence.document_version_id)
            .where(DocumentVersion.document_id == document_id)
            .distinct()
        ).scalars()
    )


def lock_relations(session: Session, relation_ids: Iterable[int]) -> list[int]:
    """Take row locks on the given relations in ascending id order.

    ``kg_temporal_refresh`` jobs of different documents write the same shared
    relations; locking the whole set up front in one fixed order makes two
    such jobs queue behind each other instead of deadlocking.

    Returns:
        The locked ids, ascending (relations that no longer exist are skipped).
    """
    wanted = sorted(set(relation_ids))
    if not wanted:
        return []
    return [
        int(relation_id)
        for relation_id in session.execute(
            select(KgRelation.id)
            .where(KgRelation.id.in_(wanted))
            .order_by(KgRelation.id)
            .with_for_update()
        ).scalars()
    ]


def refresh_document_relations(session: Session, document_id: int) -> list[int]:
    """Recompute the temporal columns of every relation evidenced by a document.

    The relations are processed (and, when not already locked by the caller,
    locked) in ascending id order; see :func:`lock_relations`.

    Args:
        session: An open session/transaction; rows are written but not
            committed.
        document_id: The document whose evidence defines the relation set.

    Returns:
        The affected relation ids, ascending (empty when the document has no evidence).
    """
    relation_ids = lock_relations(session, evidence_relation_ids(session, document_id))
    for relation_id in relation_ids:
        refresh_relation(session, relation_id)
    return relation_ids


def refresh_relation(session: Session, relation_id: int) -> bool:
    """Recompute one relation's ``status``/``valid_from``/``valid_to``.

    Args:
        session: An open session/transaction; rows are written but not
            committed.
        relation_id: The relation to refresh.

    Returns:
        True when any of the three columns changed.
    """
    relation = session.get(KgRelation, relation_id)
    if relation is None:
        return False
    facts, version_ids = _evidence_facts(session, relation_id)
    status, valid_from, valid_to = relation_status(facts, _successor_dates(session, version_ids))
    changed = (relation.status, relation.valid_from, relation.valid_to) != (
        status,
        valid_from,
        valid_to,
    )
    relation.status = status
    relation.valid_from = valid_from
    relation.valid_to = valid_to
    if changed:
        logger.debug(
            "relation %d temporal status: %s (%s..%s)", relation_id, status, valid_from, valid_to
        )
    return changed


def current_attribute_values(session: Session, entity_id: int) -> dict[str, AttributeValue]:
    """Resolve attribute supersession for one source entity.

    For each ``(entity, attribute)`` among the entity's relations with a
    non-null attribute, the current value is the one whose *current* evidence
    has the latest ``doc_date`` (ties: highest evidence confidence); an
    attribute whose relations all lost their current evidence has no current
    value. The other values become ``history`` (newest first, dated by their
    latest evidence of any currency), and relations that were still
    ``active`` are demoted to ``historical`` with ``valid_to`` set to the
    current value's date, so a re-run is idempotent. ``historical`` and
    ``stale`` relations keep their status but feed the history.

    Args:
        session: An open session/transaction; rows are written but not
            committed.
        entity_id: The source entity.

    Returns:
        One :class:`AttributeValue` per attribute that still has a current
        value.
    """
    relations = list(
        session.execute(
            select(KgRelation)
            .where(
                KgRelation.src_entity_id == entity_id,
                KgRelation.attribute.is_not(None),
                KgRelation.status.in_(("active", "historical", "stale")),
            )
            .order_by(KgRelation.id)
        ).scalars()
    )
    facts_by_attribute: dict[str, list[AttributeFact]] = {}
    statuses: dict[int, str] = {}
    for relation in relations:
        doc_date, confidence, evidence_date = _attribute_evidence(session, int(relation.id))
        statuses[int(relation.id)] = str(relation.status)
        facts_by_attribute.setdefault(str(relation.attribute), []).append(
            AttributeFact(
                relation_id=int(relation.id),
                value=str(relation.value or ""),
                doc_date=doc_date,
                confidence=confidence,
                evidence_date=evidence_date,
            )
        )
    values: dict[str, AttributeValue] = {}
    for attribute, facts in facts_by_attribute.items():
        current = pick_current_attribute(facts)
        if current is None:
            continue
        others = [fact for fact in facts if fact.relation_id != current.relation_id]
        for fact in others:
            if statuses[fact.relation_id] != "active":
                continue
            session.execute(
                update(KgRelation)
                .where(KgRelation.id == fact.relation_id)
                .values(status="historical", valid_to=current.doc_date)
            )
        history = sorted(
            (
                (
                    fact.value,
                    fact.evidence_date if fact.evidence_date is not None else fact.doc_date,
                )
                for fact in others
            ),
            key=lambda item: (item[1] is not None, item[1] or _EPOCH),
            reverse=True,
        )
        values[attribute] = AttributeValue(
            attribute=attribute,
            value=current.value,
            doc_date=current.doc_date,
            relation_id=current.relation_id,
            history=history,
        )
    return values


def timeline(session: Session, entity_id: int) -> list[TimelineItem]:
    """The temporal view of one entity: evidence, attribute values, documents.

    Args:
        session: An open session/transaction.
        entity_id: The entity (merged entities are shown as-is).

    Returns:
        Items sorted by date (undated last): one per relation evidence row
        (``kind="relation"``, or ``kind="attribute"`` when the relation carries
        an attribute/value pair, with the relation's ``status`` marking
        superseded values), plus one per document that mentions the entity at
        its first mention date.

    Raises:
        ValueError: When the entity does not exist.
    """
    entity = session.get(KgEntity, entity_id)
    if entity is None:
        raise ValueError(f"unknown kg entity {entity_id}")
    items: list[TimelineItem] = []
    names = _entity_names(session, _relation_partner_ids(session, entity_id) | {entity_id})
    evidence_rows = session.execute(
        select(KgRelation, KgRelationEvidence, File.rel_path)
        .join(KgRelationEvidence, KgRelationEvidence.relation_id == KgRelation.id)
        .join(Chunk, Chunk.id == KgRelationEvidence.chunk_id)
        .join(Document, Document.id == Chunk.document_id)
        .outerjoin(File, File.id == Document.current_file_id)
        .where(or_(KgRelation.src_entity_id == entity_id, KgRelation.dst_entity_id == entity_id))
        .order_by(KgRelationEvidence.id)
    ).all()
    for relation, evidence, rel_path in evidence_rows:
        source, target = (
            names.get(int(relation.src_entity_id), "?"),
            names.get(int(relation.dst_entity_id), "?"),
        )
        if relation.attribute is not None:
            kind: Literal["relation", "attribute"] = "attribute"
            text = f"{relation.attribute}={relation.value}"
        else:
            kind = "relation"
            text = f"{source} –{relation.type}→ {target}"  # noqa: RUF001
        items.append(
            TimelineItem(
                date=evidence.doc_date,
                kind=kind,
                text=text,
                status=relation.status,
                document_path=rel_path,
            )
        )
    for document_id, title, rel_path, status, first_date in session.execute(
        select(
            Document.id,
            Document.title,
            File.rel_path,
            Document.status,
            func.coalesce(func.min(DocumentVersion.doc_date), Document.doc_date),
        )
        .select_from(KgMention)
        .join(Chunk, Chunk.id == KgMention.chunk_id)
        .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
        .join(Document, Document.id == DocumentVersion.document_id)
        .outerjoin(File, File.id == Document.current_file_id)
        .where(KgMention.entity_id == entity_id)
        .group_by(Document.id, Document.title, File.rel_path, Document.status)
    ).all():
        items.append(
            TimelineItem(
                date=first_date,
                kind="document",
                text=title or rel_path or f"document {document_id}",
                status=status,
                document_path=rel_path,
            )
        )
    items.sort(key=lambda item: (item.date is None, item.date or _EPOCH, item.kind, item.text))
    return items


def project_entity_id(session: Session, workspace_id: int, name: str) -> int | None:
    """The active Project entity whose name or alias equals ``name`` exactly.

    Args:
        session: An open session/transaction.
        workspace_id: The workspace to search.
        name: A project name as written after ``project:`` in a reference.

    Returns:
        The lowest matching entity id, or None.
    """
    norm = normalize_key(name)
    if not norm:
        return None
    alias_ids = select(KgEntityAlias.entity_id).where(KgEntityAlias.alias_norm == norm)
    return session.execute(
        select(KgEntity.id)
        .where(
            KgEntity.workspace_id == workspace_id,
            KgEntity.type == "Project",
            KgEntity.status == "active",
            or_(
                KgEntity.canonical_name == name,
                KgEntity.norm_name == norm,
                KgEntity.id.in_(alias_ids),
            ),
        )
        .order_by(KgEntity.id)
        .limit(1)
    ).scalar_one_or_none()


def tag_timeline(session: Session, workspace_id: int, tag_name: str) -> list[TimelineItem]:
    """The documents carrying one tag, by ``doc_date`` (fallback for ``project:``).

    Args:
        session: An open session/transaction.
        workspace_id: The workspace whose tag vocabulary applies.
        tag_name: A tag name in any spelling; normalized and alias-resolved.

    Returns:
        One document item per tagged document, oldest first (undated last).
    """
    from mynotes.tags.normalize import normalize_tag
    from mynotes.tags.store import document_ids_with_tag

    document_ids = document_ids_with_tag(session, workspace_id, normalize_tag(tag_name))
    if not document_ids:
        return []
    return [
        TimelineItem(
            date=document.doc_date,
            kind="document",
            text=document.title or rel_path or f"document {document.id}",
            status=document.status,
            document_path=rel_path,
        )
        for document, rel_path in session.execute(
            select(Document, File.rel_path)
            .outerjoin(File, File.id == Document.current_file_id)
            .where(Document.id.in_(document_ids))
            .order_by(Document.doc_date.nulls_last(), Document.id)
        ).all()
    ]


def _evidence_facts(session: Session, relation_id: int) -> tuple[list[EvidenceFact], list[int]]:
    """The evidence facts of one relation and their document version ids."""
    rows = session.execute(
        select(
            KgRelationEvidence.document_version_id,
            KgRelationEvidence.doc_date,
            KgRelationEvidence.is_current,
            Document.status,
        )
        .join(DocumentVersion, DocumentVersion.id == KgRelationEvidence.document_version_id)
        .join(Document, Document.id == DocumentVersion.document_id)
        .where(KgRelationEvidence.relation_id == relation_id)
    ).all()
    facts = [
        EvidenceFact(
            doc_date=row.doc_date,
            is_current=bool(row.is_current),
            deleted=row.status == "deleted",
        )
        for row in rows
    ]
    return facts, [int(row.document_version_id) for row in rows]


def _successor_dates(session: Session, version_ids: Sequence[int]) -> list[datetime]:
    """The dates of the successors that superseded the given evidence versions.

    A newer latest version of the same document (edited in place) and every
    active supersedes successor of the evidence documents contribute their
    latest version's ``doc_date``.
    """
    if not version_ids:
        return []
    evidence_versions = session.execute(
        select(DocumentVersion.document_id, DocumentVersion.is_latest).where(
            DocumentVersion.id.in_(version_ids)
        )
    ).all()
    document_ids = {int(row.document_id) for row in evidence_versions}
    latest_dates: dict[int, datetime | None] = dict(
        session.execute(
            select(DocumentVersion.document_id, DocumentVersion.doc_date).where(
                DocumentVersion.document_id.in_(document_ids),
                DocumentVersion.is_latest.is_(True),
            )
        ).all()
    )
    dates: set[datetime] = set()
    for row in evidence_versions:
        if not row.is_latest and latest_dates.get(int(row.document_id)) is not None:
            dates.add(latest_dates[int(row.document_id)])
    successor_ids = list(
        session.execute(
            select(DocRelation.src_document_id)
            .where(
                DocRelation.dst_document_id.in_(document_ids),
                DocRelation.type == "supersedes",
                DocRelation.status.in_(_ACTIVE_SUPERSEDES),
            )
            .distinct()
        ).scalars()
    )
    if successor_ids:
        for doc_date in session.execute(
            select(DocumentVersion.doc_date).where(
                DocumentVersion.document_id.in_(successor_ids),
                DocumentVersion.is_latest.is_(True),
            )
        ).scalars():
            if doc_date is not None:
                dates.add(doc_date)
    return sorted(dates)


def _attribute_evidence(
    session: Session, relation_id: int
) -> tuple[datetime | None, float, datetime | None]:
    """Summarize a relation's evidence dates for attribute supersession.

    Returns:
        ``(doc_date, confidence, evidence_date)``: the latest doc_date among
        the relation's current evidence with that date's highest confidence
        (the current-value race key), and the latest doc_date among all
        evidence of any currency (the history display date).
    """
    rows = session.execute(
        select(
            KgRelationEvidence.doc_date,
            KgRelationEvidence.confidence,
            KgRelationEvidence.is_current,
        ).where(KgRelationEvidence.relation_id == relation_id)
    ).all()
    dated_current = [
        (row.doc_date, float(row.confidence)) for row in rows if row.is_current and row.doc_date
    ]
    doc_date = max((date for date, _confidence in dated_current), default=None)
    confidence = max((value for date, value in dated_current if date == doc_date), default=0.0)
    evidence_date = max((row.doc_date for row in rows if row.doc_date is not None), default=None)
    return doc_date, confidence, evidence_date


def _relation_partner_ids(session: Session, entity_id: int) -> set[int]:
    """The ids at the other end of the entity's relations."""
    rows = session.execute(
        select(KgRelation.src_entity_id, KgRelation.dst_entity_id).where(
            or_(KgRelation.src_entity_id == entity_id, KgRelation.dst_entity_id == entity_id)
        )
    ).all()
    partners: set[int] = set()
    for src, dst in rows:
        if int(src) != entity_id:
            partners.add(int(src))
        if int(dst) != entity_id:
            partners.add(int(dst))
    return partners


def _entity_names(session: Session, entity_ids: Iterable[int]) -> dict[int, str]:
    """Map entity ids to their canonical names."""
    ids = list(entity_ids)
    if not ids:
        return {}
    return dict(
        session.execute(
            select(KgEntity.id, KgEntity.canonical_name).where(KgEntity.id.in_(ids))
        ).all()
    )
