"""Bare dates are values, not entities (improvement plan step 6).

The extractor used to emit dates as ``Milestone`` entities ("2023-05-20",
102 mentions, degree 73 on the real corpus): a hub that connects unrelated
notes and dominates expansion. A bare date carries no identity of its own;
what the note states is "X is scheduled for <date>", a fact *about X*. This
module folds such entities away:

- :func:`fold_dates` rewrites one cached extraction at resolution time: date
  entities are dropped and a relation whose target is a date becomes a
  self-relation of its source with the date as ``attribute``/``value``
  (``scheduled_for`` with ``attribute="date"`` unless the model gave one).
  Timelines and the graph-facts block already render attribute values.
- :func:`fold_date_entities` applies the same to entities already in the
  graph (``mynotes graph fold-dates``, nightly maintenance).
- :func:`hub_entities` lists the highest-degree entities for review
  (``mynotes graph hubs``).

Because the fold runs on cached extractions, the matching prompt rule did not
bump ``EXTRACTOR_VERSION``: old and new extractions converge to the same graph.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.orm import Session

from mynotes.db.models import KgEntity, KgMention, KgRelation, KgRelationEvidence
from mynotes.graph.extract import ExtractedRelation, Extraction
from mynotes.index.outbox import enqueue_entity_deletes
from mynotes.text.normalize import normalize_key

logger = logging.getLogger(__name__)

DATE_ATTRIBUTE = "date"
"""Attribute name given to a folded date when the model named none."""

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december|"
    "janeiro|fevereiro|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro|"
    "jan|feb|fev|mar|apr|abr|mai|jun|jul|aug|ago|sep|set|oct|out|nov|dec|dez"
)
_BARE_DATE_RE = re.compile(
    r"^(?:"
    r"(?:19|20)\d{2}(?:-\d{1,2}(?:-\d{1,2})?)?"  # 2023, 2023-05, 2023-05-20
    r"|\d{1,2}[/.-]\d{1,2}[/.-](?:\d{2}|(?:19|20)\d{2})"  # 20/05/2023, 20.05.23
    r"|(?:19|20)\d{2}[/.]\d{1,2}[/.]\d{1,2}"  # 2023/05/20
    r"|(?:\d{1,2}(?: de)? )?(?:"
    + _MONTHS
    + r")\.?(?:(?: de)?,? (?:19|20)\d{2})?"  # 17 de junho de 2025
    r"|(?:q[1-4]|h[12]|s[12]) ?(?:19|20)\d{2}"  # q3 2025
    r")$"
)


def is_bare_date(name: str) -> bool:
    """Whether an entity name is nothing but a date, month, quarter or year.

    Args:
        name: The surface name (any case or accents).
    """
    return bool(_BARE_DATE_RE.match(normalize_key(name)))


def fold_dates(extraction: Extraction) -> Extraction:
    """Drop bare-date entities and turn relations to them into attribute values.

    A relation ``X -type-> <date>`` becomes ``X -type-> X`` with
    ``attribute`` = the model's attribute or :data:`DATE_ATTRIBUTE` and
    ``value`` = the model's value or the date as written. Relations whose
    source is a date are dropped. Everything else is returned unchanged.

    Args:
        extraction: The validated model answer for one section group.

    Returns:
        The folded extraction (a new object; ``extraction`` is not modified).
    """
    date_names = {
        normalize_key(e.name): e.name for e in extraction.entities if is_bare_date(e.name)
    }
    if not date_names:
        return extraction
    entities = [e for e in extraction.entities if normalize_key(e.name) not in date_names]
    relations: list[ExtractedRelation] = []
    seen: set[tuple[str, str, str, str | None, str | None]] = set()
    for relation in extraction.relations:
        source_key = normalize_key(relation.source)
        target_key = normalize_key(relation.target)
        if source_key in date_names:
            continue
        if target_key in date_names:
            relation = relation.model_copy(
                update={
                    "target": relation.source,
                    "attribute": relation.attribute or DATE_ATTRIBUTE,
                    "value": relation.value or date_names[target_key],
                }
            )
        key = (
            source_key,
            normalize_key(relation.target),
            relation.type,
            relation.attribute,
            relation.value,
        )
        if key in seen:
            continue
        seen.add(key)
        relations.append(relation)
    return Extraction.model_construct(entities=entities, relations=relations)


@dataclass(frozen=True)
class FoldReport:
    """What :func:`fold_date_entities` did (or would do)."""

    entities: int = 0
    relations_folded: int = 0
    relations_dropped: int = 0
    mentions_removed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "entities": self.entities,
            "relations_folded": self.relations_folded,
            "relations_dropped": self.relations_dropped,
            "mentions_removed": self.mentions_removed,
        }


def date_entity_ids(session: Session, workspace_id: int) -> list[tuple[int, str]]:
    """The active entities of the workspace whose name is a bare date."""
    rows = session.execute(
        select(KgEntity.id, KgEntity.canonical_name, KgEntity.norm_name)
        .where(KgEntity.workspace_id == workspace_id, KgEntity.status == "active")
        .order_by(KgEntity.id)
    ).all()
    return [(int(i), str(name)) for i, name, norm in rows if _BARE_DATE_RE.match(str(norm))]


def fold_date_entities(session: Session, workspace_id: int, *, dry_run: bool = False) -> FoldReport:
    """Fold the workspace's existing bare-date entities into attribute values.

    For every relation whose target is a date entity, the evidence moves to
    the source's self-relation ``(src, src, type, attribute, value)`` (created
    when missing); relations whose source is a date are dropped; the entity
    is deleted (mentions and aliases cascade; entities previously merged into
    it go too) and removed from the entities index. Idempotent: a second run
    finds nothing.

    Args:
        session: An open session/transaction; rows are written but not committed.
        workspace_id: The workspace to clean.
        dry_run: Count without writing.
    """
    from mynotes.graph.resolve import _upsert_relation

    targets = date_entity_ids(session, workspace_id)
    if not targets:
        return FoldReport()
    ids = [entity_id for entity_id, _name in targets]
    names = dict(targets)
    mentions = int(
        session.execute(
            select(func.count()).select_from(KgMention).where(KgMention.entity_id.in_(ids))
        ).scalar_one()
    )
    relations = list(
        session.execute(
            select(KgRelation)
            .where(or_(KgRelation.src_entity_id.in_(ids), KgRelation.dst_entity_id.in_(ids)))
            .order_by(KgRelation.id)
        ).scalars()
    )
    folded = dropped = 0
    for relation in relations:
        src = int(relation.src_entity_id)
        dst = int(relation.dst_entity_id)
        if src in names or dst not in names:
            dropped += 1
            if not dry_run:
                session.execute(delete(KgRelation).where(KgRelation.id == relation.id))
            continue
        folded += 1
        if dry_run:
            continue
        spec = SimpleNamespace(
            type=relation.type,
            raw_label=relation.raw_label,
            description=relation.description,
            attribute=relation.attribute or DATE_ATTRIBUTE,
            value=relation.value or names[dst],
        )
        new_id = _upsert_relation(session, workspace_id, src, src, spec)
        session.execute(
            update(KgRelationEvidence)
            .where(KgRelationEvidence.relation_id == relation.id)
            .values(relation_id=new_id)
        )
        session.execute(delete(KgRelation).where(KgRelation.id == relation.id))
    if not dry_run:
        # Entities merged into a date (and chains of them) are ghosts of the same date:
        # they hold no mentions or relations and only block the delete through merged_into_id.
        ghosts: list[int] = []
        frontier = list(ids)
        while frontier:
            frontier = [
                int(i)
                for (i,) in session.execute(
                    select(KgEntity.id).where(
                        KgEntity.merged_into_id.in_(frontier), KgEntity.id.not_in(ids + ghosts)
                    )
                ).all()
            ]
            ghosts.extend(frontier)
        for layer in (ghosts[::-1], ids):
            if layer:
                session.execute(delete(KgEntity).where(KgEntity.id.in_(layer)))
        enqueue_entity_deletes(session, [*ghosts, *ids])
        logger.info(
            "folded %d date entities: %d relations to attributes, %d dropped, %d mentions removed",
            len(ids),
            folded,
            dropped,
            mentions,
        )
    return FoldReport(
        entities=len(ids),
        relations_folded=folded,
        relations_dropped=dropped,
        mentions_removed=mentions,
    )


def hub_entities(
    session: Session, workspace_id: int, *, entity_type: str | None = None, limit: int = 20
) -> list[dict[str, Any]]:
    """The highest-degree active entities (distinct active relations), for review.

    Args:
        session: An open session.
        workspace_id: The workspace.
        entity_type: Restrict to one entity type.
        limit: Rows to return.

    Returns:
        ``{"id", "type", "name", "mentions", "degree"}`` rows, highest degree first.
    """
    degree = func.count(KgRelation.id.distinct()).label("degree")
    statement = (
        select(
            KgEntity.id,
            KgEntity.type,
            KgEntity.canonical_name,
            KgEntity.mention_count_current,
            degree,
        )
        .join(
            KgRelation,
            or_(KgRelation.src_entity_id == KgEntity.id, KgRelation.dst_entity_id == KgEntity.id),
        )
        .where(
            KgEntity.workspace_id == workspace_id,
            KgEntity.status == "active",
            KgRelation.status == "active",
        )
        .group_by(KgEntity.id)
        .order_by(degree.desc(), KgEntity.mention_count_current.desc(), KgEntity.id)
        .limit(limit)
    )
    if entity_type:
        statement = statement.where(KgEntity.type == entity_type)
    return [
        {"id": int(i), "type": str(t), "name": str(n), "mentions": int(m or 0), "degree": int(d)}
        for i, t, n, m, d in session.execute(statement).all()
    ]
