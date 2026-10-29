"""KG entity store: reversible merges, unmerge and entity queries (task 41).

:func:`merge_entities` folds one entity into another (aliases copied, mentions
and relations repointed, duplicates folded, outbox rows queued) while keeping
the dropped row reversible: :func:`unmerge_entity` restores it, records the
pair in ``kg_entity_distinct`` (so resolution never merges them again) and
re-queues resolution for every version that mentions the survivor. The query
helpers back the ``mynotes graph entity``/``entities`` CLI commands.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from mynotes.config import get_settings
from mynotes.db.models import (
    Chunk,
    Document,
    DocumentVersion,
    File,
    KgEntity,
    KgEntityAlias,
    KgEntityDistinct,
    KgMention,
    KgRelation,
    KgRelationEvidence,
)
from mynotes.index.outbox import enqueue_entity_deletes, enqueue_entity_upserts
from mynotes.jobs.queue import enqueue
from mynotes.jobs.registry import Priority
from mynotes.text.normalize import normalize_key

logger = logging.getLogger(__name__)

_TOP_DOCUMENTS = 5
_MAX_RESOLVE_GROUPS_PER_VERSION = 200


def variant_groups(entities: Sequence[tuple[int, str, str]]) -> list[tuple[int, list[int]]]:
    """Group entities whose names differ only by articles or generic type words.

    Args:
        entities: ``(id, type, norm_name)`` rows of active entities.

    Returns:
        ``(keep_id, drop_ids)`` per group of two or more: the entity whose
        ``norm_name`` already is the bare name survives (lowest id on ties),
        else the lowest id; the others are to be merged into it.
    """
    from mynotes.graph.resolve import entity_norm

    groups: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for entity_id, entity_type, norm_name in entities:
        groups.setdefault((entity_type, entity_norm(norm_name, entity_type)), []).append(
            (int(entity_id), norm_name)
        )
    result: list[tuple[int, list[int]]] = []
    for (_type, bare), members in groups.items():
        if len(members) < 2:
            continue
        members.sort()
        exact = [entity_id for entity_id, norm_name in members if norm_name == bare]
        keep = exact[0] if exact else members[0][0]
        result.append((keep, [entity_id for entity_id, _ in members if entity_id != keep]))
    return sorted(result)


def merge_name_variants(
    session: Session, workspace_id: int, *, dry_run: bool = False
) -> list[tuple[int, int]]:
    """Merge active entities that are type-word variants of one another.

    "Kanbanist plugin" and "Kanbanist" (same type) become one entity, the bare
    name surviving; see :func:`variant_groups`. Reversible per pair with
    :func:`unmerge_entity`.

    Args:
        session: An open session/transaction; rows are written but not committed.
        workspace_id: The workspace to clean.
        dry_run: Only report the pairs.

    Returns:
        The ``(keep_id, drop_id)`` pairs merged (or that would be merged).
    """
    rows = session.execute(
        select(KgEntity.id, KgEntity.type, KgEntity.norm_name).where(
            KgEntity.workspace_id == workspace_id, KgEntity.status == "active"
        )
    ).all()
    pairs = [(keep, drop) for keep, drops in variant_groups(rows) for drop in drops]
    if not dry_run:
        for keep, drop in pairs:
            merge_entities(session, keep_id=keep, drop_id=drop)
    return pairs


def merge_entities(session: Session, keep_id: int, drop_id: int) -> None:
    """Merge entity ``drop_id`` into ``keep_id`` (reversible via unmerge).

    Copies the drop entity's aliases and its canonical name onto the keep
    entity, repoints mentions and relations (evidence of colliding relation
    edges moves to the surviving edge), marks the drop entity ``merged`` and
    queues the outbox rows that update the entities index. Recomputes the
    current mention counts of both entities.

    Args:
        session: An open session/transaction; rows are written but not
            committed.
        keep_id: The surviving entity (conventionally the older id).
        drop_id: The entity folded into ``keep_id``.
    """
    if keep_id == drop_id:
        return
    keep = session.get(KgEntity, keep_id)
    drop = session.get(KgEntity, drop_id)
    if keep is None or drop is None:
        logger.warning("merge of missing entity pair (%s, %s) ignored", keep_id, drop_id)
        return
    if drop.status == "merged" and drop.merged_into_id == keep_id:
        return

    copied_norms = {drop.norm_name}
    for alias in session.execute(
        select(KgEntityAlias).where(KgEntityAlias.entity_id == drop_id)
    ).scalars():
        copied_norms.add(alias.alias_norm)
        _insert_alias(session, keep_id, alias.alias, alias.alias_norm)
    _insert_alias(session, keep_id, drop.canonical_name, drop.norm_name)

    session.execute(
        update(KgMention).where(KgMention.entity_id == drop_id).values(entity_id=keep_id)
    )
    relations = list(
        session.execute(
            select(KgRelation).where(
                or_(KgRelation.src_entity_id == drop_id, KgRelation.dst_entity_id == drop_id)
            )
        ).scalars()
    )
    for relation in relations:
        new_src = keep_id if relation.src_entity_id == drop_id else relation.src_entity_id
        new_dst = keep_id if relation.dst_entity_id == drop_id else relation.dst_entity_id
        duplicate_id = _colliding_relation_id(session, relation, new_src, new_dst)
        if duplicate_id is not None:
            session.execute(
                update(KgRelationEvidence)
                .where(KgRelationEvidence.relation_id == relation.id)
                .values(relation_id=duplicate_id)
            )
            session.execute(delete(KgRelation).where(KgRelation.id == relation.id))
        else:
            relation.src_entity_id = new_src
            relation.dst_entity_id = new_dst

    drop.status = "merged"
    drop.merged_into_id = keep_id
    drop.updated_at = datetime.now(UTC)
    keep.updated_at = datetime.now(UTC)
    _recount(session, [keep_id, drop_id])
    enqueue_entity_deletes(session, [drop_id])
    enqueue_entity_upserts(session, [keep_id])
    logger.info("merged kg entity %d into %d", drop_id, keep_id)


def unmerge_entity(session: Session, drop_id: int) -> int | None:
    """Undo a merge: restore the drop entity and record the pair as distinct.

    Restores the entity to ``active``, removes the aliases that the merge had
    copied onto the keep entity (the drop's own alias rows were kept, so the
    copied set is known), records ``kg_entity_distinct`` for the pair, queues
    index upserts for both entities and re-enqueues ``resolve_entities`` for
    every version that mentions the keep entity — the recomputation is
    idempotent and splits the mentions correctly thanks to the distinct pair.

    Args:
        session: An open session/transaction; rows are written but not
            committed.
        drop_id: The merged entity to restore.

    Returns:
        The keep entity id, or None when ``drop_id`` is not merged.
    """
    drop = session.get(KgEntity, drop_id)
    if drop is None or drop.status != "merged" or drop.merged_into_id is None:
        return None
    keep_id = int(drop.merged_into_id)

    drop_norms = {
        row
        for row in session.execute(
            select(KgEntityAlias.alias_norm).where(KgEntityAlias.entity_id == drop_id)
        ).scalars()
    }
    drop_norms.add(drop.norm_name)
    session.execute(
        delete(KgEntityAlias).where(
            KgEntityAlias.entity_id == keep_id, KgEntityAlias.alias_norm.in_(drop_norms)
        )
    )

    drop.status = "active"
    drop.merged_into_id = None
    drop.updated_at = datetime.now(UTC)
    session.execute(
        pg_insert(KgEntityDistinct)
        .values(a_id=min(keep_id, drop_id), b_id=max(keep_id, drop_id))
        .on_conflict_do_nothing(index_elements=["a_id", "b_id"])
    )
    _recount(session, [keep_id, drop_id])
    enqueue_entity_upserts(session, [keep_id, drop_id])

    version_ids = list(
        session.execute(
            select(KgMention.document_version_id).where(KgMention.entity_id == keep_id).distinct()
        ).scalars()
    )
    enqueued = 0
    for version_id in version_ids:
        if _enqueue_resolution(session, int(version_id)):
            enqueued += 1
    logger.info(
        "unmerged kg entity %d from %d; %d resolution job(s) re-enqueued",
        drop_id,
        keep_id,
        enqueued,
    )
    return keep_id


def _enqueue_resolution(session: Session, version_id: int) -> bool:
    """Queue one ``resolve_entities`` job with the version's current groups."""
    from mynotes.graph.extract import extractable_groups

    groups = extractable_groups(session, version_id)[:_MAX_RESOLVE_GROUPS_PER_VERSION]
    if not groups:
        return False
    return (
        enqueue(
            session,
            "resolve_entities",
            {
                "version_id": version_id,
                "groups": [
                    {
                        "chunk_ids": [str(chunk_id) for chunk_id in group.chunk_ids],
                        "content_hash": group.content_hash,
                    }
                    for group in groups
                ],
            },
            dedupe_key=f"resolve_entities:{version_id}",
            priority=Priority.RESOLVE_ENTITIES,
        )
        is not None
    )


def _colliding_relation_id(
    session: Session, relation: KgRelation, new_src: int, new_dst: int
) -> int | None:
    """An existing edge that ``relation`` would collide with after repointing."""
    if new_src == relation.src_entity_id and new_dst == relation.dst_entity_id:
        return None
    return session.execute(
        select(KgRelation.id).where(
            KgRelation.src_entity_id == new_src,
            KgRelation.dst_entity_id == new_dst,
            KgRelation.type == relation.type,
            KgRelation.attribute.is_not_distinct_from(relation.attribute),
            KgRelation.value.is_not_distinct_from(relation.value),
            KgRelation.id != relation.id,
        )
    ).scalar_one_or_none()


def _insert_alias(session: Session, entity_id: int, alias: str, alias_norm: str) -> None:
    """Copy one alias row onto ``entity_id``, ignoring an existing one."""
    session.execute(
        pg_insert(KgEntityAlias)
        .values(entity_id=entity_id, alias=alias, alias_norm=alias_norm)
        .on_conflict_do_nothing(index_elements=["entity_id", "alias_norm"])
    )


def _recount(session: Session, entity_ids: Sequence[int]) -> None:
    """Recompute ``mention_count_current`` and mark the entities' summaries stale."""
    if not entity_ids:
        return
    counts = dict(
        session.execute(
            select(KgMention.entity_id, func.count())
            .join(Chunk, Chunk.id == KgMention.chunk_id)
            .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(
                KgMention.entity_id.in_(entity_ids),
                DocumentVersion.is_latest.is_(True),
                Document.status == "current",
            )
            .group_by(KgMention.entity_id)
        ).all()
    )
    for entity_id in sorted(set(entity_ids)):
        session.execute(
            update(KgEntity)
            .where(KgEntity.id == entity_id)
            .values(
                mention_count_current=int(counts.get(entity_id, 0)),
                summary_stale=True,
                updated_at=datetime.now(UTC),
            )
        )


def list_entities(
    session: Session,
    workspace_id: int,
    *,
    type: str | None = None,
    limit: int = 50,
) -> list[KgEntity]:
    """The workspace's active entities with the most current mentions."""
    statement = (
        select(KgEntity)
        .where(KgEntity.workspace_id == workspace_id, KgEntity.status == "active")
        .order_by(KgEntity.mention_count_current.desc(), KgEntity.canonical_name, KgEntity.id)
        .limit(limit)
    )
    if type is not None:
        statement = statement.where(KgEntity.type == type)
    return list(session.execute(statement).scalars())


def find_entities(
    session: Session,
    ref: str,
    type: str | None = None,
    limit: int = 20,
    *,
    include_merged: bool = False,
) -> list[KgEntity]:
    """Resolve a name reference to candidate entities (PLAN.md §6.1 lookup).

    Tries the ``canonical_name`` first (ignoring case: entities are not told
    apart by it), then the normalized name or any alias, then trigram
    similarity on ``norm_name``; the first step with hits wins. Searches the
    workspace's active and merged entities, optionally filtered by type. A
    merged entity is returned as the live entity it was merged into (a name
    only its old record carries still resolves), so a name shared by a record
    and its merged duplicate gives one match; with ``include_merged`` the
    merged records are returned as they are, which ``graph unmerge`` needs.
    """
    matches = _find_entity_rows(session, ref, type, limit)
    if include_merged:
        return matches
    live: list[KgEntity] = []
    seen: set[int] = set()
    for entity in matches:
        target = entity if entity.status == "active" else _live_entity(session, entity)
        if target is not None and target.id not in seen:
            seen.add(int(target.id))
            live.append(target)
    return live


def _live_entity(session: Session, entity: KgEntity) -> KgEntity | None:
    """The active entity at the end of ``entity``'s ``merged_into_id`` chain, or None."""
    current = entity
    for _ in range(_MAX_MERGE_HOPS):
        if current.merged_into_id is None:
            return current if current.status == "active" else None
        current = session.get(KgEntity, int(current.merged_into_id))
        if current is None:
            return None
    logger.warning("merged_into chain from entity %s does not terminate", entity.id)
    return None


_MAX_MERGE_HOPS = 20


def _find_entity_rows(session: Session, ref: str, type: str | None, limit: int) -> list[KgEntity]:
    """The raw lookup of :func:`find_entities`: active and merged rows, first step with hits."""
    cleaned = ref.strip()
    if not cleaned:
        return []
    workspace_id = _workspace_id(session)
    if workspace_id is None:
        return []
    norm = normalize_key(cleaned)

    def query(extra: Any) -> list[KgEntity]:
        statement = (
            select(KgEntity)
            .where(
                KgEntity.workspace_id == workspace_id,
                KgEntity.status.in_(["active", "merged"]),
            )
            .where(extra)
            .order_by(KgEntity.id)
            .limit(limit)
        )
        if type is not None:
            statement = statement.where(KgEntity.type == type)
        return list(session.execute(statement).scalars())

    exact = query(func.lower(KgEntity.canonical_name) == cleaned.lower())
    if exact:
        return exact
    alias_ids = select(KgEntityAlias.entity_id).where(KgEntityAlias.alias_norm == norm)
    by_norm = query(or_(KgEntity.norm_name == norm, KgEntity.id.in_(alias_ids)))
    if by_norm:
        return by_norm
    score = func.similarity(KgEntity.norm_name, norm)
    statement = (
        select(KgEntity)
        .where(
            KgEntity.workspace_id == workspace_id,
            KgEntity.status.in_(["active", "merged"]),
            KgEntity.norm_name.op("%")(norm),
        )
        .order_by(score.desc(), KgEntity.id)
        .limit(limit)
    )
    if type is not None:
        statement = statement.where(KgEntity.type == type)
    return list(session.execute(statement).scalars())


def entity_card(session: Session, entity_id: int) -> dict[str, Any]:
    """The entity card shown by ``mynotes graph entity``.

    Args:
        session: An open session/transaction.
        entity_id: The entity id (merged entities are shown as-is, with
            ``merged_into_id`` set).

    Returns:
        ``id``/``name``/``type``/``status``, the summary-or-description text,
        aliases, active relation counts, the current mention count, and the
        top documents by current mentions.
    """
    entity = session.get(KgEntity, entity_id)
    if entity is None:
        raise ValueError(f"unknown kg entity {entity_id}")
    aliases = list(
        session.execute(
            select(KgEntityAlias.alias)
            .where(KgEntityAlias.entity_id == entity_id)
            .order_by(KgEntityAlias.alias)
        ).scalars()
    )
    relations_out = int(
        session.execute(
            select(func.count())
            .select_from(KgRelation)
            .where(KgRelation.src_entity_id == entity_id, KgRelation.status == "active")
        ).scalar_one()
    )
    relations_in = int(
        session.execute(
            select(func.count())
            .select_from(KgRelation)
            .where(KgRelation.dst_entity_id == entity_id, KgRelation.status == "active")
        ).scalar_one()
    )
    top_documents = [
        {
            "document_id": int(document_id),
            "title": title,
            "rel_path": rel_path,
            "mentions": int(count),
        }
        for document_id, title, rel_path, count in session.execute(
            select(Document.id, Document.title, File.rel_path, func.count())
            .select_from(KgMention)
            .join(Chunk, Chunk.id == KgMention.chunk_id)
            .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
            .join(Document, Document.id == DocumentVersion.document_id)
            .join(File, File.id == Document.current_file_id)
            .where(KgMention.entity_id == entity_id)
            .group_by(Document.id, Document.title, File.rel_path)
            .order_by(func.count().desc(), Document.id)
            .limit(_TOP_DOCUMENTS)
        ).all()
    ]
    return {
        "id": int(entity.id),
        "name": entity.canonical_name,
        "type": entity.type,
        "status": entity.status,
        "merged_into_id": entity.merged_into_id,
        "description": entity.summary if entity.summary else (entity.description or ""),
        "summary_stale": bool(entity.summary_stale),
        "mention_count_current": int(entity.mention_count_current),
        "aliases": aliases,
        "relations_out": relations_out,
        "relations_in": relations_in,
        "top_documents": top_documents,
    }


def _workspace_id(session: Session) -> int | None:
    """The configured workspace's id, or None when it does not exist."""
    from mynotes.tags.store import workspace_id_by_name

    return workspace_id_by_name(session, get_settings().workspace)
