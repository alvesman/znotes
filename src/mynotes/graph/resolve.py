"""Entity resolution rules and the per-group resolver (PLAN.md §8.4, task 41).

Turns one cached LLM :class:`~mynotes.graph.extract.Extraction` into canonical
graph rows. Candidate matching runs within the workspace and the entity type:
exact normalized name or alias first, then trigram similarity on ``norm_name``
(pg_trgm), then k-NN over the entities index (cosine). :func:`decide_match`
combines the three signals into a :class:`MatchDecision`; borderline cases
create the entity immediately and queue an LLM adjudication against the
candidate (:func:`adjudicate_pairs`), which either merges the two records or
records them as distinct in ``kg_entity_distinct`` — a recorded pair never
links or adjudicates again.

Every step is idempotent: the resolver deletes the group's mentions and
relation evidence before re-inserting them, so re-running a group (crash
recovery, unmerge recomputation, backfill) produces identical rows.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from mynotes.config import get_settings
from mynotes.db.models import (
    Chunk,
    Document,
    DocumentVersion,
    KgEntity,
    KgEntityAlias,
    KgEntityDistinct,
    KgExtractionCache,
    KgMention,
    KgRelation,
    KgRelationEvidence,
)
from mynotes.embed.cache import ensure_embeddings
from mynotes.embed.client import Embedder
from mynotes.graph.dates import fold_dates
from mynotes.graph.extract import ExtractedEntity, Extraction
from mynotes.graph.lexical import (
    EXTRACTOR_MENTION_CONFIDENCE,
    LexicalParams,
    link_lexical_mentions,
)
from mynotes.graph.ontology import EXTRACTOR_VERSION
from mynotes.index.outbox import enqueue_entity_upserts
from mynotes.llm.chat import ChatMessage
from mynotes.llm.structured import complete_structured
from mynotes.text.normalize import normalize_key

logger = logging.getLogger(__name__)

TRIGRAM_THRESHOLD = 0.6
"""Minimum pg_trgm ``similarity(norm_name, name)`` for a fuzzy name match."""

COSINE_THRESHOLD = 0.88
"""Minimum cosine similarity for an embedding match on the entities index."""

MERGE_CONFIDENCE = 0.8
"""Minimum adjudication confidence to merge; below, the pair is recorded distinct."""

MAX_ADJUDICATE_PAIRS_PER_JOB = 20
"""Maximum pairs per ``adjudicate_entities`` job (the enqueue-side batch size)."""

_KNN_K = 5
_MAX_CHAIN_HOPS = 10
_MAX_MENTION_EXCERPTS = 2
_EXCERPT_RADIUS = 80


@dataclass(frozen=True)
class MatchDecision:
    """The resolution decision for one extracted entity.

    Attributes:
        action: ``link`` (exact match), ``link_add_alias`` (strong fuzzy +
            embedding agreement), ``new`` (no usable candidate) or
            ``new_adjudicate`` (borderline: create the entity now, adjudicate
            later).
        entity_id: The entity to link to, for ``link``/``link_add_alias``.
        candidate_id: The pre-existing entity to adjudicate against, for
            ``new_adjudicate``.
        trigram: The best trigram similarity seen (None when no candidate).
        cosine: The best k-NN cosine seen (None when no candidate).
    """

    action: Literal["link", "link_add_alias", "new", "new_adjudicate"]
    entity_id: int | None = None
    candidate_id: int | None = None
    trigram: float | None = None
    cosine: float | None = None


def decide_match(
    *,
    exact_id: int | None,
    best_trgm: tuple[int, float] | None,
    best_knn: tuple[int, float] | None,
    distinct_pairs: set[tuple[int, int]],
) -> MatchDecision:
    """Combine the candidate signals into one :class:`MatchDecision`.

    Args:
        exact_id: Entity found by normalized-name/alias equality (already
            resolved through ``merged_into`` chains), or None.
        best_trgm: ``(entity_id, similarity)`` of the best trigram candidate,
            or None. Counts as passing at :data:`TRIGRAM_THRESHOLD`.
        best_knn: ``(entity_id, cosine)`` of the best embedding candidate, or
            None. Counts as passing at :data:`COSINE_THRESHOLD`.
        distinct_pairs: Recorded "these are different" pairs from
            ``kg_entity_distinct``, normalized ``a_id < b_id``.

    Returns:
        The decision. An exact match always links. When both fuzzy signals
        pass and agree, the surface name is added as an alias; when only one
        passes or they disagree, the entity is created now and adjudicated
        against the passing candidate — unless the two fuzzy candidates are
        already recorded as distinct, in which case the pair is settled and no
        adjudication happens (plain ``new``).
    """
    trgm_id, trgm_score = _passing(best_trgm, TRIGRAM_THRESHOLD)
    knn_id, knn_score = _passing(best_knn, COSINE_THRESHOLD)

    if exact_id is not None:
        return MatchDecision("link", entity_id=exact_id, trigram=trgm_score, cosine=knn_score)
    if trgm_id is not None and knn_id is not None:
        if trgm_id == knn_id:
            return MatchDecision(
                "link_add_alias", entity_id=trgm_id, trigram=trgm_score, cosine=knn_score
            )
        if (min(trgm_id, knn_id), max(trgm_id, knn_id)) in distinct_pairs:
            return MatchDecision("new", trigram=trgm_score, cosine=knn_score)
        return MatchDecision(
            "new_adjudicate", candidate_id=trgm_id, trigram=trgm_score, cosine=knn_score
        )
    if trgm_id is not None or knn_id is not None:
        candidate = trgm_id if trgm_id is not None else knn_id
        return MatchDecision(
            "new_adjudicate", candidate_id=candidate, trigram=trgm_score, cosine=knn_score
        )
    return MatchDecision("new", trigram=trgm_score, cosine=knn_score)


def _passing(best: tuple[int, float] | None, threshold: float) -> tuple[int | None, float | None]:
    """Split a best-candidate tuple into ``(id, score)``, id only when it passes."""
    if best is None:
        return None, None
    entity_id, score = best
    return (entity_id, score) if score >= threshold else (None, score)


def entity_text(name: str, type: str, description: str) -> str:
    """The entity's embedding input: ``"{name} ({type}): {description}"``."""
    return f"{name} ({type}): {description}".strip()


def link_task() -> str:
    """The embedding task for linking queries: ``entity_linking`` when configured."""
    configured = get_settings().models.embedding.tasks
    return "entity_linking" if "entity_linking" in configured else "retrieval"


def resolve_merged(session: Session, entity_id: int) -> int:
    """Follow ``merged_into_id`` links to the entity a mention should point at."""
    current = entity_id
    for _ in range(_MAX_CHAIN_HOPS):
        target = session.execute(
            select(KgEntity.merged_into_id).where(KgEntity.id == current)
        ).scalar_one_or_none()
        if target is None:
            return current
        current = int(target)
    logger.warning("merged_into chain from entity %s does not terminate", entity_id)
    return current


_TYPE_WORDS = (
    "plugin",
    "plug-in",
    "plugins",
    "tool",
    "tools",
    "ferramenta",
    "ferramentas",
    "extension",
    "extensao",
    "extensoes",
    "library",
    "biblioteca",
    "app",
    "aplicacao",
    "framework",
)
"""Generic type words that notes attach to a name ("the Kanbanist plugin", "ferramenta
Tagloom"); they are stripped so the mention and the entity's own note resolve together."""
_ARTICLES = ("the", "o", "a", "os", "as", "um", "uma")


def entity_norm(name: str, entity_type: str | None = None) -> str:
    """The normalized entity name used for exact matching.

    :func:`normalize_key` (accents/case folded, whitespace collapsed), then
    leading articles and leading/trailing generic type words are removed, as
    long as something remains: ``"the Kanbanist plugin"`` and ``"Kanbanist"``
    both give ``"kanbanist"``; ``"plugin"`` alone stays ``"plugin"``. People's
    names are never stripped ("Rita Ferramenta" is a surname, not a tool).

    Args:
        name: The surface name from the extraction.
        entity_type: The entity type; ``"Person"`` disables the stripping.

    Returns:
        The normalized name (never empty when ``name`` is not).
    """
    words = normalize_key(name).split()
    if entity_type == "Person":
        return " ".join(words)
    stripped = list(words)

    def bare(word: str) -> str:
        return word.strip("()[]{}")

    while len(stripped) > 1 and bare(stripped[0]) in _ARTICLES:
        stripped = stripped[1:]
    while len(stripped) > 1 and bare(stripped[0]) in _TYPE_WORDS:
        stripped = stripped[1:]
    while len(stripped) > 1 and bare(stripped[-1]) in _TYPE_WORDS:
        stripped = stripped[:-1]
    return " ".join(stripped) if stripped else " ".join(words)


def _norm_variants(norm: str) -> list[str]:
    """``norm`` plus the type-word spellings older rows may still carry."""
    return [
        norm,
        *(f"{norm} {word}" for word in _TYPE_WORDS),
        *(f"{word} {norm}" for word in _TYPE_WORDS),
    ]


def create_entity(
    session: Session, workspace_id: int, extracted: ExtractedEntity, norm: str
) -> tuple[int, bool]:
    """Insert a new entity, or link to the one a concurrent job inserted first.

    The insert runs in a savepoint; the partial unique index on active
    ``(workspace_id, type, norm_name)`` (migration 0004) rejects a duplicate,
    in which case the winner's row is re-read and returned.

    Args:
        session: An open session/transaction.
        workspace_id: The workspace.
        extracted: The extracted entity (name, type, description).
        norm: Its :func:`entity_norm`.

    Returns:
        ``(entity_id, created)``; ``created`` is False when linked to the winner.
    """
    entity = KgEntity(
        workspace_id=workspace_id,
        type=extracted.type,
        canonical_name=extracted.name,
        norm_name=norm,
        description=extracted.description or None,
        summary_stale=True,
    )
    try:
        with session.begin_nested():
            session.add(entity)
            session.flush()
    except IntegrityError:
        logger.debug("entity %r inserted concurrently; linking to it", extracted.name)
        session.expire_all()
        existing = exact_entity_id(session, workspace_id, extracted.type, norm)
        if existing is None:
            raise
        return existing, False
    return int(entity.id), True


def exact_entity_id(session: Session, workspace_id: int, entity_type: str, norm: str) -> int | None:
    """The workspace/type entity whose normalized name or alias equals ``norm``.

    ``norm`` is an :func:`entity_norm`; rows created before type words were
    stripped are matched through their variants. Merged entities resolve to
    the end of their ``merged_into_id`` chain; a tie (a data error) resolves to
    the lowest id, after preferring the exact spelling.
    """
    variants = _norm_variants(norm)
    rows = list(
        session.execute(
            select(KgEntity.id, KgEntity.norm_name)
            .where(
                KgEntity.workspace_id == workspace_id,
                KgEntity.type == entity_type,
                KgEntity.norm_name.in_(variants),
            )
            .order_by(KgEntity.id)
        ).all()
    )
    if rows:
        exact = [int(entity_id) for entity_id, norm_name in rows if norm_name == norm]
        return resolve_merged(session, exact[0] if exact else int(rows[0][0]))
    rows = list(
        session.execute(
            select(KgEntity.id)
            .join(KgEntityAlias, KgEntityAlias.entity_id == KgEntity.id)
            .where(
                KgEntity.workspace_id == workspace_id,
                KgEntity.type == entity_type,
                KgEntityAlias.alias_norm.in_(variants),
            )
            .limit(2)
        ).scalars()
    )
    if not rows:
        return None
    return resolve_merged(session, int(min(rows)))


def best_trigram_candidate(
    session: Session, workspace_id: int, entity_type: str, norm: str
) -> tuple[int, float] | None:
    """The most similar active ``norm_name`` in the workspace/type, via pg_trgm.

    The ``%`` operator lets Postgres use the GIN trigram index; thresholding
    the score is :func:`decide_match`'s job.
    """
    score = func.similarity(KgEntity.norm_name, norm)
    row = session.execute(
        select(KgEntity.id, score)
        .where(
            KgEntity.workspace_id == workspace_id,
            KgEntity.type == entity_type,
            KgEntity.status == "active",
            KgEntity.norm_name.op("%")(norm),
        )
        .order_by(score.desc(), KgEntity.id)
        .limit(1)
    ).first()
    return (int(row[0]), float(row[1])) if row is not None else None


def best_knn_candidate(
    client: Any, alias: str, workspace_id: int, entity_type: str, vector: list[float]
) -> tuple[int, float] | None:
    """The best cosine on the entities alias filtered by workspace and type.

    Args:
        client: The OpenSearch client.
        alias: The entities alias (``settings.opensearch.entities_alias``).
        workspace_id: Candidates must belong to this workspace.
        entity_type: Candidates must have this entity type.
        vector: The query embedding of the surface name.

    Returns:
        The top hit as ``(entity_id, cosine_score)``, or None when the index
        has no such entity.
    """
    body: dict[str, Any] = {
        "size": _KNN_K,
        "query": {
            "knn": {
                "embedding": {
                    "vector": vector,
                    "k": _KNN_K,
                    "filter": {
                        "bool": {
                            "filter": [
                                {"term": {"workspace_id": str(workspace_id)}},
                                {"term": {"type": entity_type}},
                            ]
                        }
                    },
                }
            }
        },
        "_source": False,
    }
    response = client.search(index=alias, body=body)
    hits = (response.get("hits") or {}).get("hits") or []
    if not hits:
        return None
    return int(hits[0]["_id"]), float(hits[0]["_score"])


def distinct_pairs_for(session: Session, entity_ids: set[int]) -> set[tuple[int, int]]:
    """The recorded distinct pairs touching any of ``entity_ids``."""
    if len(entity_ids) < 2:
        return set()
    rows = session.execute(
        select(KgEntityDistinct.a_id, KgEntityDistinct.b_id).where(
            or_(
                KgEntityDistinct.a_id.in_(entity_ids),
                KgEntityDistinct.b_id.in_(entity_ids),
            )
        )
    ).all()
    return {(int(a), int(b)) for a, b in rows}


@dataclass
class _EntityPlan:
    """One extracted entity's resolved state, for mention insertion."""

    entity_id: int
    name: str
    keys: frozenset[str]


def resolve_group(
    session: Session,
    *,
    version: DocumentVersion,
    document: Document,
    group: dict[str, Any],
    embedder: Embedder,
) -> list[tuple[int, int]]:
    """Resolve one section group's cached extraction into graph rows.

    Loads the cached :class:`Extraction` for the group's content hash (bare
    dates folded into attribute values, :mod:`mynotes.graph.dates`), deletes
    the group's existing mentions and relation evidence, then applies the
    resolution rules: entities are linked or created (aliases added, outbox
    rows queued, vectors cached for new entities), mentions are attached to
    the chunks whose normalized text contains the name or one of its aliases
    (the group's first chunk as fallback), known entities named verbatim in
    the chunks get lexical mentions (:mod:`mynotes.graph.lexical`, when
    ``graph.lexical_mentions`` is on), relations are upserted per
    ``(src, dst, type, attribute, value)`` with per-chunk evidence, and
    ``mention_count_current``/``summary_stale`` are refreshed for every
    touched entity.

    Args:
        session: An open session/transaction; rows are written but not
            committed.
        version: The document version the group belongs to.
        document: The version's document.
        group: ``{"chunk_ids": [str], "content_hash": str}``.
        embedder: The embedder used to cache new entities' vectors.

    Returns:
        The ``(new_id, candidate_id)`` adjudication pairs to enqueue (empty
        when the cached extraction is missing, an error row, or has no
        borderline matches).
    """
    cached = session.get(KgExtractionCache, (str(group["content_hash"]), EXTRACTOR_VERSION))
    if cached is None or "error" in cached.result:
        return []
    extraction = fold_dates(Extraction.model_validate(cached.result))
    chunk_ids = [UUID(value) for value in group["chunk_ids"]]
    session.execute(delete(KgMention).where(KgMention.chunk_id.in_(chunk_ids)))
    session.execute(delete(KgRelationEvidence).where(KgRelationEvidence.chunk_id.in_(chunk_ids)))
    chunks = list(
        session.execute(
            select(Chunk.id, Chunk.text).where(Chunk.id.in_(chunk_ids)).order_by(Chunk.ordinal)
        ).all()
    )
    norm_texts = [(chunk_id, normalize_key(text)) for chunk_id, text in chunks]

    plans: list[_EntityPlan] = []
    ids_by_norm: dict[str, int] = {}
    touched_ids: set[int] = set()
    pending_pairs: list[tuple[int, int]] = []
    new_texts: list[str] = []
    # Deterministic order: two groups creating the same new entities in opposite orders
    # would wait on each other's inserts (unique index) and deadlock.
    for extracted in sorted(
        extraction.entities, key=lambda e: (e.type, entity_norm(e.name, e.type), e.name)
    ):
        norm = entity_norm(extracted.name, extracted.type)
        decision = _decide_for(
            session, document.workspace_id, extracted.type, norm, extracted.name, embedder
        )
        if decision.action in ("link", "link_add_alias"):
            entity_id = int(decision.entity_id)
        else:
            entity_id, created = create_entity(session, document.workspace_id, extracted, norm)
            if created:
                new_texts.append(entity_text(extracted.name, extracted.type, extracted.description))
                enqueue_entity_upserts(session, [entity_id])
                if decision.action == "new_adjudicate" and decision.candidate_id is not None:
                    pending_pairs.append((entity_id, int(decision.candidate_id)))
        if decision.action == "link_add_alias":
            _insert_alias(session, entity_id, extracted.name, norm)
            enqueue_entity_upserts(session, [entity_id])
        for alias in extracted.aliases:
            alias_norm = entity_norm(alias, extracted.type)
            if alias_norm:
                _insert_alias(session, entity_id, alias, alias_norm)
        touched_ids.add(entity_id)
        ids_by_norm[norm] = entity_id
        ids_by_norm[normalize_key(extracted.name)] = entity_id
        keys = {
            norm,
            normalize_key(extracted.name),
            *(entity_norm(alias, extracted.type) for alias in extracted.aliases),
        }
        plans.append(_EntityPlan(entity_id=entity_id, name=extracted.name, keys=frozenset(keys)))
    if new_texts:
        ensure_embeddings(session, embedder, new_texts)

    mentioned: set[tuple[int, UUID]] = set()
    for plan in plans:
        keys = {key for key in plan.keys if key}
        matching = [chunk_id for chunk_id, text in norm_texts if any(key in text for key in keys)]
        targets = matching or ([norm_texts[0][0]] if norm_texts else [])
        for chunk_id in targets:
            mentioned.add((plan.entity_id, chunk_id))
            session.add(
                KgMention(
                    entity_id=plan.entity_id,
                    chunk_id=chunk_id,
                    document_version_id=version.id,
                    surface_form=plan.name,
                )
            )
    # Known entities named verbatim in the group's text but missed by the extractor
    # (mynotes.graph.lexical). Flushed first so the rarity bound sees this group's mentions.
    lexical = LexicalParams.from_settings(get_settings())
    if lexical is not None:
        session.flush()
        lexical_ids, _added = link_lexical_mentions(
            session,
            workspace_id=document.workspace_id,
            version_id=version.id,
            chunks=norm_texts,
            skip=mentioned,
            params=lexical,
        )
        touched_ids |= lexical_ids

    is_current = document.status == "current" and version.is_latest
    for relation in extraction.relations:
        src_id = ids_by_norm.get(entity_norm(relation.source)) or ids_by_norm.get(
            normalize_key(relation.source)
        )
        dst_id = ids_by_norm.get(entity_norm(relation.target)) or ids_by_norm.get(
            normalize_key(relation.target)
        )
        if src_id is None or dst_id is None:
            continue
        relation_id = _upsert_relation(session, document.workspace_id, src_id, dst_id, relation)
        session.add(
            KgRelationEvidence(
                relation_id=relation_id,
                chunk_id=_chunk_with_quote(norm_texts, relation.quote),
                document_version_id=version.id,
                doc_date=version.doc_date,
                quote=relation.quote or None,
                confidence=relation.confidence,
                is_current=is_current,
            )
        )

    _recount_mentions(session, touched_ids)
    return pending_pairs


def _decide_for(
    session: Session,
    workspace_id: int,
    entity_type: str,
    norm: str,
    name: str,
    embedder: Embedder,
) -> MatchDecision:
    """Run the candidate queries for one extracted name and decide."""
    exact = exact_entity_id(session, workspace_id, entity_type, norm)
    trgm: tuple[int, float] | None = None
    knn: tuple[int, float] | None = None
    if exact is None:
        trgm = best_trigram_candidate(session, workspace_id, entity_type, norm)
        knn = _knn_candidate(workspace_id, entity_type, name, embedder)
    candidates = {i for i in (exact, trgm and trgm[0], knn and knn[0]) if i is not None}
    pairs = distinct_pairs_for(session, candidates)
    return decide_match(exact_id=exact, best_trgm=trgm, best_knn=knn, distinct_pairs=pairs)


def _knn_candidate(
    workspace_id: int, entity_type: str, name: str, embedder: Embedder
) -> tuple[int, float] | None:
    """The best embedding candidate from the entities alias, or None on any error.

    Entity resolution must keep working while OpenSearch is down or the index
    does not exist yet, so every failure degrades to "no embedding signal".
    """
    from mynotes.index.opensearch import get_client

    try:
        client = get_client()
        vector = embedder.embed_query(name, task=link_task())
        alias = get_settings().opensearch.entities_alias
        return best_knn_candidate(client, alias, workspace_id, entity_type, vector)
    except Exception as exc:
        logger.warning("entity k-NN lookup unavailable: %s: %s", type(exc).__name__, exc)
        return None


def _insert_alias(session: Session, entity_id: int, alias: str, alias_norm: str) -> None:
    """Insert one alias row, ignoring one that already exists."""
    session.execute(
        pg_insert(KgEntityAlias)
        .values(entity_id=entity_id, alias=alias, alias_norm=alias_norm)
        .on_conflict_do_nothing(index_elements=["entity_id", "alias_norm"])
    )


def _upsert_relation(
    session: Session, workspace_id: int, src_id: int, dst_id: int, relation: Any
) -> int:
    """Upsert one ``kg_relation`` edge and return its id."""
    statement = (
        pg_insert(KgRelation)
        .values(
            workspace_id=workspace_id,
            src_entity_id=src_id,
            dst_entity_id=dst_id,
            type=relation.type,
            raw_label=relation.raw_label or None,
            description=relation.description or None,
            attribute=relation.attribute,
            value=relation.value,
        )
        .on_conflict_do_update(
            index_elements=["src_entity_id", "dst_entity_id", "type", "attribute", "value"],
            set_={
                "raw_label": relation.raw_label or None,
                "description": relation.description or None,
                "last_seen_at": func.now(),
                "status": "active",
            },
        )
        .returning(KgRelation.id)
    )
    return int(session.execute(statement).scalar_one())


def _chunk_with_quote(norm_texts: list[tuple[UUID, str]], quote: str) -> UUID:
    """The chunk whose normalized text contains the quote, else the first."""
    key = normalize_key(quote)
    if key:
        for chunk_id, text in norm_texts:
            if key in text:
                return chunk_id
    return norm_texts[0][0]


def _recount_mentions(session: Session, entity_ids: set[int]) -> None:
    """Recompute ``mention_count_current`` (extractor mentions in latest-version
    chunks of current documents) and mark summaries stale for the touched entities.

    Lexical mentions (:mod:`mynotes.graph.lexical`, confidence below 1) are
    evidence for expansion but not salience: the rarity weights and hub damping
    of query-time expansion were calibrated on extractor counts, and a name
    that the lexical pass finds in many chunks must not turn into a hub.
    """
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
                KgMention.confidence >= EXTRACTOR_MENTION_CONFIDENCE,
                DocumentVersion.is_latest.is_(True),
                Document.status == "current",
            )
            .group_by(KgMention.entity_id)
        ).all()
    )
    # Ascending id order: concurrent groups recounting shared entities must lock them in the
    # same order (the synthetic smoke test caught the deadlock).
    for entity_id in sorted(entity_ids):
        session.execute(
            update(KgEntity)
            .where(KgEntity.id == entity_id)
            .values(
                mention_count_current=int(counts.get(entity_id, 0)),
                summary_stale=True,
                updated_at=datetime.now(UTC),
            )
        )


class AdjudicationDecision(BaseModel):
    """One LLM verdict for a candidate pair."""

    pair: int = Field(ge=0)
    same: bool
    confidence: float = Field(default=1.0, ge=0, le=1)


class AdjudicationResult(BaseModel):
    """The validated adjudication answer for a batch of pairs."""

    decisions: list[AdjudicationDecision] = Field(default_factory=list)


@dataclass(frozen=True)
class _AdjudicationItem:
    """One pair prepared for the adjudication prompt."""

    index: int
    keep_id: int
    drop_id: int


def adjudicate_pairs(session: Session, pairs: Sequence[tuple[int, int]], chat: Any) -> int:
    """Ask the LLM whether each pair is one entity and merge or record distinct.

    Pairs already recorded in ``kg_entity_distinct`` or whose newer entity is
    already merged are skipped, so re-delivered jobs are safe. A ``same``
    verdict with ``confidence >=`` :data:`MERGE_CONFIDENCE` merges the older
    entity's record into the pair's survivor via
    :func:`mynotes.graph.store.merge_entities`; every other answered pair is
    recorded as distinct. Unanswered pairs are left undecided.

    Args:
        session: An open session/transaction; rows are written but not
            committed.
        pairs: ``(entity_a, entity_b)`` id pairs (order does not matter).
        chat: The chat model (normally ``get_chat("extract")``).

    Returns:
        How many pairs were decided (merged or recorded distinct).
    """
    from mynotes.graph.store import merge_entities

    items: list[_AdjudicationItem] = []
    seen: set[tuple[int, int]] = set()
    for id_a, id_b in pairs:
        keep_id, drop_id = min(int(id_a), int(id_b)), max(int(id_a), int(id_b))
        if keep_id == drop_id or (keep_id, drop_id) in seen:
            continue
        seen.add((keep_id, drop_id))
        if distinct_pairs_for(session, {keep_id, drop_id}):
            continue
        row = session.execute(
            select(KgEntity.status).where(KgEntity.id == drop_id)
        ).scalar_one_or_none()
        if row is None or row == "merged":
            continue
        items.append(
            _AdjudicationItem(
                index=len(items), keep_id=resolve_merged(session, keep_id), drop_id=drop_id
            )
        )
    if not items:
        return 0
    result = complete_structured(
        chat,
        build_adjudication_messages(session, items),
        AdjudicationResult,
        priority="background",
    )
    decided = 0
    for decision in result.decisions:
        if decision.pair < 0 or decision.pair >= len(items):
            logger.warning("adjudication answer references unknown pair %d", decision.pair)
            continue
        item = items[decision.pair]
        if decision.same and decision.confidence >= MERGE_CONFIDENCE:
            merge_entities(session, keep_id=item.keep_id, drop_id=item.drop_id)
        else:
            session.execute(
                pg_insert(KgEntityDistinct)
                .values(a_id=min(item.keep_id, item.drop_id), b_id=max(item.keep_id, item.drop_id))
                .on_conflict_do_nothing(index_elements=["a_id", "b_id"])
            )
        decided += 1
    return decided


def build_adjudication_messages(
    session: Session, items: Sequence[_AdjudicationItem]
) -> list[ChatMessage]:
    """Build the system and user messages for one adjudication call.

    Per pair, both records are listed with name, type, description (summary
    when present), aliases and up to two mention excerpts.
    """
    lines: list[str] = []
    for item in items:
        lines.append(f"Pair {item.index}:")
        for label, entity_id in (("A", item.keep_id), ("B", item.drop_id)):
            entity = session.get(KgEntity, entity_id)
            aliases = list(
                session.execute(
                    select(KgEntityAlias.alias)
                    .where(KgEntityAlias.entity_id == entity_id)
                    .order_by(KgEntityAlias.alias)
                ).scalars()
            )
            description = (entity.summary if entity.summary else entity.description) or ""
            lines.append(f"  {label}: {entity.canonical_name} ({entity.type})")
            if description:
                lines.append(f"     description: {description}")
            if aliases:
                lines.append("     aliases: " + ", ".join(aliases))
            excerpts = _mention_excerpts(session, entity_id)
            if excerpts:
                lines.append("     mentions: " + " | ".join(f'"{text}"' for text in excerpts))
    system = (
        "You compare knowledge-graph entity records from personal notes.\n"
        "Rules: base the decision on the names, descriptions and mention excerpts; "
        "abbreviations and spelling variants of the same thing are the same entity.\n"
        "Reply only with JSON for the schema: "
        '{"decisions": [{"pair": 0, "same": true, "confidence": 0.0}]}'
    )
    user = (
        "For each numbered pair decide whether records A and B refer to the same "
        "real-world entity.\n\n" + "\n".join(lines)
    )
    return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]


def _mention_excerpts(session: Session, entity_id: int) -> list[str]:
    """Up to two chunk excerpts around the entity's earliest mentions."""
    rows = session.execute(
        select(Chunk.text, KgMention.surface_form)
        .join(KgMention, KgMention.chunk_id == Chunk.id)
        .where(KgMention.entity_id == entity_id)
        .order_by(KgMention.id)
        .limit(_MAX_MENTION_EXCERPTS)
    ).all()
    excerpts: list[str] = []
    for text, surface in rows:
        position = text.casefold().find(surface.casefold())
        if position >= 0:
            start = max(0, position - _EXCERPT_RADIUS)
            end = min(len(text), position + len(surface) + _EXCERPT_RADIUS)
            excerpts.append(text[start:end])
        else:
            excerpts.append(text[: 2 * _EXCERPT_RADIUS])
    return excerpts
