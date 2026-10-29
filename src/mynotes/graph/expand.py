"""Query-time KG expansion: subgraph fetch, entity scoring, evidence chunks (task 43).

:func:`fetch_subgraph` walks ``kg_relation`` in both directions from the seed
entities with a cycle-safe recursive CTE (current relations by default, or the
relations valid at an ``as_of`` date for history intent), weighting every edge
by relation type and evidence count. :func:`score_entities` ranks the subgraph
either by one-hop neighbour scores (v1) or Personalized PageRank over the
undirected subgraph (v2, HippoRAG-style). :func:`evidence_chunks` turns the
top-scoring entities and edges into the concrete chunks that mention or
evidence them, restricted to the latest chunks of living documents so graph
expansion never resurfaces deleted or superseded content under current intent.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

import networkx as nx
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from mynotes.db.models import (
    Chunk,
    Document,
    DocumentVersion,
    KgEntity,
    KgMention,
    KgRelationEvidence,
)

logger = logging.getLogger(__name__)

RELATION_WEIGHTS: dict[str, float] = {
    "part_of": 1.0,
    "depends_on": 0.9,
    "uses": 0.8,
    "proposes": 0.9,
    "decided": 1.0,
    "owned_by": 0.7,
    "works_on": 0.7,
    "targets": 0.8,
    "has_requirement": 0.9,
    "has_risk": 0.8,
    "measured_by": 0.8,
    "scheduled_for": 0.8,
    "replaces": 1.0,
    "contradicts": 0.9,
    "related_to": 0.4,
}
"""Weight per normalized relation type (PLAN.md §8.2); unknown types fall back
to the ``related_to`` weight."""

DEFAULT_RELATION_WEIGHT = RELATION_WEIGHTS["related_to"]

_PPR_ALPHA = 0.85
"""Damping factor of the Personalized PageRank (task 43 spec)."""

_TOP_ENTITY_SHARE = 0.3
"""Entities scoring below this share of the best score carry no evidence."""

_TOP_ENTITY_COUNT = 10
"""Maximum entities whose evidence chunks are collected."""

_EVIDENCE_PER_ENTITY = 6
"""Mention chunks one entity may contribute: a project entity mentioned by every note of
the project would otherwise fill the limit before the tool it leads to gets a chunk."""

_HIGH_EDGE_SHARE = 0.3
"""Edges need both endpoints above this share of the best entity score."""


@dataclass
class Subgraph:
    """The neighbourhood of the seed entities.

    Attributes:
        nodes: Entity id to canonical name (seeds included).
        edges: ``(src_entity_id, dst_entity_id, relation_id, weight)`` triples.
    """

    nodes: dict[int, str] = field(default_factory=dict)
    edges: list[tuple[int, int, int, float]] = field(default_factory=list)


def _validity_sql(as_of: datetime | None) -> str:
    """The relation filter for the walk: current-only, or valid at ``as_of``."""
    if as_of is None:
        return "r.status = 'active'"
    return (
        "r.status <> 'stale' AND r.valid_from <= :as_of "
        "AND (r.valid_to IS NULL OR r.valid_to > :as_of)"
    )


def freshness_factor(date: datetime | None, reference: datetime, half_life_days: float) -> float:
    """Exponential freshness weight ``0.5 ** (age_days / half_life_days)``.

    Args:
        date: When the fact was last stated (an evidence ``doc_date``) or the
            document date; None when unknown.
        reference: The point in time freshness is measured from ("now", or the
            query's as-of date).
        half_life_days: Age at which the weight halves; must be positive.

    Returns:
        1.0 for an unknown date (unknown age is not penalized) and for dates at
        or after ``reference``; otherwise a value in ``(0, 1)`` that halves
        every ``half_life_days``.
    """
    if date is None:
        return 1.0
    age_days = (reference - date).total_seconds() / 86_400
    if age_days <= 0:
        return 1.0
    return 0.5 ** (age_days / half_life_days)


def fetch_subgraph(
    session: Session,
    seeds: Sequence[int],
    hops: int,
    as_of: datetime | None = None,
    max_edges: int = 2000,
    *,
    half_life_days: float | None = None,
    now: datetime | None = None,
) -> Subgraph:
    """Fetch the entity neighbourhood around the seeds up to ``hops`` steps.

    The recursive CTE follows relations in both directions; ``UNION`` over
    ``(node, depth)`` pairs keeps it cycle-safe. Edges are the relations
    between any two reachable nodes that pass the validity filter, capped at
    ``max_edges`` (most evidence first), each weighted
    ``RELATION_WEIGHTS[type] * log1p(evidence_count)``. With
    ``half_life_days`` set, each weight is also multiplied by
    :func:`freshness_factor` of the relation's latest evidence ``doc_date``,
    measured from ``as_of`` when given, else ``now``.

    Args:
        session: An open session/transaction.
        seeds: Seed entity ids.
        hops: Maximum relation steps (>= 1).
        as_of: None for current intent (only ``active`` relations); a date for
            history intent (any status except ``stale``, valid at that date).
        max_edges: Cap on the returned edges.
        half_life_days: Freshness half-life; None leaves the weights undecayed.
        now: The reference time when ``as_of`` is None (defaults to the
            current UTC time); tests pass a fixed value.

    Returns:
        The :class:`Subgraph` around the seeds; empty when there are no seeds.
    """
    seed_list = sorted({int(seed) for seed in seeds})
    if not seed_list or hops < 1:
        return Subgraph()
    validity = _validity_sql(as_of)
    params: dict[str, object] = {"seeds": seed_list, "hops": hops, "max_edges": max_edges}
    if as_of is not None:
        params["as_of"] = as_of
    edge_rows = session.execute(
        text(
            f"""
            WITH RECURSIVE walk(node, depth) AS (
                SELECT inp.node, 0
                FROM unnest(CAST(:seeds AS bigint[])) AS inp(node)
              UNION
                SELECT CASE WHEN r.src_entity_id = w.node
                            THEN r.dst_entity_id ELSE r.src_entity_id END,
                       w.depth + 1
                FROM walk w
                JOIN kg_relation r
                  ON (r.src_entity_id = w.node OR r.dst_entity_id = w.node)
                 AND {validity}
                WHERE w.depth < :hops
            ),
            reachable AS (SELECT DISTINCT node FROM walk)
            SELECT r.id,
                   r.src_entity_id,
                   r.dst_entity_id,
                   r.type,
                   (SELECT count(*) FROM kg_relation_evidence ev
                    WHERE ev.relation_id = r.id) AS evidence_count,
                   (SELECT max(ev.doc_date) FROM kg_relation_evidence ev
                    WHERE ev.relation_id = r.id) AS latest_doc_date
            FROM kg_relation r
            WHERE {validity}
              AND r.src_entity_id IN (SELECT node FROM reachable)
              AND r.dst_entity_id IN (SELECT node FROM reachable)
            ORDER BY evidence_count DESC, r.id
            LIMIT :max_edges
            """
        ),
        params,
    ).all()
    reference = as_of if as_of is not None else (now or datetime.now(UTC))
    edges = [
        (
            int(src),
            int(dst),
            int(relation_id),
            _edge_weight(relation_type, int(evidence_count))
            * (
                freshness_factor(latest_doc_date, reference, half_life_days)
                if half_life_days is not None
                else 1.0
            ),
        )
        for relation_id, src, dst, relation_type, evidence_count, latest_doc_date in edge_rows
    ]
    edges.sort(key=lambda edge: (-edge[3], edge[2]))
    node_ids = {node for edge in edges for node in edge[:2]} | set(seed_list)
    names: dict[int, str] = {}
    if node_ids:
        names = {
            int(entity_id): str(name)
            for entity_id, name in session.execute(
                select(KgEntity.id, KgEntity.canonical_name).where(KgEntity.id.in_(node_ids))
            ).all()
        }
    return Subgraph(nodes=names, edges=edges)


def _edge_weight(relation_type: str, evidence_count: int) -> float:
    """``RELATION_WEIGHTS[type] * log1p(evidence_count)``."""
    return RELATION_WEIGHTS.get(relation_type, DEFAULT_RELATION_WEIGHT) * math.log1p(evidence_count)


def score_entities(
    sub: Subgraph, seeds: Mapping[int, float], method: Literal["one_hop", "ppr"]
) -> dict[int, float]:
    """Score every subgraph node from the seed scores.

    Args:
        sub: The subgraph to score.
        seeds: Seed entity id to seed score (entity linking and mentions).
        method: ``"one_hop"`` sums ``seed_score * edge_weight`` over the edges
            to adjacent seeds and normalizes the result to a maximum of 1;
            ``"ppr"`` runs Personalized PageRank (``alpha=0.85``) over the
            undirected subgraph with the seeds as personalization.

    Returns:
        Entity id to score; only positive scores are included (``one_hop``),
        all PageRank values (``ppr``, they sum to 1).
    """
    if not sub.nodes:
        return {}
    if method == "ppr":
        return _score_ppr(sub, seeds)
    return _score_one_hop(sub, seeds)


def _score_one_hop(sub: Subgraph, seeds: Mapping[int, float]) -> dict[int, float]:
    """Seed scores plus the weighted seed flow into direct neighbours.

    The flow into a neighbour is damped by ``1 + ln(1 + degree)`` of that
    neighbour in the subgraph: hub entities ("team", "procedure", "note vault")
    touch many seeds and would otherwise outscore the specific entity the
    question is about.
    """
    scores: dict[int, float] = {}
    for node, score in seeds.items():
        if node in sub.nodes:
            scores[node] = max(scores.get(node, 0.0), float(score))
    degree: dict[int, int] = {}
    for src, dst, _relation_id, _weight in sub.edges:
        degree[src] = degree.get(src, 0) + 1
        degree[dst] = degree.get(dst, 0) + 1
    flow: dict[int, float] = {}
    for src, dst, _relation_id, weight in sub.edges:
        if dst in seeds and src in sub.nodes and src not in seeds:
            flow[src] = flow.get(src, 0.0) + float(seeds[dst]) * weight
        if src in seeds and dst in sub.nodes and dst not in seeds:
            flow[dst] = flow.get(dst, 0.0) + float(seeds[src]) * weight
    for node, value in flow.items():
        scores[node] = scores.get(node, 0.0) + value / (1.0 + math.log1p(degree.get(node, 0)))
    best = max(scores.values(), default=0.0)
    if best > 0:
        scores = {node: score / best for node, score in scores.items()}
    return scores


def _score_ppr(sub: Subgraph, seeds: Mapping[int, float]) -> dict[int, float]:
    """Personalized PageRank over the undirected weighted subgraph."""
    graph = nx.Graph()
    graph.add_nodes_from(sub.nodes)
    graph.add_weighted_edges_from((src, dst, weight) for src, dst, _r, weight in sub.edges)
    if graph.number_of_nodes() == 0:
        return {}
    personalization = {node: float(seeds.get(node, 0.0)) for node in graph}
    if sum(personalization.values()) <= 0:
        personalization = {node: 1.0 / graph.number_of_nodes() for node in graph}
    ranks = _personalized_pagerank(graph, _PPR_ALPHA, personalization)
    return {int(node): float(score) for node, score in ranks.items()}


def _personalized_pagerank(
    graph: nx.Graph,
    alpha: float,
    personalization: Mapping[int, float],
    *,
    max_iter: int = 100,
    tol: float = 1.0e-6,
) -> dict[int, float]:
    """Personalized PageRank by power iteration, as ``networkx.pagerank`` computes it.

    networkx's own routine needs scipy (139 MB in the image) for sparse
    matrices; the subgraphs scored here have at most a few hundred nodes, so
    plain dictionaries do. Semantics follow networkx: the personalization
    vector is normalized, a dangling node (no edges) hands its mass out by
    that vector, edge weights share a node's mass proportionally, and the
    iteration stops when the L1 change drops below ``n * tol``; after
    ``max_iter`` rounds the current estimate is returned as it is.
    """
    nodes = list(graph)
    total = sum(personalization.values())
    preference = {node: personalization[node] / total for node in nodes}
    out_weight = {
        node: sum(float(data.get("weight", 1.0)) for data in graph[node].values()) for node in nodes
    }
    ranks = dict(preference)
    for _ in range(max_iter):
        previous = ranks
        ranks = dict.fromkeys(nodes, 0.0)
        dangling = alpha * sum(previous[node] for node in nodes if out_weight[node] == 0.0)
        for node in nodes:
            if out_weight[node] == 0.0:
                continue
            share = alpha * previous[node] / out_weight[node]
            for neighbor, data in graph[node].items():
                ranks[neighbor] += share * float(data.get("weight", 1.0))
        for node in nodes:
            ranks[node] += (dangling + 1.0 - alpha) * preference[node]
        if sum(abs(ranks[node] - previous[node]) for node in nodes) < len(nodes) * tol:
            break
    return ranks


def evidence_chunks(
    session: Session,
    entity_scores: Mapping[int, float],
    sub: Subgraph,
    limit: int = 30,
    *,
    seeds: Mapping[int, float] | None = None,
    current: bool = True,
    hints: dict[UUID, int] | None = None,
    min_mention_confidence: float = 0.0,
    exclude_document_ids: Sequence[int] = (),
) -> list[tuple[UUID, float]]:
    """The chunks that mention or evidence the graph's strongest entities.

    Collects the ``kg_mention`` chunks of the top-scoring entities (seeds are
    excluded unless nothing else scores) and the ``kg_relation_evidence``
    chunks of the edges between high-scoring entities. A chunk's score is the
    highest score of the entities or edges it backs.

    Args:
        session: An open session/transaction.
        entity_scores: Entity id to score (from :func:`score_entities`).
        sub: The subgraph the edges come from.
        limit: Maximum number of chunks.
        seeds: The seed mapping, so seeds can be excluded while other
            entities score; None means every scored entity counts. Pass only
            the *question's* entities here: entities mentioned by the top hits
            must stay eligible, their own notes are the second hop.
        hints: When given, filled with ``chunk_id -> entity_id`` for the chunks
            collected as mentions of a top entity (the retrieval rerank uses
            the entity name as a query hint).
        current: True for current intent (only current evidence of current
            documents); False for history intent (superseded documents and
            any evidence currency allowed).
        min_mention_confidence: Mentions below this confidence are ignored
            (1.0 = extractor mentions only); the rest weigh their chunk by
            their confidence.
        exclude_document_ids: Documents whose chunks are never evidence (the
            top hybrid document: its chunks are already candidates and would
            only crowd the limit with what the question started from).

    Returns:
        At most ``limit`` ``(chunk_id, score)`` pairs, best first.
    """
    if not entity_scores:
        return []
    best = max(entity_scores.values())
    if best <= 0:
        return []
    seed_ids = set(seeds or {})
    threshold = _TOP_ENTITY_SHARE * best
    # Rank the candidates with hub damping: an entity mentioned in sixty notes brings
    # sixty notes' worth of noise as evidence, the specific one brings its own note.
    counts = dict(
        session.execute(
            select(KgEntity.id, KgEntity.mention_count_current).where(
                KgEntity.id.in_(list(entity_scores))
            )
        ).all()
    )
    damped = {
        entity_id: score / (1.0 + math.log1p(max(0, int(counts.get(entity_id) or 0))))
        for entity_id, score in entity_scores.items()
    }
    ranked = sorted(entity_scores.items(), key=lambda item: (-damped[item[0]], item[0]))
    non_seeds = [(entity_id, score) for entity_id, score in ranked if entity_id not in seed_ids]
    top_entities = [entity_id for entity_id, score in non_seeds if score >= threshold]
    top_entities = top_entities[:_TOP_ENTITY_COUNT]
    if not top_entities and non_seeds:
        top_entities = [non_seeds[0][0]]
    if not top_entities:
        top_entities = [entity_id for entity_id, score in ranked if score >= threshold]
        top_entities = top_entities[:_TOP_ENTITY_COUNT]
    if not top_entities:
        return []

    edge_scores: dict[int, float] = {}
    for src, dst, relation_id, _weight in sub.edges:
        if src in entity_scores and dst in entity_scores:
            low = min(entity_scores[src], entity_scores[dst])
            if low >= _HIGH_EDGE_SHARE * best:
                edge_scores[relation_id] = max(entity_scores[src], entity_scores[dst])

    document_filter = (
        Document.status == "current" if current else Document.status.in_(("current", "superseded"))
    )
    scores: dict[UUID, float] = {}
    hint_confidence: dict[UUID, float] = {}
    owners: dict[UUID, int] = {}

    def record(chunk_id: UUID, score: float, owner: int | None = None) -> None:
        if score > scores.get(chunk_id, 0.0):
            scores[chunk_id] = score
            if owner is not None:
                owners[chunk_id] = owner

    if top_entities:
        for chunk_id, entity_id, confidence in session.execute(
            select(KgMention.chunk_id, KgMention.entity_id, KgMention.confidence)
            .join(Chunk, Chunk.id == KgMention.chunk_id)
            .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(
                KgMention.entity_id.in_(top_entities),
                KgMention.confidence >= min_mention_confidence,
                DocumentVersion.is_latest.is_(True),
                document_filter,
                *(
                    [Chunk.document_id.not_in(list(exclude_document_ids))]
                    if exclude_document_ids
                    else []
                ),
            )
        ).all():
            # A lexical mention (confidence 0.6) is weaker evidence than an extractor one.
            weighted = float(entity_scores[int(entity_id)]) * float(confidence or 1.0)
            record(UUID(str(chunk_id)), weighted, int(entity_id))
            if hints is not None:
                current_hint = hints.get(UUID(str(chunk_id)))
                if current_hint is None or weighted > entity_scores.get(current_hint, 0.0) * (
                    hint_confidence.get(UUID(str(chunk_id)), 1.0)
                ):
                    hints[UUID(str(chunk_id))] = int(entity_id)
                    hint_confidence[UUID(str(chunk_id))] = float(confidence or 1.0)
    if edge_scores:
        for chunk_id, relation_id in session.execute(
            select(KgRelationEvidence.chunk_id, KgRelationEvidence.relation_id)
            .join(Chunk, Chunk.id == KgRelationEvidence.chunk_id)
            .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(
                KgRelationEvidence.relation_id.in_(list(edge_scores)),
                DocumentVersion.is_latest.is_(True),
                document_filter,
                *(
                    [Chunk.document_id.not_in(list(exclude_document_ids))]
                    if exclude_document_ids
                    else []
                ),
                *([KgRelationEvidence.is_current.is_(True)] if current else []),
            )
        ).all():
            record(UUID(str(chunk_id)), float(edge_scores[int(relation_id)]))
    ordered = sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))
    return cap_per_entity(ordered, owners, _EVIDENCE_PER_ENTITY, limit)


def cap_per_entity(
    ordered: Sequence[tuple[UUID, float]],
    owners: Mapping[UUID, int],
    cap: int,
    limit: int,
) -> list[tuple[UUID, float]]:
    """Keep at most ``cap`` chunks per owning entity, then the first ``limit`` overall.

    Chunks without an owner (relation evidence) are never capped.
    """
    taken: dict[int, int] = {}
    kept: list[tuple[UUID, float]] = []
    for chunk_id, score in ordered:
        owner = owners.get(chunk_id)
        if owner is not None:
            if taken.get(owner, 0) >= cap:
                continue
            taken[owner] = taken.get(owner, 0) + 1
        kept.append((chunk_id, score))
        if len(kept) >= limit:
            break
    return kept
