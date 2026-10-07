"""Cross-encoder reranking of the fused candidates (PLAN.md §9 step 5).

The document text sent to the reranker prepends the title and the heading
path when the chunk text does not already carry them (near-identical
chunks from template documents are only distinguishable by their title). Only
the first ``retrieval.rerank_top`` hits are reranked; an archived document
gets the ``0.9`` prior applied to its score, and hits sharing a ``project:*``
tag get a mild recency boost by ``doc_date`` rank within the project (task 35,
at most 5%, newest first). With :class:`RecencyParams` (``retrieval.recency_boost``),
every hit is also multiplied by a floored exponential freshness factor of its
``doc_date``. A reranker outage degrades to the input order instead of failing
the search.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from mynotes.graph.expand import freshness_factor
from mynotes.llm.errors import EndpointError
from mynotes.retrieval.rerank_client import Reranker
from mynotes.retrieval.types import SearchHit

logger = logging.getLogger(__name__)

_ARCHIVED_PRIOR = 0.9
"""Current documents outrank archived ones with the same rerank score."""

_RECENCY_PRIOR = 0.05
"""Ceiling of the mild recency boost within one project (PLAN.md §9 step 5)."""

_PROJECT_TAG_PREFIX = "project:"

_HINTED_BEYOND_TOP = 10
"""Hinted (second-hop) hits past ``rerank_top`` that are still reranked."""


@dataclass(frozen=True)
class RecencyParams:
    """The optional recency boost of the final ranking (``retrieval.recency_*``).

    Attributes:
        half_life_days: Age at which the freshness factor halves.
        min_multiplier: Floor of the multiplier, so a very old document keeps
            at least this share of its score (``0.8`` = 80 %).
        reference: The time freshness is measured from: the query's as-of
            date, or "now".
    """

    half_life_days: float
    min_multiplier: float
    reference: datetime

    def multiplier(self, doc_date: datetime | None) -> float:
        """``floor + (1 - floor) * freshness``; 1.0 for an unknown date."""
        if doc_date is None:
            return 1.0
        fresh = freshness_factor(doc_date, self.reference, self.half_life_days)
        return self.min_multiplier + (1.0 - self.min_multiplier) * fresh


def _project_recency_multipliers(hits: Sequence[SearchHit]) -> dict[int, float]:
    """Per-document multiplier in ``(1.0, 1.05]`` from the project recency prior.

    Hits sharing a ``project:*`` tag form one group; within a group whose
    documents carry at least two distinct known ``doc_date`` values, a
    document's multiplier is ``1 + 0.05 * r`` where ``r in [0, 1]`` ranks its
    date within the group (newest = 1, oldest = 0, equal dates share a rank).
    A document in several groups keeps its strongest multiplier.
    """
    groups: dict[str, dict[int, datetime]] = {}
    for hit in hits:
        if hit.doc_date is None:
            continue
        for tag in hit.tags:
            if tag.startswith(_PROJECT_TAG_PREFIX):
                groups.setdefault(tag, {})[hit.document_id] = hit.doc_date
    multipliers: dict[int, float] = {}
    for documents in groups.values():
        dates = sorted(set(documents.values()), reverse=True)
        if len(dates) < 2:
            continue
        span = len(dates) - 1
        for document_id, date in documents.items():
            multiplier = 1.0 + _RECENCY_PRIOR * (1.0 - dates.index(date) / span)
            if multiplier > multipliers.get(document_id, 1.0):
                multipliers[document_id] = multiplier
    return multipliers


def rerank_document_text(hit: SearchHit) -> str:
    """Build the document text sent to the reranker for one hit.

    Args:
        hit: The candidate hit.

    Returns:
        ``title`` and ``heading_path`` prepended on their own lines when they
        are not already contained in the text (case-insensitive), then the
        text.
    """
    text = hit.text
    headers: list[str] = []
    for header in (hit.title, hit.heading_path):
        stripped = (header or "").strip()
        if stripped and stripped.lower() not in text.lower():
            headers.append(stripped)
    if not headers:
        return text
    return "\n".join([*headers, text])


_BRIDGE_NOUN_RE = re.compile(
    r"\b(plugins?|tools?|ferramentas?|librar(?:y|ies)|bibliotecas?|extensions?|extens[aã]o|"
    r"extens[oõ]es|frameworks?|software)\b",
    re.IGNORECASE,
)


def resolved_query(query: str, hint: str) -> str | None:
    """The question with its bridge description replaced by the resolved entity.

    "What is the licence of the tool that the notification service of Project
    Tua is built on?" + "Memoranda" gives "What is the licence of the
    Memoranda?": the part before the bridge noun is what is asked about the
    tool, the rest only says how to find it (and talks about the bridge note,
    which drags the tool note's score down). None when the question has no
    bridge noun or nothing meaningful before it.
    """
    match = _BRIDGE_NOUN_RE.search(query)
    if match is None:
        return None
    prefix = query[: match.start()].strip()
    if len(prefix.split()) < 2:
        return None
    return f"{prefix} {hint}?"


def _apply_query_hints(query: str, head: Sequence[SearchHit], reranker: Reranker) -> None:
    """Score hinted hits against ``query + hint`` and the resolved question; keep the best.

    Second-hop evidence (a tool's own note reached through the note that names
    the tool) is relevant to the question once the entity is resolved, but the
    raw question rarely mentions the entity, so the plain rerank buries it. The
    full question plus the hint is still mostly about the bridge note, so the
    hinted hits are also scored against :func:`resolved_query` (golden set v3
    h015, h042: 0.05-0.07 with the first form). One reranker call per distinct
    hint and form; a failing call leaves the previous score.
    """
    groups: dict[str, list[SearchHit]] = {}
    for hit in head:
        if hit.query_hint:
            groups.setdefault(hit.query_hint, []).append(hit)
    for hint, hits in groups.items():
        queries = [f"{query} {hint}"]
        resolved = resolved_query(query, hint)
        if resolved is not None:
            queries.append(resolved)
        for hinted_query in queries:
            try:
                results = reranker.rerank(
                    hinted_query, [rerank_document_text(hit) for hit in hits], top_n=None
                )
            except EndpointError:
                logger.warning(
                    "hinted rerank failed for %r; keeping plain scores", hint, exc_info=True
                )
                continue
            for result in results:
                hit = hits[result.index]
                if hit.rerank_score is None or result.score > hit.rerank_score:
                    hit.rerank_score = result.score


CHANGE_CLUSTER_DROP = 0.03
"""A drop larger than this share of the best score ends the change cluster."""


def change_cluster(pool: Sequence[SearchHit], drop: float = CHANGE_CLUSTER_DROP) -> list[SearchHit]:
    """The notes that state the asked value: the first run of near-equal best scores.

    Works per note (its best chunk). Walking down the scores, the cluster ends at
    the first drop larger than ``drop`` x the best score. A cluster of one note
    that is not the newest of the pool is reranker noise (one note of the next
    value scored above the rest, golden set v4 s028) and is merged with the run
    that follows; a single newest note is a change in the latest week and stays.
    Measured on golden set v4 against a fixed share and the largest gap
    (docs/eval/series.md).
    """
    best: dict[int, SearchHit] = {}
    for hit in pool:
        if hit.rerank_score is None:
            continue
        current = best.get(hit.document_id)
        if current is None or (hit.rerank_score or 0.0) > (current.rerank_score or 0.0):
            best[hit.document_id] = hit
    notes = sorted(best.values(), key=lambda hit: -(hit.rerank_score or 0.0))
    if len(notes) < 2:
        return notes
    top = notes[0].rerank_score or 0.0

    def run_from(start: int) -> int:
        end = start + 1
        while (
            end < len(notes)
            and (notes[end - 1].rerank_score or 0.0) - (notes[end].rerank_score or 0.0)
            <= drop * top
        ):
            end += 1
        return end

    end = run_from(0)
    newest = max(
        (hit.doc_date for hit in notes if hit.doc_date is not None),
        default=None,
    )
    if end == 1 and end < len(notes) and notes[0].doc_date != newest:
        end = run_from(end)
    return notes[:end]


def order_change_cluster(hits: list[SearchHit]) -> list[SearchHit]:
    """Put the cluster of best notes in date order, earliest first, for change questions.

    "Em que reunião X passou a Y?": every note since the change states "X: Y" and the
    reranker scores them alike; the answer is the earliest of them. When the series
    expansion marked a folder (source ``series``), only its notes are considered: the
    reranker can prefer another series that tracks the same item. The latest note of
    the same folder dated before the earliest one (the old value) goes second; then the
    rest of the cluster, then everything else.
    """
    pool = [hit for hit in hits if "series" in hit.sources] or list(hits)
    cluster = change_cluster(pool)
    if len(cluster) < 2:
        return hits
    cluster.sort(
        key=lambda hit: (
            hit.doc_date is None,
            hit.doc_date or datetime.max.replace(tzinfo=UTC),
            hit.rank,
        )
    )
    first = cluster[0]
    head = [first]
    # The note just before the change, from the same folder, shows the old value: without
    # it the model cannot tell the earliest note it sees from the week of the change.
    if first.doc_date is not None:
        folder = _folder(first.path)
        cluster_docs = {hit.document_id for hit in cluster}
        earlier = [
            hit
            for hit in pool
            if hit.document_id not in cluster_docs
            and hit.doc_date is not None
            and hit.doc_date < first.doc_date
            and _folder(hit.path) == folder
        ]
        if earlier:
            head.append(
                max(earlier, key=lambda hit: (hit.doc_date, hit.rerank_score or 0.0, -hit.rank))
            )
    ordered = [*head, *[hit for hit in cluster if hit is not first]]
    reordered = [*ordered, *[hit for hit in hits if hit not in ordered]]
    for rank, hit in enumerate(reordered, start=1):
        hit.rank = rank
    return reordered


def _folder(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def promote_bridge_hit(
    hits: list[SearchHit],
    max_docs: int,
    *,
    min_score: float = 0.05,
    preferred_types: frozenset[str] = frozenset(),
    slots: int = 1,
) -> list[SearchHit]:
    """Give the best second-hop (hinted) hit a place among the context documents.

    Hints exist only for chain-shaped questions and only for entities the top
    document mentions, so a hinted hit is the note of the very thing the
    question points at through the bridge. When the best such hit (by rerank
    score, at least ``min_score``) belongs to a document that would not make
    the first ``max_docs`` distinct documents, it moves to second place, right
    after the bridge, and ranks are renumbered.

    Args:
        hits: The reranked hits, best first.
        max_docs: ``retrieval.max_docs``.
        min_score: Below this rerank score the hint was not enough; nothing moves.
        preferred_types: The bridge's entity type ("the tool that ..." →
            Technology); a hinted hit whose hint has one of these types wins
            over a higher-scoring hint of another type.
        slots: How many hinted documents to place (2 for three-hop chains).
            A hinted note of a preferred type already among the context
            documents fills a slot. Slots after the first go only to hints of
            a preferred type: a second weak hint of another type (an
            organisation the bridge note names) would push a good note out of
            the context.

    Returns:
        The (possibly reordered) hits.
    """
    if len(hits) < 2:
        return hits
    leading_docs: list[int] = []
    for hit in hits:
        if hit.document_id not in leading_docs:
            leading_docs.append(hit.document_id)
        if len(leading_docs) >= max_docs:
            break
    # A hinted note of the bridge type that already ranks among the context documents fills
    # a slot: promoting weaker hinted notes ahead of it would push it down (golden set v3
    # h041: the tool note at 0.988 fell to fourth place behind two hints at 0.07).
    satisfied = {
        hit.document_id
        for hit in hits
        if hit.query_hint
        and hit.document_id in leading_docs[1:]
        and (not preferred_types or (hit.query_hint_type or "") in preferred_types)
    }
    free = max(1, slots) - len(satisfied)
    if free <= 0:
        return hits
    candidates = [
        hit
        for hit in hits
        if hit.query_hint
        and hit.document_id not in leading_docs
        and hit.rerank_score is not None
        and hit.rerank_score >= min_score
    ]
    if not candidates:
        return hits
    candidates.sort(
        key=lambda hit: (
            1 if (hit.query_hint_type or "") in preferred_types else 0,
            hit.rerank_score or 0.0,
        ),
        reverse=True,
    )
    chosen: list[SearchHit] = []
    for hit in candidates:
        if len(chosen) >= free:
            break
        if hit.document_id in {c.document_id for c in chosen}:
            continue
        if (chosen or satisfied) and (hit.query_hint_type or "") not in preferred_types:
            continue
        chosen.append(hit)
    if not chosen:
        return hits
    reordered = [hits[0], *chosen, *[hit for hit in hits[1:] if hit not in chosen]]
    for rank, hit in enumerate(reordered, start=1):
        hit.rank = rank
    return reordered


def rerank_hits(
    query: str,
    hits: Sequence[SearchHit],
    reranker: Reranker,
    top: int,
    *,
    recency: RecencyParams | None = None,
) -> list[SearchHit]:
    """Rerank the first ``top`` hits against the query.

    Args:
        query: The query text (the analyzed query text, not the raw input).
        hits: Candidates in fused order.
        reranker: The cross-encoder client.
        top: How many of the leading hits to rerank; hits beyond ``top`` are
            dropped from the result.
        recency: The recency boost parameters, or None for no boost.

    Returns:
        The reranked hits sorted by the final score descending (rerank score
        times the ``0.9`` archived prior, the project recency prior and the
        recency multiplier when ``recency`` is given), ranks re-numbered
        from 1. ``rerank_score`` keeps the raw reranker score. When the reranker raises
        :class:`~mynotes.llm.errors.EndpointError`, the full input list is
        returned unchanged with ``rerank_score=None`` (degrade, don't fail).
    """
    hits = list(hits)
    if not hits:
        return []
    head = hits[: max(0, top)]
    # Second-hop evidence often fuses in late (one source, mid rank); keep a few hinted
    # hits beyond the window in the rerank so the hint can still lift them.
    extra = [hit for hit in hits[max(0, top) :] if hit.query_hint][:_HINTED_BEYOND_TOP]
    head = [*head, *extra]
    documents = [rerank_document_text(hit) for hit in head]
    try:
        results = reranker.rerank(query, documents, top_n=None)
    except EndpointError:
        logger.warning(
            "reranker failed; keeping the fused order for %d hit(s)", len(hits), exc_info=True
        )
        for hit in hits:
            hit.rerank_score = None
        return hits
    scores = {result.index: result.score for result in results}
    for position, hit in enumerate(head):
        hit.rerank_score = scores.get(position)
    _apply_query_hints(query, head, reranker)

    multipliers = _project_recency_multipliers(head)

    def final_score(hit: SearchHit) -> float:
        if hit.rerank_score is None:
            return -math.inf
        prior = _ARCHIVED_PRIOR if hit.doc_status == "archived" else 1.0
        boost = recency.multiplier(hit.doc_date) if recency is not None else 1.0
        return hit.rerank_score * prior * multipliers.get(hit.document_id, 1.0) * boost

    ranked = sorted(head, key=lambda hit: -final_score(hit))
    for rank, hit in enumerate(ranked, start=1):
        hit.rank = rank
    return ranked
