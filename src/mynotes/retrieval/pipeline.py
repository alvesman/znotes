"""Retrieval orchestrator: analyze, hybrid search, expand, fuse, rerank.

:func:`retrieve` is the single retrieval entry point used by the ``search``
CLI, ``ask`` (task 30) and later the HTTP API/MCP (M6). Expanders contribute
extra candidate rankings (versions in task 35, graph in task 43); they
register themselves with :func:`register_expander` and a failing expander is
logged and skipped. There is no score threshold on the search itself: whether
anything relevant was found is decided from the best rerank score against
``retrieval.min_rerank_score`` (PLAN.md §6).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from mynotes.config import Settings, get_settings
from mynotes.retrieval.analyze import QueryAnalysis, analyze_query
from mynotes.retrieval.fuse import rrf_fuse
from mynotes.retrieval.hybrid import hybrid_search
from mynotes.retrieval.rerank import (
    RecencyParams,
    order_change_cluster,
    promote_bridge_hit,
    rerank_hits,
)
from mynotes.retrieval.rerank_client import get_reranker, reranker_configured
from mynotes.retrieval.title_lock import (
    expand_series,
    front_anchored,
    lock_named_dates,
    lock_quoted_titles,
)
from mynotes.retrieval.types import SearchFilters, SearchHit

logger = logging.getLogger(__name__)


class Expander(Protocol):
    """A pluggable candidate source whose ranking is fused into the result.

    Attributes:
        name: The fusion source name (``"version"``, ``"graph"``, ...).
    """

    name: str

    def enabled(self, settings: Settings, analysis: QueryAnalysis) -> bool:
        """Whether this expander contributes for the given query."""
        ...

    def expand(self, analysis: QueryAnalysis, hybrid_hits: Sequence[SearchHit]) -> list[SearchHit]:
        """Return this source's candidate hits, best first."""
        ...


_EXPANDERS: list[Expander] = []


def register_expander(expander: Expander) -> None:
    """Add an expander to the registry, replacing one with the same name."""
    for position, existing in enumerate(_EXPANDERS):
        if existing.name == expander.name:
            _EXPANDERS[position] = expander
            return
    _EXPANDERS.append(expander)


def unregister_expander(name: str) -> None:
    """Remove the expander registered under ``name`` (no-op when absent)."""
    _EXPANDERS[:] = [expander for expander in _EXPANDERS if expander.name != name]


@dataclass
class RetrievalResult:
    """The outcome of one :func:`retrieve` call.

    Attributes:
        analysis: The query analysis (cleaned text, filters, intent, language).
        hits: The final hits, reranked when reranking ran, capped at
            ``settings.retrieval.max_chunks * 2``.
        candidates: The fused candidate count before reranking.
        relevant: Whether the best rerank score reaches
            ``settings.retrieval.min_rerank_score``; always True when
            reranking was skipped or degraded.
        timings_ms: Milliseconds spent per pipeline stage.
    """

    analysis: QueryAnalysis
    hits: list[SearchHit]
    candidates: int
    relevant: bool
    timings_ms: dict[str, int] = field(default_factory=dict)


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def recency_params(
    settings: Settings, analysis: QueryAnalysis, *, now: datetime | None = None
) -> RecencyParams | None:
    """The recency boost for one query, or None when it does not apply.

    The boost is off unless ``retrieval.recency_boost`` is set. It is measured
    from the query's ``as-of`` date when there is one, else from ``now``; a
    history question without an as-of date wants older documents, so it gets
    no boost at all.

    Args:
        settings: Application settings.
        analysis: The query analysis (intent and as-of date).
        now: The reference time (defaults to the current UTC time).

    Returns:
        The parameters for :func:`rerank_hits`, or None.
    """
    retrieval = settings.retrieval
    if not retrieval.recency_boost:
        return None
    if analysis.asks_change and retrieval.change_questions:
        # "when did X change to Y" wants the earliest note stating Y, not the newest
        return None
    if analysis.as_of is not None:
        reference = analysis.as_of
    elif analysis.intent == "history":
        return None
    else:
        reference = now or datetime.now(UTC)
    return RecencyParams(
        half_life_days=retrieval.recency_half_life_days,
        min_multiplier=retrieval.recency_min_multiplier,
        reference=reference,
    )


def retrieve(
    query: str, filters: SearchFilters | None = None, *, rerank: bool = True
) -> RetrievalResult:
    """Run the full retrieval pipeline for one query.

    Args:
        query: The raw user query, possibly with inline filter tokens.
        filters: Base filters from CLI/API options; inline filters win.
        rerank: When False, skip the reranker and return the fused order.

    Returns:
        The :class:`RetrievalResult` for the query.
    """
    settings = get_settings()
    timings: dict[str, int] = {}

    started = time.perf_counter()
    analysis = analyze_query(query, filters)
    timings["analyze"] = _elapsed_ms(started)

    started = time.perf_counter()
    hybrid_hits = hybrid_search(analysis.text, analysis.filters)
    if analysis.quoted and settings.retrieval.title_lock:
        hybrid_hits = lock_quoted_titles(hybrid_hits, analysis)
    if analysis.named_dates and settings.retrieval.date_lock:
        hybrid_hits = lock_named_dates(hybrid_hits, analysis)
    if analysis.asks_change and settings.retrieval.change_questions:
        hybrid_hits = expand_series(hybrid_hits, analysis)
    timings["hybrid"] = _elapsed_ms(started)

    started = time.perf_counter()
    expanded: dict[str, list[SearchHit]] = {}
    for expander in list(_EXPANDERS):
        try:
            if not expander.enabled(settings, analysis):
                continue
            expanded[expander.name] = expander.expand(analysis, hybrid_hits)
        except Exception:
            logger.warning("expander %r failed; skipping it", expander.name, exc_info=True)
    timings["expand"] = _elapsed_ms(started)

    started = time.perf_counter()
    fused = rrf_fuse(
        {"hybrid": hybrid_hits, **expanded},
        settings.retrieval.fusion.source_weights,
        settings.retrieval.fusion.rank_constant,
    )
    candidates = len(fused)
    timings["fuse"] = _elapsed_ms(started)

    started = time.perf_counter()
    degraded = False
    if rerank and fused and reranker_configured():
        hits = rerank_hits(
            analysis.text,
            fused,
            get_reranker(),
            settings.retrieval.rerank_top,
            recency=recency_params(settings, analysis),
        )
        degraded = bool(hits) and all(hit.rerank_score is None for hit in hits)
        from mynotes.retrieval.graph_expand import (  # late: graph_expand imports this module
            bridge_entity_types,
            is_deep_chain_question,
        )

        hits = front_anchored(hits)
        if analysis.asks_change and settings.retrieval.change_questions:
            hits = order_change_cluster(hits)
        hits = promote_bridge_hit(
            hits,
            settings.retrieval.max_docs,
            preferred_types=bridge_entity_types(analysis.raw),
            slots=2 if is_deep_chain_question(analysis.raw) else 1,
        )
    else:
        hits = fused
    timings["rerank"] = _elapsed_ms(started)

    hits = hits[: settings.retrieval.max_chunks * 2]
    if rerank:
        relevant = degraded or any(
            hit.rerank_score is not None and hit.rerank_score >= settings.retrieval.min_rerank_score
            for hit in hits
        )
    else:
        relevant = True
    logger.info(
        "retrieve %r: %d candidate(s), %d hit(s), relevant=%s",
        analysis.text,
        candidates,
        len(hits),
        relevant,
    )
    return RetrievalResult(
        analysis=analysis,
        hits=hits,
        candidates=candidates,
        relevant=relevant,
        timings_ms=timings,
    )


# The task-35 version expander/provider and the task-43 graph expander/provider
# register themselves on import; the bottom placement lets them import this
# module's registry without an import cycle.
from mynotes.retrieval import (  # noqa: E402,F401
    graph_expand,
    version_expand,
)
