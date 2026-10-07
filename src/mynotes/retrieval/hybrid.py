"""Hybrid (k-NN + BM25) chunk search through the RRF pipeline (PLAN.md §6, §9 step 2).

:func:`build_filter_clauses` turns :class:`~mynotes.retrieval.types.SearchFilters`
into the OpenSearch filter clauses that go into **both** hybrid legs, so RRF
ranks only eligible chunks. :func:`build_hybrid_body` renders the native
``hybrid`` query (leg order matters: the pipeline weights are ``[knn, bm25]``)
and :func:`hybrid_search` runs it against the chunks alias, maps the hits to
:class:`~mynotes.retrieval.types.SearchHit` and deduplicates per
document/page/section. There is no score threshold: RRF scores
depend only on rank (PLAN.md §6); "nothing relevant" is the reranker's call
in task 28.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime
from typing import Any

from mynotes.config import Settings, get_settings
from mynotes.db.session import session_scope
from mynotes.embed.client import get_embedder
from mynotes.index.mappings import PIPELINE_NAME
from mynotes.index.opensearch import get_client
from mynotes.retrieval.types import SearchFilters, SearchHit
from mynotes.text.normalize import normalize_path_key

logger = logging.getLogger(__name__)

BM25_FIELDS: list[str] = [
    "text",
    "text.pt",
    "text.en",
    "title^2",
    "title.pt^2",
    "title.en^2",
    "heading_path_text",
    "heading_path_text.pt",
    "heading_path_text.en",
]


def _status_clause(f: SearchFilters) -> dict[str, Any]:
    """The ``doc_status`` terms clause; superseded only with history intent."""
    statuses = ["current", "archived"] + (["superseded"] if f.include_superseded else [])
    return {"terms": {"doc_status": statuses}}


def _folder_clause(folders: list[str]) -> dict[str, Any]:
    """Folded folder-prefix filter over ``path_ancestors_folded`` (PLAN.md §6.1)."""
    return {
        "bool": {
            "should": [
                {"term": {"path_ancestors_folded": normalize_path_key(folder)}}
                for folder in folders
            ],
            "minimum_should_match": 1,
        }
    }


def _acl_clause(prefixes: list[str]) -> dict[str, Any]:
    """Exact-path ACL filter; an empty prefix list means no access at all.

    Linux paths are case-sensitive, so the ACL matches the exact
    ``path_ancestors``/``path`` fields and is never folded (PLAN.md §6.1).
    A ``""`` prefix (a whole-workspace grant) matches everything.
    """
    if not prefixes:
        return {"bool": {"must_not": {"match_all": {}}}}
    if "" in prefixes:
        return {"match_all": {}}
    should: list[dict[str, Any]] = []
    for prefix in prefixes:
        should.append({"term": {"path_ancestors": prefix}})
        should.append({"prefix": {"path": prefix + "/"}})
    return {"bool": {"should": should, "minimum_should_match": 1}}


def build_filter_clauses(f: SearchFilters) -> list[dict[str, Any]]:
    """Render :class:`SearchFilters` as OpenSearch filter clauses.

    Args:
        f: The filters to apply.

    Returns:
        A conjunction (``bool.filter`` list) of clauses: latest-version and
        status gates, one ``term`` per tag, the folded folder prefixes, the
        format terms, the ``doc_date`` range, the exact ACL prefixes, the
        workspace term and the ``document_id`` restriction. The list is never
        empty (the status gate is always present), so it can be dropped into
        both hybrid legs unchanged.
    """
    clauses: list[dict[str, Any]] = []
    if not f.include_old_versions:
        clauses.append({"term": {"is_latest": True}})
    clauses.append(_status_clause(f))
    clauses.extend({"term": {"tags": tag}} for tag in f.tags)
    if f.folders:
        clauses.append(_folder_clause(f.folders))
    if f.formats:
        clauses.append({"terms": {"format": f.formats}})
    if f.document_ids:
        clauses.append({"terms": {"document_id": [str(i) for i in f.document_ids]}})
    date_range: dict[str, str] = {}
    if f.since is not None:
        date_range["gte"] = f.since.isoformat()
    if f.until is not None:
        date_range["lte"] = f.until.isoformat()
    if date_range:
        clauses.append({"range": {"doc_date": date_range}})
    if f.acl_prefixes is not None:
        clauses.append(_acl_clause(f.acl_prefixes))
    if f.workspace_id is not None:
        clauses.append({"term": {"workspace_id": str(f.workspace_id)}})
    return clauses


def build_hybrid_body(
    query: str, query_vector: list[float], f: SearchFilters, size: int
) -> dict[str, Any]:
    """Build the native ``hybrid`` query body for the RRF pipeline.

    The k-NN leg comes first (the pipeline weights are ``[knn, bm25]``) with
    an inline filter; the BM25 leg is a ``most_fields`` multi_match wrapped in
    a ``bool`` with the same filter list, so RRF ranks only eligible chunks.

    Args:
        query: The raw query text (embedded as typed; folding happens in the
            BM25 analysis only, PLAN.md §6.1).
        query_vector: The query embedding.
        f: The filters to apply to both legs.
        size: Number of candidates to fetch per leg and overall.

    Returns:
        The ``client.search`` body.
    """
    clauses = build_filter_clauses(f)
    knn_leg = {
        "knn": {
            "embedding": {
                "vector": list(query_vector),
                "k": size,
                "filter": {"bool": {"filter": clauses}},
            }
        }
    }
    bm25_leg = {
        "bool": {
            "must": [
                {"multi_match": {"query": query, "type": "most_fields", "fields": BM25_FIELDS}}
            ],
            "filter": clauses,
        }
    }
    return {
        "size": size,
        "_source": {"excludes": ["embedding"]},
        "query": {"hybrid": {"queries": [knn_leg, bm25_leg]}},
    }


def _parse_doc_date(value: Any) -> datetime | None:
    """Parse an indexed ``doc_date`` ISO string (None passes through)."""
    if value is None:
        return None
    return datetime.fromisoformat(str(value))


def _to_hit(hit: dict[str, Any], rank: int) -> SearchHit:
    """Map one OpenSearch hit to a :class:`SearchHit` (rank is 1-based)."""
    source = hit.get("_source") or {}
    return SearchHit(
        chunk_id=str(source["chunk_id"]),
        document_id=int(source["document_id"]),
        version_id=int(source["version_id"]),
        rank=rank,
        score=float(hit.get("_score") or 0.0),
        text=str(source.get("text") or ""),
        title=source.get("title"),
        section=source.get("section"),
        heading_path=str(source.get("heading_path_text") or ""),
        path=str(source.get("path") or ""),
        page=source.get("page"),
        slide=source.get("slide"),
        doc_date=_parse_doc_date(source.get("doc_date")),
        doc_status=str(source.get("doc_status") or ""),
        is_latest=bool(source.get("is_latest")),
        is_table=bool(source.get("is_table")),
        format=str(source.get("format") or ""),
        language=source.get("language"),
        tags=list(source.get("tags") or []),
    )


def _group_key(hit: SearchHit) -> tuple[int, str, int | str | None]:
    """The dedup group of a hit: the document plus page/slide/section."""
    if hit.page is not None:
        return (hit.document_id, "page", hit.page)
    if hit.slide is not None:
        return (hit.document_id, "slide", hit.slide)
    return (hit.document_id, "section", hit.section)


def _dedupe(hits: list[SearchHit], per_group_limit: int) -> list[SearchHit]:
    """Keep at most ``per_group_limit`` hits per group, best-ranked first.

    The input is in ascending rank order, so keeping the first hits per group
    keeps the best-ranked; ranks are then re-numbered without gaps.

    Args:
        hits: Hits ordered by rank.
        per_group_limit: Maximum hits per ``(document, page/slide/section)``.

    Returns:
        The filtered hits with fresh 1-based ranks.
    """
    kept: list[SearchHit] = []
    counts: dict[tuple[int, str, int | str | None], int] = {}
    for hit in hits:
        key = _group_key(hit)
        if counts.get(key, 0) >= per_group_limit:
            continue
        counts[key] = counts.get(key, 0) + 1
        kept.append(hit)
    for new_rank, hit in enumerate(kept, start=1):
        hit.rank = new_rank
    return kept


def _resolve_tag_filters(settings: Settings, filters: SearchFilters) -> SearchFilters:
    """Normalize the tag filter values and resolve their aliases (task 37).

    Args:
        settings: The application settings (default workspace name).
        filters: The filters whose ``tags`` to resolve.

    Returns:
        The filters with resolved tag names; unchanged when there are no tags
        or the workspace does not exist (unknown names simply match nothing).
    """
    if not filters.tags:
        return filters
    from mynotes.tags.store import resolve_tag_names, workspace_id_by_name

    with session_scope() as session:
        workspace_id = filters.workspace_id
        if workspace_id is None:
            workspace_id = workspace_id_by_name(session, settings.workspace)
        if workspace_id is None:
            logger.debug("tag filters unresolved: workspace %r not found", settings.workspace)
            return filters
        names = resolve_tag_names(session, workspace_id, filters.tags)
    return replace(filters, tags=names)


def hit_matches_filters(hit: SearchHit, f: SearchFilters) -> bool:
    """Whether one hit passes the same rules :func:`build_filter_clauses` renders.

    Python-side mirror of the OpenSearch filter clauses, used for candidate
    hits that never went through a filtered search (graph expansion loads
    chunks by id from the index, task 43). The ACL check is exact-path on the
    hit's ancestors and path, never folded, and an empty prefix list means no
    access at all - so the ACL can never be bypassed by an expander.

    Args:
        hit: The candidate hit.
        f: The filters to apply.

    Returns:
        True when the hit would have matched a hybrid search with ``f``.
    """
    from mynotes.platform.paths import path_ancestors, path_ancestors_folded
    from mynotes.text.normalize import normalize_path_key

    if not f.include_old_versions and not hit.is_latest:
        return False
    statuses = ["current", "archived"] + (["superseded"] if f.include_superseded else [])
    if hit.doc_status not in statuses:
        return False
    if any(tag not in hit.tags for tag in f.tags):
        return False
    if f.folders:
        ancestors = path_ancestors_folded(hit.path)
        if not any(normalize_path_key(folder) in ancestors for folder in f.folders):
            return False
    if f.formats and hit.format not in f.formats:
        return False
    if f.document_ids and hit.document_id not in f.document_ids:
        return False
    if f.since is not None and (hit.doc_date is None or hit.doc_date < f.since):
        return False
    if f.until is not None and (hit.doc_date is None or hit.doc_date > f.until):
        return False
    if f.acl_prefixes is not None:
        exact = path_ancestors(hit.path)
        allowed = "" in f.acl_prefixes or any(
            prefix in exact or hit.path.startswith(prefix + "/") for prefix in f.acl_prefixes
        )
        if not allowed:
            return False
    return True


def hybrid_search(
    query: str,
    filters: SearchFilters | None = None,
    *,
    size: int | None = None,
    per_group_limit: int = 2,
) -> list[SearchHit]:
    """Run the hybrid k-NN + BM25 search and return deduplicated hits.

    The ``tags`` filter values (``--tag``, inline ``#tag``) are normalized and
    alias-resolved (task 37) before the query body is built.

    Args:
        query: The raw query text.
        filters: Filters applied to both legs; ``None`` means the default
            gates only (latest versions, current/archived documents).
        size: Candidates to fetch; defaults to ``retrieval.candidates`` (50).
        per_group_limit: Maximum hits per document/page/section group.

    Returns:
        The hits in RRF rank order (best first), ranks re-numbered after the
        deduplication. No score threshold is applied (PLAN.md §6).

    Raises:
        KeyError: If the embedding task is not configured.
        EndpointError: If the embedding endpoint fails.
    """
    settings = get_settings()
    fetch_size = size or settings.retrieval.candidates
    active_filters = filters if filters is not None else SearchFilters()
    active_filters = _resolve_tag_filters(settings, active_filters)
    query_vector = get_embedder().embed_query(query, task="retrieval")
    body = build_hybrid_body(query, query_vector, active_filters, fetch_size)
    response = get_client().search(
        index=settings.opensearch.chunks_alias,
        body=body,
        params={"search_pipeline": PIPELINE_NAME},
    )
    hits = [
        _to_hit(raw_hit, rank)
        for rank, raw_hit in enumerate(response.get("hits", {}).get("hits", []), start=1)
    ]
    result = _dedupe(hits, per_group_limit)
    logger.info("hybrid search %r: %d hit(s) after dedup", query, len(result))
    return result
