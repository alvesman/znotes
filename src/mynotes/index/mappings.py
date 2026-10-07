"""Index and search-pipeline bodies for the mynotes OpenSearch indices.

Everything the versioned physical indices need lives here: the shared
analysis settings (case- and accent-insensitive matching, PLAN.md §6.1), the
``chunks`` and ``entities`` mapping bodies, and the RRF hybrid-search pipeline
body. Later tasks (indexing, search, rebuild) build on these constants; the
mappings use ``dynamic: "strict"``, so new document fields must be added here
first.
"""

from __future__ import annotations

from typing import Any, Literal

PIPELINE_NAME = "mynotes-hybrid-rrf"

PT_FOLDED_FILTERS = ["lowercase", "pt_stop", "pt_stemmer", "asciifolding"]
EN_FOLDED_FILTERS = ["lowercase", "en_stop", "en_stemmer", "asciifolding"]

Folding = Literal["asciifolding", "icu"]


def _knn_vector(dim: int) -> dict[str, Any]:
    """k-NN vector field: Lucene HNSW, cosine space (PLAN.md §6)."""
    return {
        "type": "knn_vector",
        "dimension": dim,
        "method": {
            "name": "hnsw",
            "space_type": "cosinesimil",
            "engine": "lucene",
            "parameters": {"m": 48, "ef_construction": 256},
        },
    }


def _folded_text() -> dict[str, Any]:
    """Text field analyzed with ``folded``, plus stemmed pt/en subfields."""
    return {
        "type": "text",
        "analyzer": "folded",
        "fields": {
            "pt": {"type": "text", "analyzer": "pt_folded"},
            "en": {"type": "text", "analyzer": "en_folded"},
        },
    }


def _folded_text_with_kw() -> dict[str, Any]:
    """``_folded_text`` plus an exact keyword subfield for entity names."""
    field = _folded_text()
    field["fields"]["kw"] = {"type": "keyword", "normalizer": "folded_keyword"}
    return field


def _folded_keyword() -> dict[str, Any]:
    """Keyword field normalized with ``folded_keyword`` (filter values match folded)."""
    return {"type": "keyword", "normalizer": "folded_keyword"}


def analysis_settings(folding: Folding) -> dict[str, Any]:
    """Shared analysis settings for both indices (PLAN.md §6.1).

    Args:
        folding: ``"asciifolding"`` (default, plugin-free) or ``"icu"``
            (Unicode-complete, needs the ``analysis-icu`` plugin).

    Returns:
        The ``analysis`` settings: the ``folded_keyword`` normalizer, the
        pt/en stopword and stemmer filters, and the ``folded``, ``pt_folded``
        and ``en_folded`` analyzers.
    """
    fold_token = "icu_folding" if folding == "icu" else "asciifolding"
    simple_filters = ["icu_folding"] if folding == "icu" else ["lowercase", "asciifolding"]
    return {
        "normalizer": {
            "folded_keyword": {
                "type": "custom",
                "char_filter": ["ws_collapse"],
                "filter": [*simple_filters, "trim"],
            },
        },
        "char_filter": {
            # normalize_key collapses every Unicode whitespace run to one space;
            # the keyword normalizer must do the same or exact-key filters
            # (tags, path segments) drift from the Postgres *_norm columns.
            "ws_collapse": {"type": "pattern_replace", "pattern": r"(?U)\s+", "replacement": " "},
        },
        "filter": {
            "pt_stop": {"type": "stop", "stopwords": "_portuguese_"},
            "en_stop": {"type": "stop", "stopwords": "_english_"},
            "pt_stemmer": {"type": "stemmer", "language": "light_portuguese"},
            "en_stemmer": {"type": "stemmer", "language": "light_english"},
        },
        "analyzer": {
            "folded": {"tokenizer": "standard", "filter": list(simple_filters)},
            "pt_folded": {
                "tokenizer": "standard",
                "filter": ["lowercase", "pt_stop", "pt_stemmer", fold_token],
            },
            "en_folded": {
                "tokenizer": "standard",
                "filter": ["lowercase", "en_stop", "en_stemmer", fold_token],
            },
        },
    }


def _index_body(dim: int, folding: Folding, properties: dict[str, Any]) -> dict[str, Any]:
    """Shared index body: knn settings, single shard, analysis, strict mapping."""
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "analysis": analysis_settings(folding),
        },
        "mappings": {"dynamic": "strict", "properties": properties},
    }


def chunks_index_body(dim: int, folding: Folding = "asciifolding") -> dict[str, Any]:
    """Index body for the chunk index (PLAN.md §6 ``mynotes_chunks``).

    Args:
        dim: Embedding dimension (``models.embedding.dim``); never hard-coded.
        folding: Which folding filters the analyzers use.

    Returns:
        Settings + strict mapping for one physical chunk index.
    """
    properties: dict[str, Any] = {
        "chunk_id": {"type": "keyword"},
        "document_id": {"type": "keyword"},
        "version_id": {"type": "keyword"},
        "source_root_id": {"type": "keyword"},
        "workspace_id": {"type": "keyword"},
        "text": _folded_text(),
        "title": _folded_text(),
        "heading_path_text": _folded_text(),
        "section": {"type": "keyword"},
        "tags": _folded_keyword(),
        "format": _folded_keyword(),
        "language": _folded_keyword(),
        "doc_status": _folded_keyword(),
        # Exact for display and ACL; only the subfield is folded for user filters.
        "path": {
            "type": "keyword",
            "fields": {"folded": {"type": "keyword", "normalizer": "folded_keyword"}},
        },
        "path_ancestors": {"type": "keyword"},
        "path_ancestors_folded": _folded_keyword(),
        "doc_date": {"type": "date"},
        "is_latest": {"type": "boolean"},
        "is_table": {"type": "boolean"},
        "page": {"type": "integer"},
        "slide": {"type": "integer"},
        "ordinal": {"type": "integer"},
        "embedding": _knn_vector(dim),
        # Drift fingerprint for the reconciler (task 23): never queried, only
        # read back via _source.
        "meta_hash": {"type": "keyword", "index": False},
    }
    return _index_body(dim, folding, properties)


def entities_index_body(dim: int, folding: Folding = "asciifolding") -> dict[str, Any]:
    """Index body for the entity index (PLAN.md §6 ``mynotes_entities``).

    Args:
        dim: Embedding dimension (``models.embedding.dim``); never hard-coded.
        folding: Which folding filters the analyzers use.

    Returns:
        Settings + strict mapping for one physical entity index.
    """
    properties: dict[str, Any] = {
        "entity_id": {"type": "keyword"},
        "workspace_id": {"type": "keyword"},
        "type": _folded_keyword(),
        "status": _folded_keyword(),
        "canonical_name": _folded_text_with_kw(),
        "aliases": _folded_text_with_kw(),
        "description": _folded_text(),
        "embedding": _knn_vector(dim),
    }
    return _index_body(dim, folding, properties)


def rrf_pipeline_body(rank_constant: int, weights: list[float]) -> dict[str, Any]:
    """Search-pipeline body for RRF hybrid fusion (PLAN.md §6).

    Args:
        rank_constant: ``retrieval.fusion.rank_constant`` (default 20).
        weights: Per-leg weights from ``retrieval.fusion.hybrid_weights``.

    Returns:
        The ``PUT /_search/pipeline/{PIPELINE_NAME}`` body.
    """
    return {
        "description": "mynotes hybrid search: RRF over k-NN + BM25 legs",
        "phase_results_processors": [
            {
                "score-ranker-processor": {
                    "combination": {
                        "technique": "rrf",
                        "rank_constant": rank_constant,
                        "parameters": {"weights": list(weights)},
                    }
                }
            }
        ],
    }
