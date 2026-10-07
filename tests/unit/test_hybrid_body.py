"""Unit tests for the hybrid query body and its filter clauses (task 24).

Pure functions only: no network, no Docker, no model endpoints.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import typer

from mynotes.cli import _parse_since
from mynotes.retrieval.hybrid import BM25_FIELDS, build_filter_clauses, build_hybrid_body
from mynotes.retrieval.types import SearchFilters

_VECTOR = [0.5, 0.0, -0.5]


def _status_clauses(clauses: list[dict]) -> list[dict]:
    """The ``doc_status`` terms clauses of a filter list."""
    return [clause for clause in clauses if "doc_status" in clause.get("terms", {})]


def test_knn_leg_first_and_same_filters_in_both_legs() -> None:
    """The hybrid body is knn then BM25, both legs carrying the same filters."""
    body = build_hybrid_body("frigorifico", _VECTOR, SearchFilters(tags=["x"]), size=7)

    assert body["size"] == 7
    assert body["_source"] == {"excludes": ["embedding"]}
    legs = body["query"]["hybrid"]["queries"]
    assert list(legs[0]) == ["knn"]
    assert list(legs[1]) == ["bool"]
    knn = legs[0]["knn"]["embedding"]
    assert knn["vector"] == _VECTOR
    assert knn["k"] == 7
    bm25 = legs[1]["bool"]
    assert bm25["must"] == [
        {"multi_match": {"query": "frigorifico", "type": "most_fields", "fields": BM25_FIELDS}}
    ]
    assert BM25_FIELDS == [
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
    assert knn["filter"]["bool"]["filter"] == bm25["filter"]
    assert bm25["filter"]


def test_default_filters_gate_latest_and_status() -> None:
    """Defaults filter to latest versions and current/archived documents."""
    default = build_filter_clauses(SearchFilters())

    assert {"term": {"is_latest": True}} in default
    statuses = _status_clauses(default)
    assert len(statuses) == 1
    assert statuses[0]["terms"]["doc_status"] == ["current", "archived"]

    superseded = build_filter_clauses(SearchFilters(include_superseded=True))
    assert _status_clauses(superseded)[0]["terms"]["doc_status"] == [
        "current",
        "archived",
        "superseded",
    ]

    old_versions = build_filter_clauses(SearchFilters(include_old_versions=True))
    assert {"term": {"is_latest": True}} not in old_versions


def test_acl_uses_exact_fields_and_empty_list_blocks_everything() -> None:
    """ACL clauses match only the exact ``path_ancestors``/``path`` fields."""
    clauses = build_filter_clauses(SearchFilters(acl_prefixes=["Ideas/Alpha"]))
    assert clauses == [
        {"term": {"is_latest": True}},
        {"terms": {"doc_status": ["current", "archived"]}},
        {
            "bool": {
                "should": [
                    {"term": {"path_ancestors": "Ideas/Alpha"}},
                    {"prefix": {"path": "Ideas/Alpha/"}},
                ],
                "minimum_should_match": 1,
            }
        },
    ]

    empty = build_filter_clauses(SearchFilters(acl_prefixes=[]))
    assert {"bool": {"must_not": {"match_all": {}}}} in empty

    none = build_filter_clauses(SearchFilters(acl_prefixes=None))
    assert {"bool": {"must_not": {"match_all": {}}}} not in none


def test_folders_use_folded_ancestors() -> None:
    """Folder prefixes are folded per segment into ``path_ancestors_folded``."""
    clauses = build_filter_clauses(SearchFilters(folders=["Projetos/Ação"]))
    folder = [clause for clause in clauses if "path_ancestors_folded" in str(clause)]
    assert folder == [
        {
            "bool": {
                "should": [{"term": {"path_ancestors_folded": "projetos/acao"}}],
                "minimum_should_match": 1,
            }
        }
    ]


def test_tags_formats_dates_and_workspace_clauses() -> None:
    """Tags become one term each; formats/dates/workspace render as specified."""
    clauses = build_filter_clauses(
        SearchFilters(
            tags=["area-tecnica", "projeto-x"],
            formats=["md", "pdf"],
            since=datetime(2026, 8, 1, tzinfo=UTC),
            until=datetime(2026, 9, 15, tzinfo=UTC),
            workspace_id=7,
        )
    )

    assert {"term": {"tags": "area-tecnica"}} in clauses
    assert {"term": {"tags": "projeto-x"}} in clauses
    assert {"terms": {"format": ["md", "pdf"]}} in clauses
    assert {
        "range": {
            "doc_date": {
                "gte": "2026-08-01T00:00:00+00:00",
                "lte": "2026-09-15T00:00:00+00:00",
            }
        }
    } in clauses
    assert {"term": {"workspace_id": "7"}} in clauses


def test_cli_since_parsing() -> None:
    """``--since`` accepts YYYY-MM and YYYY-MM-DD as UTC midnight."""
    assert _parse_since("2026-08") == datetime(2026, 8, 1, tzinfo=UTC)
    assert _parse_since("2026-08-15") == datetime(2026, 8, 15, tzinfo=UTC)
    with pytest.raises(typer.BadParameter):
        _parse_since("2026")
    with pytest.raises(typer.BadParameter):
        _parse_since("not-a-date")
