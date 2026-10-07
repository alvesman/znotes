"""Unit tests for the OpenSearch index bodies and RRF pipeline body."""

from __future__ import annotations

import pytest

from mynotes.index.mappings import (
    EN_FOLDED_FILTERS,
    PIPELINE_NAME,
    PT_FOLDED_FILTERS,
    analysis_settings,
    chunks_index_body,
    entities_index_body,
    rrf_pipeline_body,
)


def test_chunks_body_embedding_dim_and_engine() -> None:
    """`chunks_index_body(8)` builds a knn_vector of dim 8 on the lucene engine."""
    embedding = chunks_index_body(8)["mappings"]["properties"]["embedding"]
    assert embedding["dimension"] == 8
    assert embedding["method"]["engine"] == "lucene"
    assert embedding["method"]["space_type"] == "cosinesimil"
    assert embedding["method"]["parameters"] == {"m": 48, "ef_construction": 256}


def test_chunks_body_settings_and_strict_mapping() -> None:
    """The chunk index enables knn, uses one shard and a strict mapping."""
    body = chunks_index_body(8)
    assert body["settings"]["index"]["knn"] is True
    assert body["settings"]["index"]["number_of_shards"] == 1
    assert body["settings"]["index"]["number_of_replicas"] == 0
    assert body["mappings"]["dynamic"] == "strict"


def test_entities_body_matches_spec_fields() -> None:
    """The entity index has the spec'd fields with kw subfields on names."""
    props = entities_index_body(8)["mappings"]["properties"]
    assert props["entity_id"] == {"type": "keyword"}
    assert props["type"] == {"type": "keyword", "normalizer": "folded_keyword"}
    assert props["canonical_name"]["fields"]["kw"]["normalizer"] == "folded_keyword"
    for name in ("canonical_name", "aliases"):
        assert props[name]["fields"]["kw"]["type"] == "keyword"
    assert props["description"]["fields"]["pt"]["analyzer"] == "pt_folded"
    assert props["description"]["fields"]["en"]["analyzer"] == "en_folded"


def test_analysis_default_uses_asciifolding() -> None:
    """Default analysis matches PLAN.md §6.1 exactly."""
    analysis = analysis_settings("asciifolding")
    assert analysis["normalizer"]["folded_keyword"]["filter"] == [
        "lowercase",
        "asciifolding",
        "trim",
    ]
    assert analysis["normalizer"]["folded_keyword"]["char_filter"] == ["ws_collapse"]
    assert analysis["char_filter"]["ws_collapse"]["pattern"] == r"(?U)\s+"
    assert analysis["analyzer"]["folded"]["filter"] == ["lowercase", "asciifolding"]
    assert analysis["analyzer"]["pt_folded"]["filter"] == PT_FOLDED_FILTERS
    assert analysis["analyzer"]["en_folded"]["filter"] == EN_FOLDED_FILTERS
    assert analysis["filter"]["pt_stemmer"]["language"] == "light_portuguese"
    assert analysis["filter"]["en_stemmer"]["language"] == "light_english"


def test_analysis_icu_replaces_folding_only() -> None:
    """ICU folding replaces asciifolding; lowercase stays in the stemmed analyzers."""
    analysis = analysis_settings("icu")
    assert analysis["normalizer"]["folded_keyword"]["filter"] == ["icu_folding", "trim"]
    assert analysis["normalizer"]["folded_keyword"]["char_filter"] == ["ws_collapse"]
    assert analysis["analyzer"]["folded"]["filter"] == ["icu_folding"]
    assert analysis["analyzer"]["pt_folded"]["filter"] == [
        "lowercase",
        "pt_stop",
        "pt_stemmer",
        "icu_folding",
    ]
    assert analysis["analyzer"]["en_folded"]["filter"] == [
        "lowercase",
        "en_stop",
        "en_stemmer",
        "icu_folding",
    ]


def test_folded_text_fields_have_pt_en_subfields() -> None:
    """text/title/heading_path_text carry the pt/en stemmed subfields."""
    props = chunks_index_body(8)["mappings"]["properties"]
    for name in ("text", "title", "heading_path_text"):
        field = props[name]
        assert field["analyzer"] == "folded"
        assert field["fields"]["pt"]["analyzer"] == "pt_folded"
        assert field["fields"]["en"]["analyzer"] == "en_folded"


def test_acl_ancestors_stay_exact() -> None:
    """`path_ancestors` is exact (ACL); only the folded variant is normalized."""
    props = chunks_index_body(8)["mappings"]["properties"]
    assert props["path_ancestors"] == {"type": "keyword"}
    assert props["path_ancestors_folded"] == {
        "type": "keyword",
        "normalizer": "folded_keyword",
    }
    assert props["path"]["type"] == "keyword"
    assert "normalizer" not in props["path"]
    assert props["path"]["fields"]["folded"]["normalizer"] == "folded_keyword"


def test_rrf_pipeline_body() -> None:
    """The pipeline body carries technique, rank_constant and weights."""
    body = rrf_pipeline_body(60, [0.5, 0.5])
    processor = body["phase_results_processors"][0]["score-ranker-processor"]
    combination = processor["combination"]
    assert combination["technique"] == "rrf"
    assert combination["rank_constant"] == 60
    assert combination["parameters"]["weights"] == [0.5, 0.5]
    assert body["description"].startswith("mynotes hybrid search")


def test_pipeline_name_is_stable() -> None:
    """The pipeline name matches PLAN.md and cannot drift silently."""
    assert PIPELINE_NAME == "mynotes-hybrid-rrf"


@pytest.mark.parametrize("folding", ["asciifolding", "icu"])
def test_both_index_bodies_share_analysis(folding: str) -> None:
    """Both index bodies embed the same analysis settings for a folding choice."""
    chunks = chunks_index_body(8, folding)["settings"]["analysis"]
    entities = entities_index_body(8, folding)["settings"]["analysis"]
    assert chunks == entities == analysis_settings(folding)
