"""Integration tests for OpenSearch admin helpers (real OpenSearch 3.8.0)."""

from __future__ import annotations

import pytest
from opensearchpy import OpenSearch

from mynotes.config import Settings
from mynotes.index.mappings import PIPELINE_NAME, chunks_index_body
from mynotes.index.opensearch import (
    create_index,
    current_physical_index,
    ensure_indices,
    ensure_pipeline,
    get_client,
    inspect_live,
    physical_name,
    swap_alias,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def os_client(os_settings: Settings) -> OpenSearch:
    """A client built through :func:`get_client` with the patched settings."""
    return get_client()


def _physical_indices(client: OpenSearch, alias: str) -> list[str]:
    """All physical indices whose name starts with ``alias``."""
    return sorted(client.indices.get(index=f"{alias}*"))


def test_ensure_indices_creates_alias_and_is_idempotent(
    os_client: OpenSearch, os_settings: Settings
) -> None:
    """First call creates <alias>_v1 + alias; the second call changes nothing."""
    result = ensure_indices(os_client, os_settings)
    assert result == {
        os_settings.opensearch.chunks_alias: f"{os_settings.opensearch.chunks_alias}_v1",
        os_settings.opensearch.entities_alias: f"{os_settings.opensearch.entities_alias}_v1",
    }
    assert _physical_indices(os_client, os_settings.opensearch.chunks_alias) == [
        f"{os_settings.opensearch.chunks_alias}_v1"
    ]

    again = ensure_indices(os_client, os_settings)
    assert again == result
    assert _physical_indices(os_client, os_settings.opensearch.chunks_alias) == [
        f"{os_settings.opensearch.chunks_alias}_v1"
    ]


def test_ensure_pipeline_upserts_and_is_readable(
    os_client: OpenSearch, os_settings: Settings
) -> None:
    """After the upsert, GET /_search/pipeline returns the configured body."""
    ensure_pipeline(os_client, os_settings)
    ensure_pipeline(os_client, os_settings)  # upsert, must not raise
    stored = os_client.transport.perform_request("GET", f"/_search/pipeline/{PIPELINE_NAME}")
    combination = stored[PIPELINE_NAME]["phase_results_processors"][0]["score-ranker-processor"][
        "combination"
    ]
    fusion = os_settings.retrieval.fusion
    assert combination["technique"] == "rrf"
    assert combination["rank_constant"] == fusion.rank_constant
    assert combination["parameters"]["weights"] == fusion.hybrid_weights


def test_analyzers_fold_case_and_accents(os_client: OpenSearch, os_settings: Settings) -> None:
    """`folded` strips diacritics; the `folded_keyword` normalizer does too."""
    ensure_indices(os_client, os_settings)
    physical = f"{os_settings.opensearch.chunks_alias}_v1"

    folded = os_client.indices.analyze(
        index=physical, body={"analyzer": "folded", "text": "Frigorífico"}
    )
    assert [token["token"] for token in folded["tokens"]] == ["frigorifico"]

    normalized = os_client.indices.analyze(
        index=physical, body={"normalizer": "folded_keyword", "text": "Área-Técnica"}
    )
    assert [token["token"] for token in normalized["tokens"]] == ["area-tecnica"]


def test_inspect_live_clean_then_dimension_problem(
    os_client: OpenSearch, os_settings: Settings
) -> None:
    """No problems after ensure; a dim mismatch is reported with the fix."""
    ensure_indices(os_client, os_settings)
    ensure_pipeline(os_client, os_settings)

    clean = inspect_live(os_client, os_settings)
    assert [report.alias for report in clean] == [
        os_settings.opensearch.chunks_alias,
        os_settings.opensearch.entities_alias,
    ]
    assert all(report.problems == [] for report in clean)
    assert all(report.has_folded_subfields for report in clean)

    mismatched = os_settings.model_copy(deep=True)
    mismatched.models.embedding.dim = 16
    problems = inspect_live(os_client, mismatched)
    assert all(report.problems for report in problems)
    for report in problems:
        assert any("dimension" in problem for problem in report.problems)
        assert any("mynotes index rebuild" in problem for problem in report.problems)


def test_swap_alias_moves_atomically(os_client: OpenSearch, os_settings: Settings) -> None:
    """The alias points to _v2 only after the swap; _v1 keeps its mapping."""
    ensure_indices(os_client, os_settings)
    alias = os_settings.opensearch.chunks_alias
    v2 = physical_name(alias, 2)
    create_index(os_client, v2, chunks_index_body(os_settings.models.embedding.dim))

    swap_alias(os_client, alias, v2)

    assert current_physical_index(os_client, alias) == v2
    assert _physical_indices(os_client, alias) == [f"{alias}_v1", v2]
    v1_aliases = os_client.indices.get_alias(index=f"{alias}_v1")
    assert alias not in v1_aliases[f"{alias}_v1"]["aliases"]
