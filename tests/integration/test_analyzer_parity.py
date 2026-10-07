"""Permanent parity checks from the task-12 spike (real OpenSearch 3.8.0).

Pins two guarantees on the chunks index created by ``ensure_indices``:

- the ``folded_keyword`` normalizer returns exactly what Python's
  ``normalize_key`` returns for every entry in ``FOLDING_TEST_STRINGS``;
- the ``pt_folded``/``en_folded`` analyzers collapse the word groups the spike
  decided with to a single token and keep the documented divergences of the
  kept filter order (``lowercase -> stop -> stemmer -> asciifolding``), so any
  accidental filter-order or stemmer change fails loudly.
"""

from __future__ import annotations

import pytest
from opensearchpy import OpenSearch

from mynotes.config import Settings
from mynotes.index.opensearch import ensure_indices, get_client
from mynotes.text.normalize import FOLDING_TEST_STRINGS

pytestmark = pytest.mark.integration


@pytest.fixture
def os_client(os_settings: Settings) -> OpenSearch:
    """A client built through :func:`get_client` with the patched settings."""
    return get_client()


def _tokens(os_client: OpenSearch, index: str, body: dict[str, str]) -> list[str]:
    """Run ``_analyze`` on the index and return the produced tokens."""
    response = os_client.indices.analyze(index=index, body=body)
    return [token["token"] for token in response["tokens"]]


def test_folded_keyword_matches_normalize_key(os_client: OpenSearch, os_settings: Settings) -> None:
    """Every folding test string normalizes to its expected Python key."""
    ensure_indices(os_client, os_settings)
    physical = f"{os_settings.opensearch.chunks_alias}_v1"
    for value, expected in FOLDING_TEST_STRINGS:
        tokens = _tokens(os_client, physical, {"normalizer": "folded_keyword", "text": value})
        assert " ".join(tokens) == expected, f"{value!r} -> {tokens!r}, expected {expected!r}"


def test_analyzers_collapse_single_token_groups(
    os_client: OpenSearch, os_settings: Settings
) -> None:
    """The word groups the spike kept as single-token stay single-token."""
    ensure_indices(os_client, os_settings)
    physical = f"{os_settings.opensearch.chunks_alias}_v1"
    single_token_groups: dict[str, list[list[str]]] = {
        "pt_folded": [
            ["coração", "coracao", "corações"],
            ["frigorífico", "frigorifico", "frigoríficos"],
            ["informação", "informacao", "informações"],
            ["área", "area", "áreas"],
            ["público", "publico", "públicos"],
        ],
        "en_folded": [
            ["idea", "ideas"],
            ["policy", "policies"],
            ["café", "cafe"],
        ],
    }
    for analyzer, groups in single_token_groups.items():
        for group in groups:
            outputs = [
                tuple(_tokens(os_client, physical, {"analyzer": analyzer, "text": word}))
                for word in group
            ]
            distinct = set(outputs)
            assert len(distinct) == 1, f"{analyzer} {group}: {outputs}"
            assert next(iter(distinct)) != (), f"{analyzer} {group}: dropped entirely"


def test_analyzers_pin_known_divergences(os_client: OpenSearch, os_settings: Settings) -> None:
    """The groups that diverge under the kept filter order stay as measured."""
    ensure_indices(os_client, os_settings)
    physical = f"{os_settings.opensearch.chunks_alias}_v1"
    pinned: dict[str, dict[str, list[str]]] = {
        "pt_folded": {
            "ação": ["acao"],
            "acao": ["acao"],
            "ações": ["acao"],
            "acoes": ["acoe"],
            "AÇÕES": ["acao"],
            "país": ["pai"],
            "pais": ["pai"],
            "países": ["pais"],
        },
        "en_folded": {"run": ["run"], "running": ["running"]},
    }
    for analyzer, expectations in pinned.items():
        for word, expected in expectations.items():
            tokens = _tokens(os_client, physical, {"analyzer": analyzer, "text": word})
            assert tokens == expected, f"{analyzer} {word!r} -> {tokens!r}, expected {expected!r}"
