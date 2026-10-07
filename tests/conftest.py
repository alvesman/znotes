"""Shared pytest fixtures for mynotes tests.

Test markers (`integration`, `live`) are registered in `pyproject.toml`
under `[tool.pytest.ini_options]`. Live tests (task 06) are skipped unless
`MYNOTES_LIVE_TESTS=1`.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from mynotes.chunk.tokenizer import WhitespaceCounter, reset_token_counter, set_token_counter
from mynotes.config import Settings, reset_settings_cache
from mynotes.llm.limits import reset_guards


@pytest.fixture(autouse=True)
def _fresh_settings_and_guards() -> Iterator[None]:
    """Reset the settings and guard caches so every test starts clean."""
    reset_settings_cache()
    reset_guards()
    yield


@pytest.fixture(autouse=True)
def _whitespace_token_counter() -> Iterator[None]:
    """Install WhitespaceCounter so no test ever downloads a tokenizer."""
    set_token_counter(WhitespaceCounter())
    yield
    reset_token_counter()


@pytest.fixture(autouse=True)
def _unregister_version_retrieval() -> Iterator[None]:
    """Keep the task-35 version expander/notes out of tests by default.

    Both register themselves when the retrieval pipeline is imported and open
    database sessions, so unit tests that assemble contexts or run retrieval
    would otherwise try to connect. Tests that exercise them re-register with
    ``mynotes.retrieval.version_expand.register()``; the import-time state is
    restored after every test.
    """
    from mynotes.retrieval.assemble import unregister_extra_block
    from mynotes.retrieval.pipeline import unregister_expander

    unregister_expander("version")
    unregister_extra_block("version_notes")
    yield
    from mynotes.retrieval import version_expand

    version_expand.register()


@pytest.fixture(autouse=True)
def _unregister_graph_retrieval() -> Iterator[None]:
    """Keep the task-43 graph expander/facts out of tests by default.

    Same reason as the version fixture: the graph expander opens database
    sessions and would make unit tests that run retrieval or assembly try to
    connect. Tests that exercise it re-register with
    ``mynotes.retrieval.graph_expand.register()``; the import-time state is
    restored after every test.
    """
    from mynotes.retrieval.assemble import unregister_extra_block
    from mynotes.retrieval.pipeline import unregister_expander

    unregister_expander("graph")
    unregister_extra_block("graph_facts")
    yield
    from mynotes.retrieval import graph_expand

    graph_expand.register()


@pytest.fixture
def test_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Small in-memory settings, patched into every ``get_settings`` import site."""
    settings = Settings(
        models={
            "embedding": {"dim": 8, "base_url": "http://embed.test/v1", "model": "test-embed"},
            "reranker": {"url": "http://rerank.test/v1/rerank", "model": "test-rerank"},
            "llm": {"base_url": "http://llm.test/v1", "model": "test-llm"},
        }
    )
    for target in (
        "mynotes.config.get_settings",
        "mynotes.llm.limits.get_settings",
        "mynotes.llm.chat.get_settings",
        "mynotes.embed.client.get_settings",
        "mynotes.retrieval.rerank_client.get_settings",
    ):
        monkeypatch.setattr(target, lambda: settings)
    return settings


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip items marked ``live`` unless ``MYNOTES_LIVE_TESTS=1``."""
    if os.environ.get("MYNOTES_LIVE_TESTS") == "1":
        return
    skip_live = pytest.mark.skip(reason="live tests need MYNOTES_LIVE_TESTS=1")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
