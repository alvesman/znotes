"""Without model endpoints mynotes refuses clearly instead of calling someone else's server."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mynotes.config import (
    EmbeddingConfig,
    LLMConfig,
    ModelsNotConfigured,
    Settings,
    reset_settings_cache,
)


@pytest.fixture(autouse=True)
def _no_user_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An empty cwd, no MYNOTES_* variables and no user config: the true defaults."""
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("MYNOTES_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(
        "mynotes.config.platformdirs.user_config_dir",
        lambda *args, **kwargs: str(tmp_path / "user-config"),
    )
    reset_settings_cache()
    yield
    reset_settings_cache()


def test_clients_refuse_missing_endpoints() -> None:
    from mynotes.embed.client import EmbeddingClient
    from mynotes.llm.chat import ChatClient

    with pytest.raises(ModelsNotConfigured, match=r"models\.embedding\.base_url"):
        EmbeddingClient(EmbeddingConfig())
    with pytest.raises(ModelsNotConfigured, match=r"models\.llm\.model"):
        ChatClient(LLMConfig(base_url="http://llm.test/v1"))


def test_init_stops_before_the_databases(monkeypatch: pytest.MonkeyPatch) -> None:
    from mynotes.core import setup

    monkeypatch.setattr(setup, "get_settings", lambda: Settings())
    monkeypatch.setattr(setup, "_resolve_yaml_path", lambda: None)
    called: list[str] = []
    monkeypatch.setattr(setup, "_check_postgres", lambda: called.append("postgres"))
    results = setup.run_init()
    assert [r.name for r in results] == ["config", "models"]
    assert results[-1].level == "fail" and "models.embedding.dim" in results[-1].detail
    assert called == []  # indices are created with the dimension: nothing else runs


def test_reranker_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    from mynotes.core import setup
    from mynotes.retrieval import rerank_client

    unconfigured = Settings()
    monkeypatch.setattr(setup, "get_settings", lambda: unconfigured)
    monkeypatch.setattr(rerank_client, "get_settings", lambda: unconfigured)
    [result] = setup._check_reranker()
    assert result.level == "warn" and "not configured" in result.detail
    assert rerank_client.reranker_configured() is False
    with pytest.raises(ModelsNotConfigured):
        rerank_client.get_reranker()
