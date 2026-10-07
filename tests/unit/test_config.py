"""Unit tests for mynotes.config."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from mynotes.config import Settings, get_settings, mask_secret, reset_settings_cache

EXAMPLE_YAML = Path(__file__).resolve().parents[2] / "mynotes.example.yaml"


@pytest.fixture(autouse=True)
def _isolated_config_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Run every test in an empty cwd with no MYNOTES_* variables and no user config."""
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


def test_defaults_load_without_yaml_or_env() -> None:
    settings = get_settings()
    assert settings.workspace == "default"
    assert settings.roots == []
    assert settings.models.embedding.dim is None  # no model defaults: each install configures
    assert settings.models.llm.limits.timeout_s == 120


def test_missing_model_settings_lists_what_to_configure() -> None:
    from mynotes.config import missing_model_settings

    assert missing_model_settings(Settings()) == [
        "models.embedding.base_url",
        "models.embedding.model",
        "models.embedding.dim",
        "models.llm.base_url",
        "models.llm.model",
    ]
    configured = Settings(
        models={
            "embedding": {"base_url": "http://e/v1", "model": "m", "dim": 768},
            "llm": {"base_url": "http://l/v1", "model": "c"},
        }
    )
    assert missing_model_settings(configured) == []  # the reranker is optional
    configured.models.reranker.url = "http://r/v1/rerank"
    assert missing_model_settings(configured) == ["models.reranker.model"]


def test_yaml_values_are_applied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    yaml_file = tmp_path / "my.yaml"
    yaml_file.write_text(
        "workspace: research\nscan:\n  settle_seconds: 3\nmodels:\n  embedding:\n    dim: 8\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MYNOTES_CONFIG", str(yaml_file))
    settings = Settings()
    assert settings.workspace == "research"
    assert settings.scan.settle_seconds == 3
    assert settings.models.embedding.dim == 8


def test_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    yaml_file = tmp_path / "my.yaml"
    yaml_file.write_text(
        "models:\n  llm:\n    model: a\n    max_tokens: 1000\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MYNOTES_CONFIG", str(yaml_file))
    monkeypatch.setenv("MYNOTES_LLM_MODEL", "b")
    settings = Settings()
    assert settings.models.llm.model == "b"
    assert settings.models.llm.max_tokens == 1000


def test_mynotes_yaml_in_cwd_is_used(tmp_path: Path) -> None:
    (tmp_path / "mynotes.yaml").write_text("workspace: cwd\n", encoding="utf-8")
    assert Settings().workspace == "cwd"


def test_dotenv_in_cwd_is_read(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        'MYNOTES_LLM_CONTEXT_WINDOW="64000"  # thinking budget\nMYNOTES_EMBEDDING_DIM=8\n',
        encoding="utf-8",
    )
    settings = Settings()
    assert settings.models.llm.context_window == 64000
    assert settings.models.embedding.dim == 8


def test_llm_timeout_env_reaches_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MYNOTES_LLM_TIMEOUT_S", "90")
    settings = Settings()
    assert settings.models.llm.limits.timeout_s == 90


def test_empty_env_values_do_not_clobber_defaults(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "MYNOTES_LLM_API_KEY=\nMYNOTES_EMBEDDING_API_KEY=\n",
        encoding="utf-8",
    )
    settings = Settings()
    assert settings.models.llm.api_key is None
    assert settings.models.embedding.api_key is None


def test_secrets_are_masked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MYNOTES_LLM_API_KEY", "supersecretkey123")
    monkeypatch.setenv("MYNOTES_PG_DSN", "postgresql+psycopg://user:hunter2@db/mynotes")
    settings = Settings()
    rendered = f"{settings!r} {settings!s}"
    assert "supersecretkey123" not in rendered
    assert "hunter2" not in rendered
    redacted = settings.redacted_dict()
    assert "supersecretkey123" not in str(redacted)
    assert "hunter2" not in str(redacted)
    assert redacted["models"]["llm"]["api_key"] == mask_secret("supersecretkey123")
    assert redacted["postgres"]["dsn"] == mask_secret(
        "postgresql+psycopg://user:hunter2@db/mynotes"
    )


def test_invalid_hybrid_weights_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(retrieval={"fusion": {"hybrid_weights": [0.6, 0.2]}})


def test_zero_embedding_dim_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(models={"embedding": {"dim": 0}})


def test_duplicate_root_names_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(roots=[{"name": "a", "path": "/tmp/a"}, {"name": "a", "path": "/tmp/b"}])


def test_root_path_expands_user() -> None:
    settings = Settings(roots=[{"name": "ideas", "path": "~/notes"}])
    assert settings.roots[0].path == Path.home() / "notes"


def test_example_yaml_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MYNOTES_CONFIG", str(EXAMPLE_YAML))
    settings = Settings()
    assert settings.models.embedding.dim is None  # endpoints are left for the user to fill in
    assert settings.models.llm.thinking_model is False
    assert [root.name for root in settings.roots] == ["ideas", "shared"]


def test_mask_secret() -> None:
    assert mask_secret(None) == "(none)"
    assert mask_secret("") == "(none)"
    assert mask_secret("abcdefghij") == "abc…ij (10 chars)"
