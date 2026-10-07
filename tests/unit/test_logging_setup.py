"""Unit tests for mynotes.logging_setup."""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import pytest

from mynotes.config import Settings, mask_secret, reset_settings_cache
from mynotes.logging_setup import configure_logging

SECRET = "supersecretvalue123"
_NOISY = ("httpx", "httpcore", "urllib3", "opensearch")


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Run with no MYNOTES_* variables, no repo .env and no user config file."""
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


@pytest.fixture(autouse=True)
def _clean_root_logger():
    """Remove the mynotes handler and restore logger levels after each test."""
    root = logging.getLogger()
    saved_root_level = root.level
    saved_noisy_levels = {name: logging.getLogger(name).level for name in _NOISY}
    yield
    for handler in list(root.handlers):
        if handler.get_name() == "mynotes":
            root.removeHandler(handler)
            handler.close()
    root.setLevel(saved_root_level)
    for name, level in saved_noisy_levels.items():
        logging.getLogger(name).setLevel(level)


def _capture_stderr(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Replace ``sys.stderr`` so the handler installed by configure_logging writes there."""
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    return stream


def _patch_settings(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> Settings:
    settings = Settings(**overrides)
    monkeypatch.setattr("mynotes.logging_setup.get_settings", lambda: settings)
    return settings


def test_llm_secret_is_masked_in_text_output(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, models={"llm": {"api_key": SECRET}})
    stream = _capture_stderr(monkeypatch)
    configure_logging()

    logging.getLogger("mynotes.test").info("connecting with key %s", SECRET)

    output = stream.getvalue()
    assert SECRET not in output
    assert mask_secret(SECRET) in output
    assert "mynotes.test" in output


def test_all_settings_secrets_are_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    embedding_key = "embedding-key-4567"
    reranker_key = "reranker-key-890"
    pg_password = "pgpass901"
    os_password = "ospass9012"
    _patch_settings(
        monkeypatch,
        models={
            "embedding": {"api_key": embedding_key},
            "reranker": {"api_key": reranker_key},
            "llm": {"api_key": SECRET},
        },
        postgres={"dsn": f"postgresql+psycopg://user:{pg_password}@db:5432/mynotes"},
        opensearch={"password": os_password},
    )
    stream = _capture_stderr(monkeypatch)
    configure_logging()

    logging.getLogger("mynotes.test").warning(
        "e=%s r=%s llm=%s dsn=%s pw=%s os=%s",
        embedding_key,
        reranker_key,
        SECRET,
        f"postgresql+psycopg://user:{pg_password}@db:5432/mynotes",
        pg_password,
        os_password,
    )

    output = stream.getvalue()
    assert SECRET not in output
    assert embedding_key not in output
    assert reranker_key not in output
    assert pg_password not in output
    assert os_password not in output
    assert mask_secret(SECRET) in output
    assert mask_secret(os_password) in output


def test_short_secrets_are_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, models={"llm": {"api_key": "abc12"}})
    stream = _capture_stderr(monkeypatch)
    configure_logging()

    logging.getLogger("mynotes.test").info("short key abc12 stays")

    assert "abc12" in stream.getvalue()


def test_json_output_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, models={"llm": {"api_key": SECRET}})
    stream = _capture_stderr(monkeypatch)
    configure_logging(json=True)

    logging.getLogger("mynotes.json").warning("hello %s", "world", extra={"request_id": 7})

    line = stream.getvalue().strip()
    payload = json.loads(line)
    assert payload["level"] == "WARNING"
    assert payload["logger"] == "mynotes.json"
    assert payload["msg"] == "hello world"
    assert payload["request_id"] == 7
    assert "ts" in payload


def test_configure_twice_keeps_one_handler() -> None:
    configure_logging()
    configure_logging(level="DEBUG", json=True)

    mine = [h for h in logging.getLogger().handlers if h.get_name() == "mynotes"]
    assert len(mine) == 1
    assert logging.getLogger().level == logging.DEBUG


def test_noisy_libraries_are_quiet() -> None:
    configure_logging()

    for name in _NOISY:
        assert logging.getLogger(name).level == logging.WARNING


def test_unknown_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown logging level"):
        configure_logging(level="NOT_A_LEVEL")


def test_log_file_var_adds_rotating_file_handler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log_file = tmp_path / "worker.log"
    monkeypatch.setenv("MYNOTES_LOG_FILE", str(log_file))
    _patch_settings(monkeypatch, models={"llm": {"api_key": SECRET}})
    stream = _capture_stderr(monkeypatch)
    configure_logging()

    file_handlers = [
        handler
        for handler in logging.getLogger().handlers
        if isinstance(handler, RotatingFileHandler)
    ]
    assert len(file_handlers) == 1
    assert file_handlers[0].maxBytes == 10 * 1024 * 1024
    assert file_handlers[0].backupCount == 5

    logging.getLogger("mynotes.file").warning("to file with key %s", SECRET)

    written = log_file.read_text(encoding="utf-8")
    assert "mynotes.file" in written
    assert SECRET not in written
    assert mask_secret(SECRET) in written
    assert SECRET in stream.getvalue() or mask_secret(SECRET) in stream.getvalue()


def test_log_file_var_reconfigured_replaces_handler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MYNOTES_LOG_FILE", str(tmp_path / "worker.log"))
    configure_logging()
    configure_logging()

    file_handlers = [
        handler
        for handler in logging.getLogger().handlers
        if isinstance(handler, RotatingFileHandler)
    ]
    assert len(file_handlers) == 1


def test_log_file_var_unopenable_is_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MYNOTES_LOG_FILE", str(tmp_path / "missing-dir" / "worker.log"))
    stream = _capture_stderr(monkeypatch)
    configure_logging()

    file_handlers = [
        handler
        for handler in logging.getLogger().handlers
        if isinstance(handler, RotatingFileHandler)
    ]
    assert file_handlers == []
    logging.getLogger("mynotes.still").info("stderr keeps working")
    assert "mynotes.still" in stream.getvalue()
    assert "could not open log file" in stream.getvalue()
