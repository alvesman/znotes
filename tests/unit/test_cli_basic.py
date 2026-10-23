"""Unit tests for the CLI commands of task 11.

The core setup functions are monkeypatched, so the tests never touch Docker,
Postgres, OpenSearch or the model endpoints.
"""

from __future__ import annotations

import json
import logging

import pytest
from typer.testing import CliRunner

from mynotes.cli import app
from mynotes.core.setup import CheckResult

runner = CliRunner()


def _result(name: str, level: str) -> CheckResult:
    """A canned :class:`CheckResult` with the given level."""
    return CheckResult(name=name, ok=level != "fail", detail=f"{name} detail", level=level)  # type: ignore[arg-type]


def test_init_exits_zero_when_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, bool] = {}

    def fake_run_init(apply_migrations: bool = True) -> list[CheckResult]:
        seen["apply_migrations"] = apply_migrations
        return [_result("config", "ok"), _result("postgres reachable", "ok")]

    monkeypatch.setattr("mynotes.core.setup.run_init", fake_run_init)
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert seen["apply_migrations"] is True
    assert "postgres reachable" in result.output


def test_init_exits_zero_with_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mynotes.core.setup.run_init",
        lambda apply_migrations=True: [_result("tool: tesseract", "warn")],
    )
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0


def test_init_exits_one_on_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mynotes.core.setup.run_init",
        lambda apply_migrations=True: [
            _result("config", "ok"),
            _result("postgres reachable", "fail"),
        ],
    )
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 1


def test_init_no_migrate_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, bool] = {}

    def fake_run_init(apply_migrations: bool = True) -> list[CheckResult]:
        seen["apply_migrations"] = apply_migrations
        return []

    monkeypatch.setattr("mynotes.core.setup.run_init", fake_run_init)
    result = runner.invoke(app, ["init", "--no-migrate"])
    assert result.exit_code == 0
    assert seen["apply_migrations"] is False


def test_status_json_is_valid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    canned: dict = {
        "config_file": None,
        "workspace": "default",
        "roots": [],
        "postgres": {
            "files_by_status": {},
            "documents_by_status": {},
            "versions": 0,
            "chunks": 0,
            "jobs": {"queued": 0, "running": 0, "failed": 0},
            "outbox_unprocessed": 0,
        },
        "opensearch": {"aliases": {}, "problems": []},
        "endpoints": [],
    }
    monkeypatch.setattr("mynotes.core.setup.collect_status", lambda: canned)
    result = runner.invoke(app, ["status", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == canned


def test_status_panels_render(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mynotes.core.setup.collect_status",
        lambda: {
            "config_file": None,
            "workspace": "default",
            "roots": [{"name": "ideas", "path": "/tmp/notes", "source_root_id": 3}],
            "postgres": {"error": "unavailable: boom"},
            "opensearch": {"aliases": {}, "problems": []},
            "endpoints": [],
        },
    )
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "PostgreSQL" in result.output
    assert "unavailable: boom" in result.output


def test_check_endpoints_exits_zero_when_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, bool] = {}

    def fake_check_endpoints(include_llm: bool = True) -> list[CheckResult]:
        seen["include_llm"] = include_llm
        return [_result("embedding: dimension", "ok"), _result("llm: chat", "ok")]

    monkeypatch.setattr("mynotes.core.setup.check_endpoints", fake_check_endpoints)
    result = runner.invoke(app, ["check-endpoints"])
    assert result.exit_code == 0
    assert seen["include_llm"] is True
    assert "llm: chat" in result.output


def test_check_endpoints_exits_one_on_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mynotes.core.setup.check_endpoints",
        lambda include_llm=True: [_result("llm: chat", "fail")],
    )
    result = runner.invoke(app, ["check-endpoints"])
    assert result.exit_code == 1


def test_check_endpoints_no_llm_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, bool] = {}

    def fake_check_endpoints(include_llm: bool = True) -> list[CheckResult]:
        seen["include_llm"] = include_llm
        return [_result("embedding", "ok")]

    monkeypatch.setattr("mynotes.core.setup.check_endpoints", fake_check_endpoints)
    result = runner.invoke(app, ["check-endpoints", "--no-llm"])
    assert result.exit_code == 0
    assert seen["include_llm"] is False


def test_verbose_sets_debug_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mynotes.core.setup.run_init", lambda apply_migrations=True: [])
    result = runner.invoke(app, ["--verbose", "init"])
    assert result.exit_code == 0
    assert logging.getLogger().level == logging.DEBUG


def test_default_level_is_info(monkeypatch: pytest.MonkeyPatch) -> None:
    logging.getLogger().setLevel(logging.DEBUG)
    monkeypatch.setattr("mynotes.core.setup.run_init", lambda apply_migrations=True: [])
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert logging.getLogger().level == logging.INFO
