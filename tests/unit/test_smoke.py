"""Smoke tests for the project bootstrap."""

from __future__ import annotations

import tomllib
from pathlib import Path

from typer.testing import CliRunner

import mynotes
from mynotes.cli import app

runner = CliRunner()

PYPROJECT_VERSION = tomllib.loads(
    (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")
)["project"]["version"]


def test_version_constant_matches_pyproject() -> None:
    assert mynotes.__version__ == PYPROJECT_VERSION


def test_cli_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"mynotes {PYPROJECT_VERSION}" in result.output
