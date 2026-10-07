"""Smoke tests for the project bootstrap."""

from __future__ import annotations

from typer.testing import CliRunner

import mynotes
from mynotes.cli import app

runner = CliRunner()


def test_version_constant() -> None:
    assert mynotes.__version__ == "0.1.0"


def test_cli_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "mynotes 0.1.0" in result.output
