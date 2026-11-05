"""``mynotes agent setup``: the config writer and the wizard against a fake server."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml
from typer.testing import CliRunner

from mynotes import __version__
from mynotes.agent import setup as setup_mod
from mynotes.agent.client import AgentAuthError
from mynotes.agent.setup import (
    AgentSetupError,
    config_target_path,
    load_yaml_mapping,
    normalise_path,
    normalise_server_url,
    write_agent_settings,
)
from mynotes.cli import app
from mynotes.config import reset_settings_cache

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty cwd, no MYNOTES_* variables, a user config dir under tmp_path."""
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("MYNOTES_"):
            monkeypatch.delenv(key)
    config_dir = tmp_path / "user-config"
    for target in ("mynotes.config.platformdirs", "mynotes.agent.setup.platformdirs"):
        monkeypatch.setattr(f"{target}.user_config_dir", lambda *a, **k: str(config_dir))
    reset_settings_cache()
    yield config_dir
    reset_settings_cache()


class FakeClient:
    """Stands in for ServerClient: records the settings it was built with."""

    built: ClassVar[list[dict[str, Any]]] = []
    answer: ClassVar[Any] = {
        "files": {"active": 2, "placeholder": 1},
        "server_version": __version__,
    }

    def __init__(self, server_url: str, token: str, root: str, **kwargs: Any) -> None:
        FakeClient.built.append({"server_url": server_url, "token": token, "root": root})

    def status(self) -> dict[str, Any]:
        if isinstance(FakeClient.answer, Exception):
            raise FakeClient.answer
        return dict(FakeClient.answer)

    def close(self) -> None:
        pass


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> type[FakeClient]:
    FakeClient.built = []
    FakeClient.answer = {"files": {"active": 2, "placeholder": 1}, "server_version": __version__}
    monkeypatch.setattr(setup_mod, "ServerClient", FakeClient)
    return FakeClient


# --- the writer ---------------------------------------------------------------


def test_write_merges_into_existing_file_and_keeps_other_keys(tmp_path: Path) -> None:
    target = tmp_path / "cfg" / "mynotes.yaml"
    target.parent.mkdir()
    target.write_text(
        "workspace: default\nagent:\n  watch: false\n  token: old\n", encoding="utf-8"
    )
    values = {"server_url": "https://s", "token": "mn_new", "root": "r", "path": "/n"}
    write_agent_settings(target, values)
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert data["workspace"] == "default"
    assert data["agent"] == {"watch": False, **values}
    assert target.read_text(encoding="utf-8").startswith("# Written by 'mynotes agent setup'")
    if sys.platform != "win32":
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not target.with_name("mynotes.yaml.tmp").exists()


def test_write_creates_the_directory(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "er" / "mynotes.yaml"
    write_agent_settings(target, {"root": "r"})
    assert yaml.safe_load(target.read_text(encoding="utf-8")) == {"agent": {"root": "r"}}


def test_load_rejects_a_non_mapping(tmp_path: Path) -> None:
    target = tmp_path / "mynotes.yaml"
    target.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(AgentSetupError, match="not a YAML mapping"):
        load_yaml_mapping(target)
    with pytest.raises(AgentSetupError):
        write_agent_settings(target, {"root": "r"})
    assert load_yaml_mapping(tmp_path / "missing.yaml") == {}


def test_target_path_prefers_mynotes_config(
    _isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert config_target_path() == _isolated / "mynotes.yaml"
    monkeypatch.setenv("MYNOTES_CONFIG", str(tmp_path / "elsewhere.yaml"))
    assert config_target_path() == tmp_path / "elsewhere.yaml"


def test_normalisers(tmp_path: Path) -> None:
    assert normalise_server_url(" https://notes.example.org/ ") == "https://notes.example.org"
    with pytest.raises(AgentSetupError, match="http"):
        normalise_server_url("notes.example.org")
    assert normalise_path(str(tmp_path)) == tmp_path.resolve()
    with pytest.raises(AgentSetupError, match="not a folder"):
        normalise_path(str(tmp_path / "nope"))


# --- the wizard ---------------------------------------------------------------


def test_wizard_prompts_verifies_and_writes(
    _isolated: Path, fake_client: type[FakeClient], tmp_path: Path
) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    answers = f"https://notes.example.org/\nmn_secret\nalice-notes\n{notes}\n"
    result = runner.invoke(app, ["agent", "setup"], input=answers)
    assert result.exit_code == 0, result.output
    assert fake_client.built == [
        {"server_url": "https://notes.example.org", "token": "mn_secret", "root": "alice-notes"}
    ]
    assert "ok" in result.output and "3 file(s) known to the server" in result.output
    assert f"server {__version__}, this agent {__version__}" in result.output
    assert "Next:" in result.output and "service install --component agent" in result.output
    data = yaml.safe_load((_isolated / "mynotes.yaml").read_text(encoding="utf-8"))
    assert data == {
        "agent": {
            "server_url": "https://notes.example.org",
            "token": "mn_secret",
            "root": "alice-notes",
            "path": str(notes.resolve()),
        }
    }
    assert "mn_secret" not in result.output  # the hidden prompt never echoes it


def test_wizard_keeps_the_current_token_on_enter(
    _isolated: Path, fake_client: type[FakeClient], tmp_path: Path
) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    write_agent_settings(
        _isolated / "mynotes.yaml",
        {"server_url": "https://old", "token": "mn_old", "root": "old-root", "path": str(notes)},
    )
    # every prompt answered with Enter: the stored values stay, the token included
    result = runner.invoke(app, ["agent", "setup"], input="\n\n\n\n")
    assert result.exit_code == 0, result.output
    assert fake_client.built[0]["token"] == "mn_old"
    data = yaml.safe_load((_isolated / "mynotes.yaml").read_text(encoding="utf-8"))
    assert data["agent"]["token"] == "mn_old" and data["agent"]["root"] == "old-root"


def test_wizard_writes_nothing_when_the_server_refuses(
    _isolated: Path, fake_client: type[FakeClient], tmp_path: Path
) -> None:
    fake_client.answer = AgentAuthError("401 from the server: invalid token")
    args = ["agent", "setup", "--server-url", "https://s", "--token", "mn_x", "--root", "r"]
    result = runner.invoke(app, [*args, "--path", str(tmp_path)])
    assert result.exit_code == 1
    assert "invalid token" in result.output and "Nothing was written" in result.output
    assert not (_isolated / "mynotes.yaml").exists()


def test_wizard_reports_an_unreachable_server(
    _isolated: Path, fake_client: type[FakeClient], tmp_path: Path
) -> None:
    import httpx

    fake_client.answer = httpx.ConnectError("connection refused")
    args = ["agent", "setup", "--server-url", "https://s", "--token", "mn_x", "--root", "r"]
    result = runner.invoke(app, [*args, "--path", str(tmp_path)])
    assert result.exit_code == 1
    assert "Could not reach https://s" in result.output
    assert not (_isolated / "mynotes.yaml").exists()


def test_no_verify_and_options_need_no_server_and_no_prompts(
    _isolated: Path, fake_client: type[FakeClient], tmp_path: Path
) -> None:
    fake_client.answer = AssertionError("the server must not be called")
    args = ["agent", "setup", "--no-verify", "--server-url", "https://s", "--token", "mn_x"]
    result = runner.invoke(app, [*args, "--root", "r", "--path", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert fake_client.built == []
    assert (
        yaml.safe_load((_isolated / "mynotes.yaml").read_text(encoding="utf-8"))["agent"]["root"]
        == "r"
    )


def test_env_variables_fill_the_answers_and_are_flagged_as_overrides(
    _isolated: Path, fake_client: type[FakeClient], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MYNOTES_AGENT_SERVER_URL", "https://env")
    monkeypatch.setenv("MYNOTES_AGENT_TOKEN", "mn_env")
    monkeypatch.setenv("MYNOTES_AGENT_ROOT", "env-root")
    monkeypatch.setenv("MYNOTES_AGENT_PATH", str(tmp_path))
    result = runner.invoke(app, ["agent", "setup", "--no-verify"])
    assert result.exit_code == 0, result.output
    data = yaml.safe_load((_isolated / "mynotes.yaml").read_text(encoding="utf-8"))
    assert data["agent"]["server_url"] == "https://env" and data["agent"]["token"] == "mn_env"
    assert "MYNOTES_AGENT_TOKEN is set in your environment and overrides" in result.output


def test_missing_folder_is_an_error_without_a_terminal(
    _isolated: Path, fake_client: type[FakeClient], tmp_path: Path
) -> None:
    args = ["agent", "setup", "--no-verify", "--server-url", "https://s", "--token", "mn_x"]
    result = runner.invoke(app, [*args, "--root", "r", "--path", str(tmp_path / "nope")])
    assert result.exit_code == 2 and "not a folder" in result.output
    assert not (_isolated / "mynotes.yaml").exists()


def test_wizard_warns_when_another_file_wins_from_this_folder(
    _isolated: Path, fake_client: type[FakeClient], tmp_path: Path
) -> None:
    (tmp_path / "mynotes.yaml").write_text("workspace: default\n", encoding="utf-8")  # the cwd
    args = ["agent", "setup", "--no-verify", "--server-url", "https://s", "--token", "mn_x"]
    result = runner.invoke(app, [*args, "--root", "r", "--path", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Note: from this folder mynotes reads" in result.output
    assert (_isolated / "mynotes.yaml").exists()  # written to the user config file all the same


def test_show_masks_the_token(_isolated: Path, tmp_path: Path) -> None:
    write_agent_settings(
        _isolated / "mynotes.yaml",
        {"server_url": "https://s", "token": "mn_verysecretvalue", "root": "r", "path": "/n"},
    )
    reset_settings_cache()
    result = runner.invoke(app, ["agent", "setup", "--show"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "Settings file:" in result.output and "https://s" in result.output
    assert "mn_verysecretvalue" not in result.output and "chars)" in result.output


def test_show_without_settings_points_to_setup(_isolated: Path) -> None:
    result = runner.invoke(app, ["agent", "setup", "--show"])
    assert result.exit_code == 1 and "run 'mynotes agent setup'" in result.output
