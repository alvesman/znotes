"""``mynotes agent setup``: store the agent's settings in the user config file.

The thin agent of server mode needs four values (server URL, token, root
name, notes folder). Instead of exporting ``MYNOTES_AGENT_*`` in a shell,
which a login service never sees, the wizard writes them into the ``agent:``
section of the user config file (``platformdirs.user_config_dir("mynotes")``,
or ``MYNOTES_CONFIG`` when set), checked against the server first. The
environment variables still override the file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import platformdirs
import yaml

from mynotes.agent.client import ServerClient

AGENT_KEYS = ("server_url", "token", "root", "path")
"""The ``agent:`` keys the wizard writes, in this order."""

_HEADER = "# Written by 'mynotes agent setup'; the token is in here, keep this file private.\n"


class AgentSetupError(RuntimeError):
    """A value the wizard cannot use, or a config file it cannot rewrite."""


def config_target_path() -> Path:
    """The file the wizard writes: ``MYNOTES_CONFIG`` if set, else the user config file.

    ``./mynotes.yaml`` is never the target: that is the project (server) file.
    """
    configured = os.environ.get("MYNOTES_CONFIG")
    if configured:
        return Path(configured).expanduser()
    return Path(platformdirs.user_config_dir("mynotes")) / "mynotes.yaml"


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """The file's top-level mapping (``{}`` when the file is missing or empty).

    Raises:
        AgentSetupError: When the file is not a YAML mapping.
    """
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise AgentSetupError(f"{path} is not a YAML mapping; fix or remove it first")
    return data


def current_agent_block(path: Path) -> dict[str, Any]:
    """The existing ``agent:`` mapping of ``path`` (``{}`` when absent or unusable)."""
    try:
        block = load_yaml_mapping(path).get("agent")
    except AgentSetupError:
        return {}
    return dict(block) if isinstance(block, dict) else {}


def write_agent_settings(path: Path, values: dict[str, Any]) -> None:
    """Merge ``values`` into the file's ``agent:`` section and write it with mode 600.

    Other sections and other ``agent`` keys are kept; comments are not (the
    file is re-serialized). The write goes through a temporary file in the
    same folder, so a crash leaves the old file intact.

    Raises:
        AgentSetupError: When the file is not a YAML mapping.
    """
    data = load_yaml_mapping(path)
    block = data.get("agent")
    merged = dict(block) if isinstance(block, dict) else {}
    merged.update(values)
    data["agent"] = merged
    path.parent.mkdir(parents=True, exist_ok=True)
    body = _HEADER + yaml.safe_dump(
        data, sort_keys=False, allow_unicode=True, default_flow_style=False
    )
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    if sys.platform != "win32":
        os.chmod(path, 0o600)


def verify_agent(server_url: str, token: str, root: str, *, verify_tls: bool = True) -> dict:
    """Ask the server for the root's status with these settings; errors propagate.

    Returns:
        The status payload (``files``, ``server_version`` on newer servers, ...).
    """
    client = ServerClient(server_url, token, root, verify_tls=verify_tls)
    try:
        return client.status()
    finally:
        client.close()


def normalise_server_url(value: str) -> str:
    """A server URL with its scheme checked and no trailing slash.

    Raises:
        AgentSetupError: When the value is not an ``http(s)://`` URL.
    """
    cleaned = value.strip().rstrip("/")
    if not cleaned.startswith(("http://", "https://")):
        raise AgentSetupError(f"the server URL must start with http:// or https://: {value!r}")
    return cleaned


def normalise_path(value: str) -> Path:
    """The notes folder as an absolute path.

    Raises:
        AgentSetupError: When it is not an existing folder.
    """
    folder = Path(value.strip()).expanduser()
    if not folder.is_dir():
        raise AgentSetupError(f"not a folder: {folder}")
    return folder.resolve()
