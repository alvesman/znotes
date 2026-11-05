"""Common interface for the worker autostart installers (PLAN.md §7.8).

``mynotes service install`` renders and installs the native autostart
definition for the current OS: a systemd user unit on Linux, a LaunchAgent
plist on macOS and a Task Scheduler task on Windows. The CLI (``mynotes
service`` in :mod:`mynotes.cli`) is the only consumer; nothing here runs at
import time.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import platformdirs

SERVICE_NAME = "mynotes-worker"

COMPONENTS: dict[str, dict[str, Any]] = {
    "worker": {"args": ["worker"], "description": "mynotes worker (scan, index, graph)"},
    "web": {"args": ["serve"], "description": "mynotes web interface and API (serve)"},
    "agent": {
        "args": ["agent", "run"],
        "description": "mynotes agent (feeds a remote root of a mynotes server)",
    },
}
"""The installable components: the CLI subcommand each runs and its unit description."""

DEFAULT_COMPONENTS: tuple[str, ...] = ("worker", "web")
"""What ``--component all`` installs: never the agent, which needs ``mynotes agent setup``."""

_INSTALLED_TREE_MARKERS = frozenset({"site-packages", "dist-packages"})


class ServiceError(RuntimeError):
    """Raised when the service definition cannot be rendered or installed."""


@dataclass
class ServiceSpec:
    """Everything a backend needs to render an autostart definition.

    Attributes:
        name: The service name, ``"mynotes-worker"``.
        exec_args: Absolute executable plus arguments; on Linux/macOS the uv
            path plus ``["run", "--directory", <project dir>, "mynotes",
            "worker"]``, on Windows the project venv's ``pythonw.exe`` plus
            ``["-m", "mynotes", "worker"]``; for an installed package the
            running interpreter plus ``["-m", "mynotes", ...]``.
        working_dir: The project directory (contains ``pyproject.toml``), or
            the user config directory for an installed package.
        log_dir: Per-OS user log directory (``platformdirs.user_log_dir``).
        env: Extra environment variables for the service; ``MYNOTES_CONFIG``
            when set, plus ``MYNOTES_LOG_FILE`` on Windows. Never secrets:
            those stay in the ``.env`` or settings file of ``working_dir``.
    """

    name: str
    exec_args: list[str]
    working_dir: Path
    log_dir: Path
    env: dict[str, str] = field(default_factory=dict)
    component: str = "worker"
    description: str = "mynotes worker (scan, index, graph)"

    @property
    def log_name(self) -> str:
        """The log file name under ``log_dir`` (``worker.log``, ``web.log``)."""
        return f"{self.component}.log"


class ServiceBackend(Protocol):
    """Render and manage the native autostart definition for one OS."""

    def render(self, spec: ServiceSpec) -> dict[str, str]:
        """Return the files to write as ``path -> content``.

        On Windows, where nothing is written, the single key describes the
        command (the PowerShell install script) instead of a file path.
        """
        ...

    def install(self, spec: ServiceSpec) -> None:
        """Write the definition and enable the service.

        Raises:
            ServiceError: When a definition file or system command fails.
        """
        ...

    def uninstall(self, spec: ServiceSpec) -> None:
        """Disable the service and remove the definition."""
        ...

    def status(self, spec: ServiceSpec) -> str:
        """Return a short human-readable state of the service."""
        ...

    def logs_hint(self, spec: ServiceSpec) -> str:
        """Return the command that follows the service logs."""
        ...

    def hints(self, spec: ServiceSpec) -> list[str]:
        """Return post-install guidance lines (permissions, linger, ...)."""
        ...


def backend_for_platform() -> ServiceBackend:
    """Return the autostart backend for the current OS.

    Returns:
        :class:`~mynotes.platform.service.systemd.SystemdBackend` on Linux,
        :class:`~mynotes.platform.service.launchd.LaunchdBackend` on macOS and
        :class:`~mynotes.platform.service.windows_task.WindowsTaskBackend` on
        Windows.

    Raises:
        ServiceError: On any other platform.
    """
    if sys.platform == "win32":
        from mynotes.platform.service.windows_task import WindowsTaskBackend

        return WindowsTaskBackend()
    if sys.platform == "darwin":
        from mynotes.platform.service.launchd import LaunchdBackend

        return LaunchdBackend()
    if sys.platform.startswith("linux"):
        from mynotes.platform.service.systemd import SystemdBackend

        return SystemdBackend()
    raise ServiceError(f"unsupported platform for service install: {sys.platform}")


def _project_dir() -> Path | None:
    """The uv project this package runs from, or None for an installed package.

    Returns:
        The first ancestor of this file with a ``pyproject.toml`` (the normal
        case: the repo checked out and used through ``uv run --directory``);
        None when the package sits in a ``site-packages`` tree (``uv tool
        install``, pip) or no ``pyproject.toml`` is found at all.
    """
    for candidate in Path(__file__).resolve().parents:
        if candidate.name in _INSTALLED_TREE_MARKERS:
            return None
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def default_spec(component: str = "worker") -> ServiceSpec:
    """Build the :class:`ServiceSpec` for this machine.

    From a uv checkout: locates ``uv`` on PATH (on Windows the project venv's
    ``pythonw.exe`` instead) and runs ``mynotes`` through the project. From an
    installed package (``uv tool install``): runs this interpreter with
    ``-m mynotes`` from the user config directory, where the settings file
    (and a ``.env``) live. Both carry ``MYNOTES_CONFIG`` from the environment
    when set.

    Args:
        component: ``"worker"`` (``mynotes worker``), ``"web"`` (``mynotes
            serve``, localhost:8765) or ``"agent"`` (``mynotes agent run``);
            see :data:`COMPONENTS`.

    Returns:
        The ready-to-install spec named ``"mynotes-<component>"``.

    Raises:
        ServiceError: When ``uv`` is missing (on Windows: ``pythonw.exe``) in a
            checkout, or the component is unknown.
    """
    if component not in COMPONENTS:
        raise ServiceError(f"unknown component {component!r}; one of {sorted(COMPONENTS)}")
    command = list(COMPONENTS[component]["args"])
    log_dir = Path(platformdirs.user_log_dir("mynotes"))
    env: dict[str, str] = {}
    config_env = os.environ.get("MYNOTES_CONFIG")
    if config_env:
        env["MYNOTES_CONFIG"] = config_env
    project = _project_dir()
    if project is None:
        working_dir = Path(platformdirs.user_config_dir("mynotes"))
        working_dir.mkdir(parents=True, exist_ok=True)  # systemd refuses a missing directory
        python = Path(sys.executable)
        if sys.platform == "win32":
            windowless = python.with_name("pythonw.exe")
            if windowless.is_file():
                python = windowless
            env.setdefault("MYNOTES_LOG_FILE", str(log_dir / f"{component}.log"))
        exec_args = [str(python), "-m", "mynotes", *command]
        return ServiceSpec(
            name=f"mynotes-{component}",
            exec_args=exec_args,
            working_dir=working_dir,
            log_dir=log_dir,
            env=env,
            component=component,
            description=str(COMPONENTS[component]["description"]),
        )
    working_dir = project
    if sys.platform == "win32":
        pythonw = working_dir / ".venv" / "Scripts" / "pythonw.exe"
        if not pythonw.is_file():
            raise ServiceError(
                f"pythonw.exe not found at {pythonw}; run 'uv sync' in {working_dir} first"
            )
        exec_args = [str(pythonw), "-m", "mynotes", *command]
        env.setdefault("MYNOTES_LOG_FILE", str(log_dir / f"{component}.log"))
    else:
        uv = shutil.which("uv")
        if uv is None:
            raise ServiceError(
                "uv was not found on PATH; install it first (https://docs.astral.sh/uv/)"
            )
        exec_args = [
            str(Path(uv).resolve()),
            "run",
            "--directory",
            str(working_dir),
            "mynotes",
            *command,
        ]
    return ServiceSpec(
        name=f"mynotes-{component}",
        exec_args=exec_args,
        working_dir=working_dir,
        log_dir=log_dir,
        env=env,
        component=component,
        description=str(COMPONENTS[component]["description"]),
    )
