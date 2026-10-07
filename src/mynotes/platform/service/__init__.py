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
}
"""The installable components: the CLI subcommand each runs and its unit description."""


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
            ``["-m", "mynotes", "worker"]``.
        working_dir: The project directory (contains ``pyproject.toml``).
        log_dir: Per-OS user log directory (``platformdirs.user_log_dir``).
        env: Extra environment variables for the service; ``MYNOTES_CONFIG``
            when set, plus ``MYNOTES_LOG_FILE`` on Windows. Never secrets:
            those stay in the project's ``.env``.
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


def _project_dir() -> Path:
    """Best-effort project directory: the nearest ancestor holding pyproject.toml.

    Returns:
        The first ancestor of this file with a ``pyproject.toml`` (the normal
        case: the repo checked out and used through ``uv run --directory``);
        the current working directory when the package is installed into a
        site-packages tree without one.
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


def default_spec(component: str = "worker") -> ServiceSpec:
    """Build the :class:`ServiceSpec` for this machine.

    Locates ``uv`` on PATH (on Windows the project venv's ``pythonw.exe``
    instead), the project directory and the per-OS log directory, and carries
    ``MYNOTES_CONFIG`` from the environment when set.

    Args:
        component: ``"worker"`` (``mynotes worker``) or ``"web"`` (``mynotes
            serve``, localhost:8765); see :data:`COMPONENTS`.

    Returns:
        The ready-to-install spec named ``"mynotes-<component>"``.

    Raises:
        ServiceError: When ``uv`` is missing (on Windows: ``pythonw.exe``), or
            the component is unknown.
    """
    if component not in COMPONENTS:
        raise ServiceError(f"unknown component {component!r}; one of {sorted(COMPONENTS)}")
    command = list(COMPONENTS[component]["args"])
    working_dir = _project_dir()
    log_dir = Path(platformdirs.user_log_dir("mynotes"))
    env: dict[str, str] = {}
    config_env = os.environ.get("MYNOTES_CONFIG")
    if config_env:
        env["MYNOTES_CONFIG"] = config_env
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
