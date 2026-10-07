"""systemd user unit backend (Linux, PLAN.md §7.8).

The unit lives in ``~/.config/systemd/user/mynotes-worker.service`` and is
enabled with ``systemctl --user enable --now``. Logs go to journald; the only
extra step a user may want is ``loginctl enable-linger`` so the worker also
runs without an open session.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from mynotes.platform.service import ServiceError, ServiceSpec

logger = logging.getLogger(__name__)

_UNIT_DIR = Path(".config") / "systemd" / "user"


def _quote(value: str) -> str:
    """Quote ``value`` for a systemd unit value when it contains whitespace.

    Args:
        value: A path or argument to place on an ``ExecStart``/directory line.

    Returns:
        ``value`` unchanged when it has no whitespace, else double-quoted with
        backslashes and quotes escaped the way systemd expects.
    """
    if not any(ch in value for ch in " \t"):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_unit(spec: ServiceSpec) -> str:
    """Render the systemd user unit for ``spec``.

    Args:
        spec: The service definition.

    Returns:
        The complete unit file text, ``PYTHONUNBUFFERED=1`` plus the spec's
        extra environment variables included as ``Environment=`` lines.
    """
    env_lines = ["Environment=PYTHONUNBUFFERED=1"]
    env_lines += [f"Environment={key}={_quote(value)}" for key, value in sorted(spec.env.items())]
    exec_start = " ".join(_quote(arg) for arg in spec.exec_args)
    return (
        "[Unit]\n"
        f"Description={spec.description}\n"
        "After=network-online.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"WorkingDirectory={_quote(str(spec.working_dir))}\n"
        f"ExecStart={exec_start}\n"
        "Restart=on-failure\n"
        "RestartSec=10\n" + "\n".join(env_lines) + "\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


class SystemdBackend:
    """Install the worker as a systemd *user* unit."""

    def unit_path(self, spec: ServiceSpec) -> Path:
        """Return the unit file path under ``~/.config/systemd/user``."""
        return Path.home() / _UNIT_DIR / f"{spec.name}.service"

    def render(self, spec: ServiceSpec) -> dict[str, str]:
        """Return ``{unit file path: unit content}``."""
        return {str(self.unit_path(spec)): render_unit(spec)}

    def install(self, spec: ServiceSpec) -> None:
        """Write the unit, reload the user manager and enable the service.

        Args:
            spec: The service definition.

        Raises:
            ServiceError: When a ``systemctl --user`` command fails.
        """
        path = self.unit_path(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_unit(spec), encoding="utf-8")
        logger.info("wrote %s", path)
        self._check(self._systemctl("daemon-reload"), "daemon-reload")
        self._check(self._systemctl("enable", "--now", spec.name), f"enable --now {spec.name}")

    def uninstall(self, spec: ServiceSpec) -> None:
        """Stop and disable the service, delete the unit and reload.

        Args:
            spec: The service definition.

        Raises:
            ServiceError: When a ``systemctl --user`` command fails.
        """
        self._check(self._systemctl("disable", "--now", spec.name), f"disable --now {spec.name}")
        self.unit_path(spec).unlink(missing_ok=True)
        self._check(self._systemctl("daemon-reload"), "daemon-reload")

    def status(self, spec: ServiceSpec) -> str:
        """Return the ``is-active`` state (active, inactive, failed, ...)."""
        result = self._systemctl("is-active", spec.name)
        return (result.stdout or result.stderr).strip() or "unknown"

    def logs_hint(self, spec: ServiceSpec) -> str:
        """Return the ``journalctl`` command that follows the service logs."""
        return f"journalctl --user -u {spec.name} -f"

    def hints(self, spec: ServiceSpec) -> list[str]:
        """Return the linger hint for running without an open session."""
        return [
            "To keep the worker running without an open session, run:",
            "  loginctl enable-linger $USER",
        ]

    @staticmethod
    def _systemctl(*args: str) -> subprocess.CompletedProcess[str]:
        """Run ``systemctl --user`` with the given arguments, never raising.

        Args:
            args: The systemctl subcommand and its arguments.

        Returns:
            The completed process with captured stdout/stderr.

        Raises:
            ServiceError: When the ``systemctl`` binary itself is missing.
        """
        try:
            return subprocess.run(
                ["systemctl", "--user", *args],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ServiceError("systemctl was not found; is systemd installed?") from exc

    @staticmethod
    def _check(result: subprocess.CompletedProcess[str], action: str) -> None:
        """Raise :class:`ServiceError` when ``result`` reports a failure."""
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise ServiceError(f"systemctl --user {action} failed: {detail}")
