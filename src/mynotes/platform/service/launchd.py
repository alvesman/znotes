"""launchd LaunchAgent backend (macOS, PLAN.md §7.8).

The agent plist lives in ``~/Library/LaunchAgents/com.mynotes.worker.plist``
(``RunAtLoad`` + ``KeepAlive``), logs go to ``~/Library/Logs/mynotes/
worker.log``. Watching folders macOS protects (Documents, Desktop, Downloads,
iCloud Drive) from a LaunchAgent needs Full Disk Access (or Files and
Folders) for the uv binary and its Python; :meth:`LaunchdBackend.hints`
detects protected roots and prints the steps.
"""

from __future__ import annotations

import logging
import os
import plistlib
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

from mynotes.platform.paths import is_within
from mynotes.platform.service import ServiceError, ServiceSpec

logger = logging.getLogger(__name__)

TCC_PROTECTED_DIRS: tuple[str, ...] = (
    "Documents",
    "Desktop",
    "Downloads",
    "Library/Mobile Documents",
)
"""Home-relative folders macOS gates behind TCC (Full Disk Access)."""

_LABEL_PREFIX = "com.mynotes."

_STATE_RE = re.compile(r"^\s*state = (\S+)", re.MULTILINE)


def tcc_protected_roots(roots: Iterable[Path], home: Path | None = None) -> list[Path]:
    """Return the roots that sit under a TCC-protected folder.

    Args:
        roots: Candidate source-root directories.
        home: The home directory; defaults to :meth:`pathlib.Path.home`.

    Returns:
        The roots equal to or inside ``~/Documents``, ``~/Desktop``,
        ``~/Downloads`` or ``~/Library/Mobile Documents``, in input order.
    """
    base = (home or Path.home()).resolve()
    protected = [base / Path(parts) for parts in TCC_PROTECTED_DIRS]
    return [root for root in roots if any(is_within(folder, root) for folder in protected)]


def python_for_uv(uv: Path) -> str | None:
    """Find the Python interpreter ``uv`` uses (``uv python find 3.12``).

    Args:
        uv: Path to the uv binary.

    Returns:
        The interpreter path as a string, or None when uv cannot answer.
    """
    try:
        result = subprocess.run(
            [str(uv), "python", "find", "3.12"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def tcc_steps(protected: Iterable[Path], uv: Path, python: str | None) -> list[str]:
    """Build the Full Disk Access guidance for protected roots.

    Args:
        protected: The roots that need the permission.
        uv: The uv binary the service runs.
        python: The interpreter reported by ``uv python find 3.12``, if found.

    Returns:
        Human-readable steps mentioning the binaries to whitelist.
    """
    roots_text = ", ".join(str(path) for path in protected)
    lines = [
        f"These roots are in folders macOS protects (TCC): {roots_text}",
        "The worker cannot read them from a LaunchAgent until you grant access:",
        "  1. System Settings, Privacy & Security, Full Disk Access (or Files and Folders).",
        f"  2. Add the uv binary: {uv}",
    ]
    if python:
        lines.append(f"  3. Add the Python interpreter uv uses: {python}")
    else:
        lines.append("  3. Add the Python interpreter uv uses (find it with: uv python find 3.12)")
    return lines


class LaunchdBackend:
    """Install the worker as a LaunchAgent."""

    def label(self, spec: ServiceSpec) -> str:
        """Return the launchd label derived from the service name."""
        return _LABEL_PREFIX + spec.name.removeprefix("mynotes-")

    def plist_path(self, spec: ServiceSpec) -> Path:
        """Return the plist path under ``~/Library/LaunchAgents``."""
        return Path.home() / "Library" / "LaunchAgents" / f"{self.label(spec)}.plist"

    def log_file(self, spec: ServiceSpec) -> Path:
        """Return the combined stdout/stderr log file of the agent."""
        return spec.log_dir / spec.log_name

    def render(self, spec: ServiceSpec) -> dict[str, str]:
        """Return ``{plist path: plist content}``, generated with plistlib."""
        payload = {
            "Label": self.label(spec),
            "ProgramArguments": list(spec.exec_args),
            "WorkingDirectory": str(spec.working_dir),
            "RunAtLoad": True,
            "KeepAlive": True,
            "StandardOutPath": str(self.log_file(spec)),
            "StandardErrorPath": str(self.log_file(spec)),
            "EnvironmentVariables": dict(spec.env),
        }
        content = plistlib.dumps(payload, fmt=plistlib.FMT_XML).decode("utf-8")
        return {str(self.plist_path(spec)): content}

    def install(self, spec: ServiceSpec) -> None:
        """Write the plist and bootstrap the agent (falling back to load -w).

        Args:
            spec: The service definition.

        Raises:
            ServiceError: When both ``launchctl bootstrap`` and the fallback
                ``launchctl load -w`` fail.
        """
        path = self.plist_path(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(spec)[str(path)], encoding="utf-8")
        logger.info("wrote %s", path)
        gui = f"gui/{os.getuid()}"
        bootstrap = self._launchctl("bootstrap", gui, str(path))
        if bootstrap.returncode == 0:
            return
        fallback = self._launchctl("load", "-w", str(path))
        if fallback.returncode != 0:
            detail = (bootstrap.stderr or bootstrap.stdout).strip()
            raise ServiceError(f"launchctl bootstrap failed: {detail}")

    def uninstall(self, spec: ServiceSpec) -> None:
        """Bootout the agent and delete the plist.

        A failed bootout (agent not loaded) is logged and ignored so the plist
        is still removed.
        """
        result = self._launchctl("bootout", f"gui/{os.getuid()}/{self.label(spec)}")
        if result.returncode != 0:
            logger.warning(
                "launchctl bootout failed (agent not loaded?): %s",
                (result.stderr or result.stdout).strip(),
            )
        self.plist_path(spec).unlink(missing_ok=True)

    def status(self, spec: ServiceSpec) -> str:
        """Return the agent state from ``launchctl print`` (e.g. running)."""
        result = self._launchctl("print", f"gui/{os.getuid()}/{self.label(spec)}")
        if result.returncode != 0:
            return "not installed"
        match = _STATE_RE.search(result.stdout)
        return match.group(1) if match else result.stdout.strip()[:200] or "unknown"

    def logs_hint(self, spec: ServiceSpec) -> str:
        """Return the ``tail -f`` command for the agent log file."""
        return f"tail -f {self.log_file(spec)}"

    def hints(self, spec: ServiceSpec) -> list[str]:
        """Return the TCC steps when a configured root needs Full Disk Access."""
        protected = tcc_protected_roots(self._configured_roots())
        if not protected:
            return []
        uv = Path(spec.exec_args[0])
        return tcc_steps(protected, uv, python_for_uv(uv))

    @staticmethod
    def _configured_roots() -> list[Path]:
        """Read the configured source roots; empty when settings are broken."""
        try:
            from mynotes.config import get_settings

            return [root.path for root in get_settings().roots]
        except Exception:
            logger.warning("could not read configured roots for the TCC check", exc_info=True)
            return []

    @staticmethod
    def _launchctl(*args: str) -> subprocess.CompletedProcess[str]:
        """Run ``launchctl`` with the given arguments, never raising.

        Raises:
            ServiceError: When the ``launchctl`` binary itself is missing.
        """
        try:
            return subprocess.run(["launchctl", *args], capture_output=True, text=True, check=False)
        except FileNotFoundError as exc:
            raise ServiceError("launchctl was not found; is this macOS?") from exc
