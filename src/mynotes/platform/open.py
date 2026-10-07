"""Open files in the operating system's default application (PLAN.md §10.4).

``open`` on macOS, ``xdg-open`` on Linux and ``os.startfile`` on Windows. The
launcher runs detached (the caller never waits), so the facade's
``open_original`` (task 45) returns immediately after handing the file to the OS.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from mynotes.platform.paths import long_path

logger = logging.getLogger(__name__)


def open_command(path: Path) -> list[str] | None:
    """Build the launcher command for ``path`` on the current OS.

    Args:
        path: The file to open.

    Returns:
        The command list (``open`` on macOS, ``xdg-open`` elsewhere on POSIX);
        ``None`` on Windows, where :func:`os.startfile` takes the path directly.
    """
    if sys.platform == "darwin":
        return ["open", str(path)]
    if sys.platform == "win32":
        return None
    return ["xdg-open", str(path)]


def open_in_default_app(path: Path) -> None:
    """Open ``path`` in the OS default application without waiting.

    Args:
        path: The file to open (any format the OS has an association for).

    Raises:
        RuntimeError: When no launcher is available (``xdg-open`` missing on
            Linux, ``os.startfile`` absent, or the OS refused to launch).
    """
    command = open_command(path)
    if command is None:
        startfile = getattr(os, "startfile", None)
        if startfile is None:
            raise RuntimeError(f"cannot open {path.name!r}: this platform has no file launcher")
        try:
            startfile(str(long_path(path)))
        except OSError as exc:
            raise RuntimeError(f"cannot open {path.name!r}: {exc}") from exc
        logger.info("opened %s via os.startfile", path.name)
        return
    try:
        subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"cannot open {path.name!r}: launcher {command[0]!r} was not found"
            + ("; install the xdg-utils package" if command[0] == "xdg-open" else "")
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"cannot open {path.name!r}: {exc}") from exc
    logger.info("opened %s via %s", path.name, command[0])
