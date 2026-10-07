"""Filesystem and platform introspection for scan and watcher decisions.

Detects case sensitivity per source root, filesystem type (network mounts,
cloud placeholders) and picks the watcher backend (PLAN.md §7.8). Everything is
best-effort: unknown answers are ``None``/``False`` instead of exceptions.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

NETWORK_FS: frozenset[str] = frozenset(
    {
        "nfs",
        "nfs4",
        "cifs",
        "smbfs",
        "smb3",
        "fuse.sshfs",
        "afpfs",
        "webdav",
        "9p",
        "drvfs",
        "remote",
    }
)

DRIVE_REMOTE = 4
FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x00400000
FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x00040000

# Supported document formats (PLAN.md §12.1); zero-byte files with these
# extensions are cloud placeholders or empty notes and never carry content.
DOCUMENT_SUFFIXES: frozenset[str] = frozenset(
    {".md", ".txt", ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".epub"}
)

_MOUNT_ESCAPE = {"\\040": " ", "\\011": "\t", "\\012": "\n", "\\015": "\r", "\\134": "\\"}

WatchBackend = Literal["native", "polling"]


def probe_case_sensitive(root: Path) -> bool:
    """Probe whether the filesystem holding ``root`` is case-sensitive.

    Args:
        root: The directory to probe.

    Returns:
        True if creating ``.mynotes-case-probe-<uuid>`` does not make the
        upper-cased name exist. When the probe file cannot be created the
        platform default is returned instead (Linux: True, macOS/Windows:
        False) and a warning is logged.
    """
    probe_name = f".mynotes-case-probe-{uuid.uuid4().hex}"
    probe_path = root / probe_name
    try:
        with probe_path.open("x"):
            pass
    except OSError:
        logger.warning(
            "Cannot write case probe in %s; assuming case sensitivity for this platform",
            root,
        )
        return sys.platform == "linux"
    try:
        return not (root / probe_name.upper()).exists()
    finally:
        try:
            probe_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to remove case probe file %s", probe_path)


def filesystem_type(path: Path) -> str | None:
    """Return the filesystem type backing ``path``, or None if unknown.

    Args:
        path: The path to inspect.

    Returns:
        Linux: the fstype from ``/proc/mounts`` for the longest mount-point
        prefix match. macOS: the ``stat -f %T`` output (e.g. ``apfs``,
        ``smbfs``). Windows: ``"remote"`` for UNC paths or remote drives,
        ``"local"`` otherwise. Any error yields None.
    """
    try:
        if sys.platform == "linux":
            return _fstype_linux(path)
        if sys.platform == "darwin":
            return _fstype_macos(path)
        if sys.platform == "win32":
            return _fstype_windows(path)
    except Exception:
        logger.debug("Could not determine filesystem type of %s", path, exc_info=True)
        return None
    return None


def _decode_mount_component(component: str) -> str:
    for escaped, char in _MOUNT_ESCAPE.items():
        component = component.replace(escaped, char)
    return component


def _fstype_linux(path: Path) -> str | None:
    resolved = str(path.resolve())
    best: tuple[int, str] | None = None
    with open("/proc/mounts", encoding="utf-8") as fh:
        for line in fh:
            fields = line.split(" ", 3)
            if len(fields) < 3:
                continue
            mount_point = _decode_mount_component(fields[1])
            if mount_point == "/":
                prefix_len = 1
            elif resolved == mount_point or resolved.startswith(f"{mount_point}/"):
                prefix_len = len(mount_point)
            else:
                continue
            if best is None or prefix_len > best[0]:
                best = (prefix_len, fields[2])
    return best[1] if best else None


def _fstype_macos(path: Path) -> str | None:
    result = subprocess.run(
        ["stat", "-f", "%T", str(path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return result.stdout.strip() or None


def _fstype_windows(path: Path) -> str | None:
    import ctypes

    resolved = str(Path(path).resolve())
    if resolved.startswith("\\\\"):
        return "remote"
    drive_root = os.path.splitdrive(resolved)[0] + "\\"
    if ctypes.windll.kernel32.GetDriveTypeW(ctypes.c_wchar_p(drive_root)) == DRIVE_REMOTE:
        return "remote"
    return "local"


def is_network_fs(path: Path) -> bool:
    """Check whether ``path`` sits on a network filesystem.

    Args:
        path: The path to inspect.

    Returns:
        True if :func:`filesystem_type` reports a type in :data:`NETWORK_FS`.
    """
    return filesystem_type(path) in NETWORK_FS


def is_wsl_windows_mount(path: Path) -> bool:
    """Check whether ``path`` is under a Windows drive mounted in WSL.

    Native inotify events are not delivered on ``/mnt/<letter>/`` paths, so the
    watcher must poll there.

    Args:
        path: The path to inspect.

    Returns:
        True on Linux under WSL (``microsoft`` kernel release) when ``path``
        starts with ``/mnt/<letter>/``.
    """
    if sys.platform != "linux":
        return False
    if "microsoft" not in platform.release().lower():
        return False
    text = str(path)
    return text.startswith("/mnt/") and len(text) >= 7 and text[6] == "/"


def in_container() -> bool:
    """Check whether the process runs inside a container.

    Args:
        None.

    Returns:
        True if ``/.dockerenv`` exists or ``/proc/1/cgroup`` mentions
        ``docker`` or ``containerd``.
    """
    if Path("/.dockerenv").exists():
        return True
    try:
        cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "docker" in cgroup or "containerd" in cgroup


def is_cloud_placeholder(path: Path, st: os.stat_result) -> bool:
    """Check whether a file is a cloud-storage placeholder, not real content.

    Args:
        path: The file path (only the name/extension is inspected).
        st: The ``os.stat_result`` for ``path``.

    Returns:
        True for iCloud evicted stubs (``*.icloud`` names), for Windows
        files with the ``RECALL_ON_DATA_ACCESS`` or ``RECALL_ON_OPEN``
        attributes, and for zero-byte files with a supported document
        extension (an empty note has no content either).
    """
    if path.name.endswith(".icloud"):
        return True
    if sys.platform == "win32":
        attributes = getattr(st, "st_file_attributes", 0)
        if (
            attributes & FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
            or attributes & FILE_ATTRIBUTE_RECALL_ON_OPEN
        ):
            return True
    return st.st_size == 0 and path.suffix.lower() in DOCUMENT_SUFFIXES


def watch_backend_for(root: Path, configured: Literal["auto", "native", "polling"]) -> WatchBackend:
    """Resolve the watcher backend for a source root.

    Args:
        root: The source root that would be watched.
        configured: The configuration value (``auto``, ``native`` or
            ``polling``).

    Returns:
        ``native`` or ``polling`` as configured, or for ``auto``: polling when
        the root is on a network filesystem, a WSL Windows mount or the worker
        runs in a container; native otherwise.
    """
    if configured == "polling":
        return "polling"
    if configured == "native":
        return "native"
    if is_network_fs(root) or is_wsl_windows_mount(root) or in_container():
        return "polling"
    return "native"


def inotify_watch_limit() -> int | None:
    """Return the Linux per-user inotify watch limit, if known.

    Args:
        None.

    Returns:
        The value of ``/proc/sys/fs/inotify/max_user_watches`` on Linux;
        None on other platforms or when the limit cannot be read.
    """
    if sys.platform != "linux":
        return None
    try:
        return int(
            Path("/proc/sys/fs/inotify/max_user_watches").read_text(encoding="ascii").strip()
        )
    except (OSError, ValueError):
        return None
