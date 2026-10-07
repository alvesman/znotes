"""Canonical relative paths and long-path handling for every OS.

All stored paths are relative to a source root with POSIX ``/`` separators and
Unicode NFC normalization (PLAN.md §7.8), so document identity does not change
between macOS (NFD filenames on disk) and the other platforms. :func:`is_within`
protects path consumers (file serving, task 45) against ``..`` and symlink
escapes.
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

from mynotes.text.normalize import normalize_path_key

WINDOWS_LONG_PATH_PREFIX = "\\\\?\\"
WINDOWS_MAX_PATH = 260


def to_rel_path(root: Path, abs_path: Path) -> str:
    """Return the canonical relative path of ``abs_path`` inside ``root``.

    Args:
        root: The source root directory.
        abs_path: An absolute path inside ``root``.

    Returns:
        The path relative to ``root`` with POSIX ``/`` separators, Unicode NFC
        normalization and no leading ``./``.

    Raises:
        ValueError: If ``abs_path`` is outside ``root`` (both paths are
            resolved first, so symlinked components cannot escape).
    """
    resolved_root = root.resolve()
    rel = abs_path.resolve().relative_to(resolved_root)
    return unicodedata.normalize("NFC", rel.as_posix())


def from_rel_path(root: Path, rel_path: str) -> Path:
    """Turn a stored relative path back into a real filesystem path.

    Args:
        root: The source root directory.
        rel_path: A stored relative path with ``/`` separators.

    Returns:
        The absolute path ``root`` joined with the ``/``-separated segments of
        ``rel_path``.
    """
    return root / Path(*rel_path.split("/"))


def path_ancestors(rel_path: str) -> list[str]:
    """List the containing folders of a relative path, nearest first.

    Args:
        rel_path: A stored relative path with ``/`` separators.

    Returns:
        The folder prefixes excluding the file itself, e.g. ``"a/b/c.md"`` →
        ``["a", "a/b"]`` and ``"c.md"`` → ``[]``.
    """
    parts = rel_path.split("/")
    return ["/".join(parts[:i]) for i in range(1, len(parts))]


def path_ancestors_folded(rel_path: str) -> list[str]:
    """List the folded keys of the containing folders of a relative path.

    Args:
        rel_path: A stored relative path with ``/`` separators.

    Returns:
        :func:`path_ancestors` with every entry passed through
        :func:`mynotes.text.normalize.normalize_path_key`, e.g.
        ``"Projetos/Ação/x.md"`` → ``["projetos", "projetos/acao"]``.
    """
    return [normalize_path_key(ancestor) for ancestor in path_ancestors(rel_path)]


def long_path(p: Path) -> Path:
    """Return ``p`` with the Windows extended-length prefix when needed.

    Args:
        p: The path to open.

    Returns:
        On Windows, ``\\\\?\\``-prefixed path when the resolved path is at or
        over the 260-character limit and not already extended; unchanged
        otherwise. On other operating systems ``p`` itself.
    """
    if sys.platform != "win32":
        return p
    resolved = str(p.resolve())
    if len(resolved) >= WINDOWS_MAX_PATH and not resolved.startswith(WINDOWS_LONG_PATH_PREFIX):
        return Path(WINDOWS_LONG_PATH_PREFIX + resolved)
    return p


def is_within(root: Path, candidate: Path) -> bool:
    """Check whether ``candidate`` stays inside ``root``.

    Both paths are resolved first, so this protects against ``..`` components
    and symlinks that point outside ``root``.

    Args:
        root: The root directory that must contain the candidate.
        candidate: The path to test.

    Returns:
        True if the resolved ``candidate`` equals or is inside the resolved
        ``root``.
    """
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
