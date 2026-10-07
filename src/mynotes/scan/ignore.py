"""Ignore rules for the scanner (gitignore semantics via ``pathspec``).

Two sources feed :class:`IgnoreRules`: the configured globs (``scan.ignore``;
defaults in :data:`DEFAULT_IGNORES`) and every ``.mynotesignore`` file the
walker discovers. A ``.mynotesignore`` in folder ``X`` applies to everything
under ``X`` with patterns relative to ``X``, like ``.gitignore``. The last
matching pattern within one file wins (so ``!keep.pdf`` can un-ignore), and a
deeper ignore file overrides the decision of a shallower one.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pathspec import GitIgnoreSpec

IGNORE_FILE_NAME = ".mynotesignore"

DEFAULT_IGNORES: list[str] = [
    ".git/",
    "node_modules/",
    "~$*",
    ".~lock*",
    "*.tmp",
    "*.crdownload",
    "*.partial",
    ".DS_Store",
    "._*",
    "Thumbs.db",
    "desktop.ini",
    "Icon\r",
]
"""Default ``scan.ignore`` globs (PLAN.md §12) plus the macOS ``Icon\\r``
clutter file. ``*.icloud`` is deliberately absent: cloud placeholders are
reported, not ignored (PLAN.md §7.8)."""


class IgnoreRules:
    """Gitignore-style matcher combining config globs and ignore files.

    Args:
        root: The source root the relative paths refer to.
        config_globs: The configured ignore globs (``scan.ignore``), always
            active for the whole tree.
    """

    def __init__(self, root: Path, config_globs: Sequence[str]) -> None:
        self.root = root
        self._config_spec = GitIgnoreSpec.from_lines([*config_globs, IGNORE_FILE_NAME])
        self._file_specs: list[tuple[str, GitIgnoreSpec]] = []

    def add_ignore_file(self, folder_rel: str, text: str) -> None:
        """Add the patterns of one discovered ``.mynotesignore`` file.

        Args:
            folder_rel: The folder containing the file, relative to the
                source root (``""`` for the root itself), POSIX separators.
            text: The raw file content; blank lines and ``#`` comments are
                ignored by the gitignore parser.
        """
        base = folder_rel.strip("/")
        if not text.strip():
            return
        self._file_specs.append((base, GitIgnoreSpec.from_lines(text.splitlines())))

    def is_ignored(self, rel_path: str, is_dir: bool) -> bool:
        """Check whether a path is ignored.

        Args:
            rel_path: Path relative to the source root, POSIX separators.
            is_dir: Whether the path is a directory; directory paths are
                tested with a trailing ``/`` so patterns like
                ``node_modules/`` match.

        Returns:
            True if the path should be ignored. The decision comes from the
            deepest matching source (config globs first, then ignore files in
            discovery order); within one source the last matching pattern
            wins, including ``!`` negations.
        """
        rel = rel_path.replace("\\", "/").strip("/")
        if not rel:
            return False
        candidate = f"{rel}/" if is_dir else rel
        decision = False
        config_result = self._config_spec.check_file(candidate)
        if config_result.include is not None:
            decision = config_result.include
        for base, spec in self._file_specs:
            if base:
                prefix = f"{base}/"
                if not rel.startswith(prefix):
                    continue
                scoped = rel[len(prefix) :]
            else:
                scoped = rel
            result = spec.check_file(f"{scoped}/" if is_dir else scoped)
            if result.include is not None:
                decision = result.include
        return decision
