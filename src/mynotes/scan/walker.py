"""Read-only scanner: walk a source root, apply ignore rules, flag placeholders.

Implements PLAN.md §7.1 steps 1-2 (walk + stat): every content file under a
root becomes a :class:`FileStat`. Cloud placeholders are flagged (never read
later), files modified less than ``settle_seconds`` ago are marked as not
settled, and legacy formats are recorded as ``"unsupported"``. The walk only
stats entries; hashing and database classification happen later (task 16).
"""

from __future__ import annotations

import logging
import os
import stat as stat_module
import time
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from mynotes.platform.fsinfo import is_cloud_placeholder
from mynotes.platform.paths import from_rel_path, is_within
from mynotes.scan.formats import format_for
from mynotes.scan.ignore import IGNORE_FILE_NAME, IgnoreRules

logger = logging.getLogger(__name__)

NS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class FileStat:
    """One content file seen by the scanner (PLAN.md §7.1).

    Attributes:
        rel_path: Path relative to the source root, POSIX separators, NFC.
        abs_path: The on-disk path (for symlinked files, the link itself).
        size: File size in bytes (of the symlink target, if any).
        mtime_ns: Modification time in nanoseconds since the epoch.
        format: Format name (``"md"``, ``"pdf"``) or ``"unsupported"`` for
            legacy extensions the scanner records but never extracts.
        placeholder: True for cloud placeholders (iCloud stubs, OneDrive
            online-only files, zero-byte documents); these are reported but
            never hashed or read.
        settled: False while the file was modified less than
            ``settle_seconds`` ago and may still be being written.
    """

    rel_path: str
    abs_path: Path
    size: int
    mtime_ns: int
    format: str
    placeholder: bool
    settled: bool
    content_hash: str | None = None
    """A hash computed elsewhere (an agent's manifest); the classifier then never reads the file."""


def walk_root(
    root: Path,
    rules: IgnoreRules,
    *,
    settle_seconds: int,
    skip_placeholders: bool = True,
    now_ns: int | None = None,
) -> Iterator[FileStat]:
    """Walk a source root and yield every content file.

    Ignores directories are pruned before descending, directory symlinks are
    never followed (no loops), and file symlinks are followed only when their
    target stays inside the walked root. Permission errors on a directory or
    file are logged and skipped; the walk never aborts.

    Args:
        root: The directory to walk.
        rules: The ignore rules (``rules.root`` anchors the relative paths
            when it contains ``root``; used by :func:`stat_paths`).
        settle_seconds: Grace period for files still being written.
        skip_placeholders: Forwarded intent for the pipeline: placeholders
            are always yielded (flagged) but must not be hashed or read when
            this is true.
        now_ns: Current time in nanoseconds (defaults to the real clock);
            pass a fixed value in tests.

    Yields:
        A :class:`FileStat` per content file, in deterministic (sorted,
        depth-first) order.
    """
    root = Path(root)
    anchor = _anchor_root(rules, root)
    now = now_ns if now_ns is not None else time.time_ns()
    yield from _walk_dir(root, rules, anchor, root, settle_seconds, skip_placeholders, now)


def stat_paths(
    root: Path,
    rel_paths: Iterable[str],
    rules: IgnoreRules,
    *,
    settle_seconds: int,
    now_ns: int | None = None,
) -> tuple[list[FileStat], list[str]]:
    """Stat specific paths of a source root (watcher hints, ``scan --paths``).

    Args:
        root: The source root the relative paths refer to.
        rel_paths: Relative paths to inspect; a path pointing at a directory
            expands to its contents (like walking that subfolder).
        rules: The ignore rules, anchored at ``root``.
        settle_seconds: Grace period for files still being written.
        now_ns: Current time in nanoseconds (defaults to the real clock).

    Returns:
        A tuple of the found file stats and the requested paths that do not
        resolve to scannable content (missing, ignored or not a document).
    """
    root = Path(root)
    anchor = _anchor_root(rules, root)
    now = now_ns if now_ns is not None else time.time_ns()
    found: list[FileStat] = []
    missing: list[str] = []
    for requested in rel_paths:
        requested = str(requested)
        rel = requested.replace("\\", "/").strip("/")
        target = from_rel_path(root, rel) if rel else None
        if target is None:
            missing.append(requested)
            continue
        try:
            if not is_within(root, target):
                missing.append(requested)
                continue
            st = os.stat(target)
        except OSError:
            missing.append(requested)
            continue
        if stat_module.S_ISDIR(st.st_mode):
            dir_rel = _rel_path(anchor, target)
            if not dir_rel or rules.is_ignored(dir_rel, True):
                missing.append(requested)
                continue
            found.extend(_walk_dir(target, rules, anchor, root, settle_seconds, True, now))
        elif stat_module.S_ISREG(st.st_mode):
            item = _stat_file(target, st, rules, anchor, settle_seconds, now)
            if item is None:
                missing.append(requested)
            else:
                found.append(item)
        else:
            missing.append(requested)
    return found, missing


def _anchor_root(rules: IgnoreRules, root: Path) -> Path:
    """Return the root that relative paths and ignore rules refer to.

    Lexical (not resolved) so the comparison is stable; falls back to the
    walk root when ``rules.root`` is not a prefix of it.
    """
    anchor = os.path.normpath(str(rules.root))
    walked = os.path.normpath(str(root))
    if walked == anchor or walked.startswith(anchor + os.sep):
        return rules.root
    return root


def _rel_path(anchor: Path, abs_path: Path) -> str:
    """Return the NFC, POSIX relative path of ``abs_path`` under ``anchor``.

    Lexical on purpose: symlinked files keep their own path instead of the
    resolved target's path (two links to one target must not collide, and a
    link must not shadow its target). For every non-symlink entry the result
    equals :func:`mynotes.platform.paths.to_rel_path`.

    Returns:
        The relative path, or ``""`` when ``abs_path`` is not strictly under
        ``anchor``.
    """
    try:
        rel = os.path.relpath(abs_path, anchor)
    except ValueError:  # e.g. different drives on Windows
        return ""
    if rel == "." or rel.startswith(".."):
        return ""
    return unicodedata.normalize("NFC", Path(rel).as_posix())


def _walk_dir(
    dir_path: Path,
    rules: IgnoreRules,
    anchor: Path,
    root: Path,
    settle_seconds: int,
    skip_placeholders: bool,
    now_ns: int,
) -> Iterator[FileStat]:
    """Walk one directory, honoring discovered ``.mynotesignore`` files."""
    try:
        with os.scandir(dir_path) as it:
            entries = sorted(it, key=lambda entry: entry.name)
    except OSError as exc:
        logger.warning("Cannot list directory %s: %s", dir_path, exc)
        return

    ignore_entry = next((e for e in entries if e.name == IGNORE_FILE_NAME), None)
    if ignore_entry is not None:
        _load_ignore_file(ignore_entry, rules, anchor)

    for entry in entries:
        if entry.name == IGNORE_FILE_NAME:
            continue
        try:
            if entry.is_symlink():
                item = _walk_symlink(entry, rules, anchor, root, settle_seconds, now_ns)
                if item is not None:
                    yield item
            elif entry.is_dir(follow_symlinks=False):
                rel = _rel_path(anchor, Path(entry.path))
                if rel and rules.is_ignored(rel, True):
                    continue
                yield from _walk_dir(
                    Path(entry.path), rules, anchor, root, settle_seconds, skip_placeholders, now_ns
                )
            elif entry.is_file(follow_symlinks=False):
                item = _stat_entry(entry, rules, anchor, settle_seconds, now_ns)
                if item is not None:
                    yield item
        except OSError as exc:
            logger.warning("Skipping %s during scan: %s", entry.path, exc)
            continue


def _walk_symlink(
    entry: os.DirEntry,
    rules: IgnoreRules,
    anchor: Path,
    root: Path,
    settle_seconds: int,
    now_ns: int,
) -> FileStat | None:
    """Handle one symlink entry: dir links are never followed."""
    if entry.is_dir(follow_symlinks=True):
        logger.debug("Not following directory symlink %s", entry.path)
        return None
    if not entry.is_file(follow_symlinks=True):
        return None
    if not is_within(root, Path(entry.path)):
        logger.debug("Ignoring file symlink pointing outside %s: %s", root, entry.path)
        return None
    return _stat_entry(entry, rules, anchor, settle_seconds, now_ns)


def _stat_entry(
    entry: os.DirEntry,
    rules: IgnoreRules,
    anchor: Path,
    settle_seconds: int,
    now_ns: int,
) -> FileStat | None:
    """Build a :class:`FileStat` for one file entry, or None to skip it."""
    fmt = format_for(entry.path)
    if fmt is None:
        return None
    rel = _rel_path(anchor, Path(entry.path))
    if not rel or rules.is_ignored(rel, False):
        return None
    st = entry.stat(follow_symlinks=True)
    return _file_stat(rel, Path(entry.path), st, fmt, settle_seconds, now_ns)


def _stat_file(
    abs_path: Path,
    st: os.stat_result,
    rules: IgnoreRules,
    anchor: Path,
    settle_seconds: int,
    now_ns: int,
) -> FileStat | None:
    """Build a :class:`FileStat` from a plain path (used by :func:`stat_paths`)."""
    fmt = format_for(abs_path)
    if fmt is None:
        return None
    rel = _rel_path(anchor, abs_path)
    if not rel or rules.is_ignored(rel, False):
        return None
    return _file_stat(rel, abs_path, st, fmt, settle_seconds, now_ns)


def _file_stat(
    rel: str,
    abs_path: Path,
    st: os.stat_result,
    fmt: str,
    settle_seconds: int,
    now_ns: int,
) -> FileStat:
    """Assemble a :class:`FileStat` from an already resolved stat result."""
    return FileStat(
        rel_path=rel,
        abs_path=abs_path,
        size=st.st_size,
        mtime_ns=st.st_mtime_ns,
        format=fmt,
        placeholder=is_cloud_placeholder(abs_path, st),
        settled=(now_ns - st.st_mtime_ns) >= settle_seconds * NS_PER_SECOND,
    )


def _load_ignore_file(entry: os.DirEntry, rules: IgnoreRules, anchor: Path) -> None:
    """Read a discovered ``.mynotesignore`` and add it to the rules."""
    try:
        text = Path(entry.path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Cannot read ignore file %s: %s", entry.path, exc)
        return
    rules.add_ignore_file(_rel_path(anchor, Path(entry.path).parent), text)
