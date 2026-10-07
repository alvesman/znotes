"""Folder tags from configurable rules and folder segments (PLAN.md §7.4 step 1).

:func:`folder_tags` matches a document's folder against the ``folder_rules``
globs (``{name}`` captures one segment, ``**`` any depth, ``*`` within a
segment) and turns every remaining folder segment into a ``folder:<segment>``
tag when ``folder_segments_as_tags`` is on. Matching happens on the folded
folder key (:func:`~mynotes.text.normalize.normalize_path_key`), so rules are
case- and accent-insensitive.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from mynotes.config import TagsConfig, VersionsConfig
from mynotes.tags.normalize import normalize_tag
from mynotes.text.normalize import normalize_key, normalize_path_key

logger = logging.getLogger(__name__)

_PLACEHOLDER_END = "}"


def _is_wildcard_segment(segment: str) -> bool:
    """A bare ``*``/``**`` segment expresses no intent for its folder segment."""
    return segment in ("*", "**")


def _fold_pattern_segments(pattern: str) -> list[str]:
    """Fold every pattern segment for accent/case-insensitive comparison."""
    return [normalize_key(segment) for segment in pattern.strip().strip("/").split("/")]


def _match_rule(segments: list[str], pattern: str) -> tuple[dict[str, str], set[int]] | None:
    """Match folded path segments against one rule pattern.

    Args:
        segments: The folded folder segments (no empties).
        pattern: The rule's ``pattern`` with ``{name}`` placeholders and globs.

    Returns:
        The placeholder captures (folder segment each, un-normalized) plus the
        indices of the subject segments that a non-wildcard pattern segment
        consumed, or ``None`` when the pattern does not match the whole folder.
    """
    pattern_segments = _fold_pattern_segments(pattern)
    memo: dict[tuple[int, int], tuple[dict[str, str], set[int]] | None] = {}

    def walk(pattern_index: int, segment_index: int) -> tuple[dict[str, str], set[int]] | None:
        key = (pattern_index, segment_index)
        if key in memo:
            return memo[key]
        if pattern_index == len(pattern_segments):
            result: tuple[dict[str, str], set[int]] | None = (
                ({}, set()) if segment_index == len(segments) else None
            )
        else:
            pattern_segment = pattern_segments[pattern_index]
            result = None
            if pattern_segment == "**":
                for take in range(len(segments) - segment_index + 1):
                    rest = walk(pattern_index + 1, segment_index + take)
                    if rest is not None:
                        result = rest
                        break
            elif segment_index < len(segments):
                subject = segments[segment_index]
                if _is_wildcard_segment(pattern_segment):
                    rest = walk(pattern_index + 1, segment_index + 1)
                    if rest is not None:
                        result = rest
                elif pattern_segment.startswith("{") and pattern_segment.endswith("}"):
                    name = pattern_segment[1:-1]
                    rest = walk(pattern_index + 1, segment_index + 1)
                    if rest is not None:
                        captures, excluded = rest
                        result = ({**captures, name: subject}, excluded)
                elif pattern_segment == normalize_key(subject):
                    rest = walk(pattern_index + 1, segment_index + 1)
                    if rest is not None:
                        result = rest
                if result is not None and not _is_wildcard_segment(pattern_segment):
                    captures, excluded = result
                    result = (captures, {*excluded, segment_index})
        memo[key] = result
        return result

    return walk(0, 0)


def _fill_template(template: str, captures: dict[str, str]) -> str | None:
    """Fill a rule's tag template with the normalized captures, or None.

    Args:
        template: A tag template such as ``"project:{project}"``.
        captures: The rule's captured folder segments, already normalized.

    Returns:
        The filled template, or ``None`` when a placeholder has no capture or
        a capture normalized to the empty string.
    """
    filled = template
    start = template.find("{")
    while start != -1:
        end = template.find(_PLACEHOLDER_END, start + 1)
        if end == -1:
            return filled
        name = template[start + 1 : end]
        value = captures.get(name)
        if not value:
            logger.debug("folder rule template %r: no capture for {%s}", template, name)
            return None
        filled = filled.replace("{" + name + "}", value)
        start = filled.find("{", start + len(value))
    return filled


def folder_tags(
    rel_path: str,
    cfg: TagsConfig,
    archive_folders: Sequence[str] | None = None,
) -> list[str]:
    """Compute the folder-derived tags of a document path.

    Args:
        rel_path: The document path relative to its source root (POSIX ``/``).
        cfg: The ``tags`` settings (``folder_rules``,
            ``folder_segments_as_tags``, ``ignore_segments``).
        archive_folders: Names of archive folders whose segments never become
            ``folder:`` tags; ``None`` uses the built-in
            ``versions.archive_folders`` default.

    Returns:
        Normalized, de-duplicated tags: the matched rules' filled templates
        first (in rule order), then ``folder:<segment>`` for every folder
        segment that no matched rule consumed and that is not ignored or an
        archive folder.
    """
    if archive_folders is None:
        archive_folders = VersionsConfig().archive_folders
    folder = rel_path.rsplit("/", 1)[0] if "/" in rel_path else ""
    key = normalize_path_key(folder)
    segments = [segment for segment in key.split("/") if segment]
    tags: list[str] = []
    consumed: set[int] = set()
    for rule in cfg.folder_rules:
        match = _match_rule(segments, rule.pattern)
        if match is None:
            continue
        captures, rule_consumed = match
        consumed |= rule_consumed
        captures = {name: normalize_tag(value) for name, value in captures.items()}
        for template in rule.tags:
            filled = _fill_template(template, captures)
            if filled is not None:
                tags.append(normalize_tag(filled))
    if cfg.folder_segments_as_tags:
        ignored = {normalize_key(segment) for segment in cfg.ignore_segments}
        archives = {normalize_key(segment) for segment in archive_folders}
        for index, segment in enumerate(segments):
            if index in consumed:
                continue
            if segment in ignored or segment in archives:
                continue
            tags.append(normalize_tag(f"folder:{segment}"))
    unique: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if tag and tag not in seen:
            seen.add(tag)
            unique.append(tag)
    return unique
