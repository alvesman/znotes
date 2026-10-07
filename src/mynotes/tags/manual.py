"""Manual tags of a document version (PLAN.md §7.4 steps 1-3).

Frontmatter ``tags``/``keywords``, inline hashtags and (with ``rel_path``)
folder rules and segments, all through the shared :mod:`mynotes.tags.normalize`
slug form. The functions are pure: :func:`manual_tags` falls back to the
built-in ``tags``/``versions`` defaults, so tests need no settings; production
callers pass the configured values explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mynotes.config import TagsConfig, VersionsConfig
from mynotes.tags.folder_rules import folder_tags
from mynotes.tags.normalize import normalize_tag

_FRONTMATTER_TAG_KEYS = ("tags", "keywords")


def _iter_frontmatter_tags(value: Any) -> list[str]:
    """Flatten a frontmatter ``tags``/``keywords`` value into raw tag strings.

    Args:
        value: The frontmatter value: a string (possibly comma-separated), a
            list/tuple of strings, or any scalar.

    Returns:
        The raw, un-normalized tag strings in document order.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",")]
    if isinstance(value, (list, tuple)):
        tags: list[str] = []
        for item in value:
            tags.extend(_iter_frontmatter_tags(item))
        return tags
    return [str(value)]


def _normalized_unique(raw: Sequence[str]) -> list[str]:
    """Normalize raw tags, dropping empties and duplicates (first wins)."""
    tags: list[str] = []
    seen: set[str] = set()
    for item in raw:
        tag = normalize_tag(item)
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tags


def frontmatter_tags(frontmatter: dict) -> list[str]:
    """Collect the normalized ``tags``/``keywords`` of a frontmatter.

    Args:
        frontmatter: Parsed frontmatter; ``tags`` and ``keywords`` are read,
            each a list or a comma-separated string.

    Returns:
        Normalized tags with duplicates removed keeping the first occurrence.
    """
    raw: list[str] = []
    for key in _FRONTMATTER_TAG_KEYS:
        raw.extend(_iter_frontmatter_tags(frontmatter.get(key)))
    return _normalized_unique(raw)


def hashtag_tags(hashtags: Sequence[str]) -> list[str]:
    """Collect the normalized inline hashtags (without the ``#``).

    Args:
        hashtags: Inline hashtags without the ``#``, in document order.

    Returns:
        Normalized tags with duplicates removed keeping the first occurrence.
    """
    return _normalized_unique(hashtags)


def manual_tags(
    frontmatter: dict,
    hashtags: Sequence[str],
    rel_path: str | None = None,
    cfg: TagsConfig | None = None,
) -> list[str]:
    """Collect the manually assigned tags of a document version.

    Args:
        frontmatter: Parsed frontmatter; ``tags`` and ``keywords`` are read,
            each a list or a comma-separated string.
        hashtags: Inline hashtags without the ``#``, in document order.
        rel_path: Document path relative to its source root; when given, the
            folder tags (rules and segments) are appended.
        cfg: The ``tags`` settings for the folder rules; ``None`` uses the
            built-in defaults (no rules, folder segments on).

    Returns:
        Normalized tags (lowercase, accent-folded, slug characters) with
        duplicates removed keeping the first occurrence: frontmatter tags,
        then hashtags, then folder tags.
    """
    tags = frontmatter_tags(frontmatter) + hashtag_tags(hashtags)
    if rel_path is not None:
        config = cfg if cfg is not None else TagsConfig()
        tags.extend(folder_tags(rel_path, config, VersionsConfig().archive_folders))
    return _normalized_unique(tags)
