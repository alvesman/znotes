"""Tag name normalization and kinds (PLAN.md §7.4 step 3).

Every tag stored in ``tag.name`` or used as a query filter goes through
:func:`normalize_tag`: the shared :func:`~mynotes.text.normalize.normalize_key`
folding (case and accent), spaces as ``-``, and a slug character set that keeps
one ``:`` prefix separator (``project:``, ``status:``, ...) so tag kinds survive
normalization.
"""

from __future__ import annotations

import re

from mynotes.text.normalize import normalize_key

_ALLOWED_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-:/_")
_SEPARATOR_RUN = re.compile(r"[-:]+")

_PREFIX_KINDS: tuple[tuple[str, str], ...] = (
    ("project:", "project"),
    ("status:", "status"),
    ("folder:", "folder"),
    ("topic:", "topic"),
)


def normalize_tag(raw: str) -> str:
    """Fold one raw tag into its normalized slug form.

    Args:
        raw: The tag as written in the frontmatter, a hashtag, a folder rule
            template or a search filter.

    Returns:
        :func:`~mynotes.text.normalize.normalize_key` with spaces as ``-``,
        characters outside ``[a-z0-9\\-:/_]`` dropped, runs of ``-``/``:``
        collapsed to a single ``:`` when the run contains one (the prefix
        separator, ``"Project: Alpha"`` → ``"project:alpha"``) and to ``-``
        otherwise, and ``-``/``:`` stripped from both ends. Empty when nothing
        remains (callers skip the tag).
    """
    folded = normalize_key(raw).replace(" ", "-")
    cleaned = "".join(char for char in folded if char in _ALLOWED_CHARS)
    collapsed = _SEPARATOR_RUN.sub(lambda m: ":" if ":" in m.group() else "-", cleaned)
    return collapsed.strip("-:")


def tag_kind(name: str) -> str:
    """Return the kind of a normalized tag name.

    Args:
        name: A normalized tag (``normalize_tag`` output).

    Returns:
        ``project``/``status``/``folder``/``topic`` for the matching ``:``
        prefix, else ``other``.
    """
    for prefix, kind in _PREFIX_KINDS:
        if name.startswith(prefix):
            return kind
    return "other"
