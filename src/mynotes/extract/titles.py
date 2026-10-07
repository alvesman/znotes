"""Title resolution following the PLAN.md §7.2 precedence order."""

from __future__ import annotations

import re
from pathlib import Path

from mynotes.extract.ir import IRBlock

_EMPHASIS_RE = re.compile(r"[*_]+")
_SEPARATOR_RE = re.compile(r"[_\-\s]+")


def humanize_filename(filename: str) -> str:
    """Turn a file name into a human title.

    Args:
        filename: A file name (the extension is dropped).

    Returns:
        The stem with ``_``/``-`` runs replaced by single spaces and the first
        letter capitalized: ``"my_idea-v2.md"`` becomes ``"My idea v2"``.
    """
    stem = Path(filename).stem
    collapsed = " ".join(_SEPARATOR_RE.sub(" ", stem).split())
    if not collapsed:
        return ""
    return collapsed[0].upper() + collapsed[1:]


def resolve_title(
    *,
    frontmatter: dict,
    blocks: list[IRBlock],
    filename: str,
    largest_font_line: str | None = None,
    first_slide_title: str | None = None,
    metadata_title: str | None = None,
) -> str:
    """Resolve a document title.

    Args:
        frontmatter: Parsed frontmatter of the document.
        blocks: Extracted blocks, in document order.
        filename: File name used for the humanized-stem fallback.
        largest_font_line: Largest-font line (PDF extractor), if any.
        first_slide_title: First slide title (PPTX extractor), if any.
        metadata_title: Document-metadata title (Office core properties), if
            any; must already be filtered for placeholders by the caller.

    Returns:
        The first available of: frontmatter ``title``; first level-1 heading
        with at least 3 characters after stripping ``*``/``_``; the largest
        font line; the first slide title; the metadata title; the humanized
        file name stem.
    """
    value = frontmatter.get("title")
    if value is not None and str(value).strip():
        return str(value).strip()
    for block in blocks:
        if block.kind == "heading" and block.level == 1:
            without_emphasis = _EMPHASIS_RE.sub("", block.text).strip()
            if len(without_emphasis) >= 3:
                return block.text.strip()
    if largest_font_line and largest_font_line.strip():
        return largest_font_line.strip()
    if first_slide_title and first_slide_title.strip():
        return first_slide_title.strip()
    if metadata_title and metadata_title.strip():
        return metadata_title.strip()
    return humanize_filename(filename)
