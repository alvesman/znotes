"""Server-side Markdown rendering for the document view (task 48).

The extracted Markdown is rendered to HTML with :mod:`markdown_it` (raw HTML
in the source disabled, tables enabled). Every heading receives an ``id`` of
``sec-<slug>`` so citation links can scroll to it, and, when the URL carries
``?chunk=<ordinal>``, the paragraphs belonging to that chunk are wrapped in
``<mark>`` (best effort: the first 60 characters of each chunk block are
matched against the rendered paragraphs).
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Mapping

from markdown_it import MarkdownIt
from markdown_it.token import Token

from mynotes.text.normalize import normalize_key

_CHUNK_MATCH_CHARS = 60
"""Length of the chunk-block prefix used for the best-effort paragraph match."""


def heading_slug(text: str | None) -> str:
    """The ``sec-`` anchor slug for a heading (task 48).

    ``normalize_key`` first (accent/case folding, whitespace collapsed to one
    space), then spaces become ``-``, every other character outside
    ``[a-z0-9-]`` is removed and leading/trailing dashes are stripped.

    Args:
        text: The heading text, or None.

    Returns:
        The slug (empty for headings with no usable characters).
    """
    folded = normalize_key(text).replace(" ", "-")
    return re.sub(r"[^a-z0-9-]", "", folded).strip("-")


def render_document_markdown(markdown: str, chunk_text: str | None = None) -> str:
    """Render extracted Markdown to HTML for the document page (task 48).

    Args:
        markdown: The extracted Markdown of the document's latest version.
        chunk_text: The text of the ``?chunk=`` chunk to highlight, or None
            for no highlighting.

    Returns:
        The HTML. Raw HTML from the Markdown source is escaped by the parser;
        the only markup added here is heading ``id`` attributes and ``mark``
        wrappers, so the result is safe to include with autoescaping off.
    """
    parser = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = parser.parse(markdown)
    _add_heading_ids(tokens)
    if chunk_text:
        _mark_chunk_paragraphs(tokens, _chunk_block_keys(chunk_text))
    return parser.renderer.render(tokens, parser.options, {})


def render_help_markdown(
    markdown: str, *, doc_links: Mapping[str, str], source_base_url: str
) -> str:
    """Render one of the manuals for the Help page.

    Same parser and heading anchors as the document view, plus link rewriting
    so the manuals' own links keep working inside the UI:

    - ``#github-style-anchor`` becomes ``#sec-<slug>`` (the headings' ids);
    - ``other-manual.md[#anchor]`` becomes the Help tab in ``doc_links``
      (keyed by file name), keeping the anchor;
    - any other relative link points at the file under ``source_base_url``
      (the repository's ``docs/`` folder online).

    Args:
        markdown: The manual's Markdown source.
        doc_links: ``{"user-manual.md": "/ui/help/cli", ...}``.
        source_base_url: Base URL for relative links to files not shown in the UI.

    Returns:
        Safe HTML (raw HTML in the source is escaped by the parser).
    """
    parser = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = parser.parse(markdown)
    _add_heading_ids(tokens)
    for token in tokens:
        for child in token.children or []:
            if child.type == "link_open":
                href = child.attrGet("href")
                if isinstance(href, str):
                    child.attrSet("href", _rewrite_help_link(href, doc_links, source_base_url))
    return parser.renderer.render(tokens, parser.options, {})


def _rewrite_help_link(href: str, doc_links: Mapping[str, str], source_base_url: str) -> str:
    """Map a manual link to its Help-page equivalent (see :func:`render_help_markdown`)."""
    if href.startswith("#"):
        return f"#sec-{href[1:]}" if href[1:] and not href.startswith("#sec-") else href
    if re.match(r"^[a-z][a-z0-9+.-]*:", href) or href.startswith("/"):
        return href  # absolute URL, mailto:, or an app route
    path, _, fragment = href.partition("#")
    target = doc_links.get(posixpath.basename(path))
    if target is not None:
        return f"{target}#sec-{fragment}" if fragment else target
    normalized = posixpath.normpath(posixpath.join("docs", path))
    return f"{source_base_url.rstrip('/')}/{normalized}" + (f"#{fragment}" if fragment else "")


def _add_heading_ids(tokens: list[Token]) -> None:
    """Give every heading an ``id`` of ``sec-<slug>`` (empty slugs skipped)."""
    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        inline = tokens[index + 1] if index + 1 < len(tokens) else None
        if inline is None or inline.type != "inline":
            continue
        slug = heading_slug(inline.content)
        if slug:
            token.attrSet("id", f"sec-{slug}")


def _chunk_block_keys(chunk_text: str) -> set[str]:
    """The normalized 60-character prefix of every block of the chunk."""
    keys: set[str] = set()
    for block in chunk_text.split("\n\n"):
        cleaned = " ".join(block.split())
        if cleaned:
            keys.add(cleaned[:_CHUNK_MATCH_CHARS])
    return keys


def _mark_chunk_paragraphs(tokens: list[Token], block_keys: set[str]) -> None:
    """Wrap the paragraphs starting like a chunk block in ``<mark>``."""
    if not block_keys:
        return
    for index in range(len(tokens) - 1, -1, -1):
        if tokens[index].type != "paragraph_open":
            continue
        inline = tokens[index + 1] if index + 1 < len(tokens) else None
        if inline is None or inline.type != "inline":
            continue
        if " ".join(inline.content.split())[:_CHUNK_MATCH_CHARS] not in block_keys:
            continue
        close = _paragraph_close(tokens, index)
        if close is None:
            continue
        tokens.insert(close, Token("html_inline", "", 0, content="</mark>"))
        tokens.insert(index + 1, Token("html_inline", "", 0, content="<mark>"))


def _paragraph_close(tokens: list[Token], open_index: int) -> int | None:
    """The index of the ``paragraph_close`` following ``open_index``, or None."""
    for index in range(open_index + 1, len(tokens)):
        if tokens[index].type == "paragraph_close":
            return index
    return None
