"""Markdown and plain-text extractors (PLAN.md §7.2).

Both produce the common IR: the Markdown extractor parses frontmatter with
python-frontmatter and the body with markdown-it-py (GFM tables enabled),
slicing block text from the original source lines; the plain-text extractor
splits paragraphs on blank lines. ``markdown_to_blocks`` is the shared
Markdown-to-IR entry point reused by the PDF extractor (task 18).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import frontmatter
import yaml
from markdown_it import MarkdownIt
from markdown_it.token import Token

from mynotes.extract.dates import resolve_doc_date
from mynotes.extract.ir import IRBlock, IRDocument, Link
from mynotes.extract.registry import register
from mynotes.extract.titles import humanize_filename, resolve_title
from mynotes.text.language import detect_language

logger = logging.getLogger(__name__)


def markdown_to_blocks(md: str, page: int | None = None) -> tuple[list[IRBlock], list[Link]]:
    """Parse a Markdown string into the common IR blocks and links.

    Shared by the Markdown extractor and the PDF extractor (which feeds it one
    page of pymupdf4llm output at a time).

    Args:
        md: The Markdown source to parse (GFM tables enabled).
        page: 1-based page number stamped on every block; None for md/txt.

    Returns:
        ``(blocks, links)`` with blocks in document order and the Markdown
        links, URLs and wikilinks found in the source.
    """
    tokens = MarkdownIt("commonmark").enable("table").parse(md)
    lines = md.splitlines()
    blocks = _build_blocks(tokens, lines, page=page)
    links = _collect_links(tokens)
    links.extend(_collect_wikilinks(blocks))
    return blocks, links


_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]")
_HASHTAG_RE = re.compile(r"(?<![\w&/#])#([^\W\d_][\w\-/]*)")
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*\]\([^)]*\)")
_AUTOLINK_RE = re.compile(r"<https?://[^>\s]+>")
_ANY_WIKILINK_RE = re.compile(r"\[\[[^\]]*\]\]")
_MIN_TITLE_LENGTH = 120


def _read_text(path: Path, warnings: list[str]) -> str:
    """Read a file as UTF-8, replacing undecodable bytes with a warning."""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        logger.debug("Invalid UTF-8 in %s; replacing undecodable bytes", path)
        warnings.append("invalid_utf8")
        return raw.decode("utf-8", errors="replace")


def _split_frontmatter(text: str, warnings: list[str]) -> tuple[dict, str]:
    """Front matter and body; a leading ``---`` block that is not valid YAML stays text.

    Obsidian notes may start with a horizontal rule, or carry ``%% comments %%``
    or templating inside the block; python-frontmatter then raises a YAML
    error. Such a note is extracted whole, with a warning, rather than failing.
    """
    try:
        post = frontmatter.loads(text)
    except yaml.YAMLError as exc:
        warnings.append(f"front matter is not valid YAML, kept as text: {exc}".splitlines()[0])
        return {}, text
    return dict(post.metadata), post.content


class MarkdownExtractor:
    """Extractor for Markdown notes (``.md``/``.markdown``)."""

    name = "markdown"
    version = "1"
    formats = ("md",)

    def extract(self, path: Path) -> IRDocument:
        """Extract a Markdown file into the common IR.

        Args:
            path: The file to read.

        Returns:
            The IR document with blocks sliced from the source, links,
            hashtags, resolved language, title and document date.
        """
        warnings: list[str] = []
        text = _read_text(path, warnings)
        meta, content = _split_frontmatter(text, warnings)

        blocks, links = markdown_to_blocks(content)
        document = IRDocument(
            blocks=blocks,
            frontmatter=meta,
            links=links,
            hashtags=_collect_hashtags(blocks),
            extractor=self.name,
            extractor_version=self.version,
            warnings=warnings,
        )
        document.language = detect_language(document.plain_text())
        document.title = resolve_title(frontmatter=meta, blocks=blocks, filename=path.name)
        document.doc_date = resolve_doc_date(
            frontmatter=meta,
            filename=path.name,
            metadata_date=None,
            mtime_ns=path.stat().st_mtime_ns,
        )
        return document


class TextExtractor:
    """Extractor for plain-text files (``.txt``)."""

    name = "text"
    version = "1"
    formats = ("txt",)

    def extract(self, path: Path) -> IRDocument:
        """Extract a plain-text file into the common IR.

        Args:
            path: The file to read.

        Returns:
            The IR document with one paragraph block per blank-line-separated
            paragraph, the title taken from the first non-empty line when it
            is shorter than 120 characters (the humanized file name
            otherwise), plus language and document date.
        """
        warnings: list[str] = []
        text = _read_text(path, warnings)
        blocks = [
            IRBlock(kind="paragraph", text=paragraph)
            for paragraph in (part.strip() for part in re.split(r"\n\s*\n", text))
            if paragraph
        ]
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        title = first_line if first_line and len(first_line) < _MIN_TITLE_LENGTH else None
        document = IRDocument(
            blocks=blocks,
            title=title if title is not None else humanize_filename(path.name),
            extractor=self.name,
            extractor_version=self.version,
            warnings=warnings,
        )
        document.language = detect_language(document.plain_text())
        document.doc_date = resolve_doc_date(
            frontmatter={},
            filename=path.name,
            metadata_date=None,
            mtime_ns=path.stat().st_mtime_ns,
        )
        return document


def _build_blocks(tokens: list[Token], lines: list[str], page: int | None = None) -> list[IRBlock]:
    """Build IR blocks from the top-level tokens of a parsed document."""
    blocks: list[IRBlock] = []
    for index, token in enumerate(tokens):
        if token.level != 0 or token.map is None:
            continue
        start, end = token.map
        source = "\n".join(lines[start:end]).rstrip("\n")
        if token.type == "heading_open":
            blocks.append(
                IRBlock(
                    kind="heading",
                    text=_inline_content(tokens, index),
                    level=int(token.tag[1]),
                    page=page,
                )
            )
        elif token.type == "paragraph_open":
            blocks.append(IRBlock(kind="paragraph", text=source, page=page))
        elif token.type in ("bullet_list_open", "ordered_list_open"):
            blocks.append(IRBlock(kind="list", text=source, page=page))
        elif token.type == "table_open":
            blocks.append(IRBlock(kind="table", text=source, page=page))
        elif token.type in ("fence", "code_block"):
            blocks.append(IRBlock(kind="code", text=source, page=page))
        elif token.type == "blockquote_open":
            blocks.append(IRBlock(kind="paragraph", text=source, page=page))
    return blocks


def _inline_content(tokens: list[Token], index: int) -> str:
    """Return the inline text following an opening token such as heading_open."""
    for token in tokens[index + 1 :]:
        if token.type == "inline":
            return token.content
        if token.type.endswith("_close"):
            break
    return ""


def _collect_links(tokens: list[Token]) -> list[Link]:
    """Collect markdown links and bare URLs from inline tokens, in order."""
    links: list[Link] = []
    for token in tokens:
        if token.type != "inline" or not token.children:
            continue
        children = token.children
        position = 0
        while position < len(children):
            child = children[position]
            if child.type != "link_open":
                position += 1
                continue
            href = child.attrGet("href") or ""
            text_parts: list[str] = []
            depth = 1
            position += 1
            while position < len(children) and depth > 0:
                inner = children[position]
                if inner.type == "link_open":
                    depth += 1
                elif inner.type == "link_close":
                    depth -= 1
                elif depth == 1 and inner.type == "text":
                    text_parts.append(inner.content)
                position += 1
            kind = "url" if href.startswith(("http://", "https://")) else "markdown"
            text = "".join(text_parts).strip()
            links.append(Link(target=href, kind=kind, text=text or None))
    return links


def _collect_wikilinks(blocks: list[IRBlock]) -> list[Link]:
    """Collect wikilinks from every non-code block, in document order."""
    links: list[Link] = []
    for block in blocks:
        if block.kind == "code":
            continue
        for match in _WIKILINK_RE.finditer(block.text):
            links.append(Link(target=match.group(1).strip(), kind="wikilink", text=match.group(2)))
    return links


def _collect_hashtags(blocks: list[IRBlock]) -> list[str]:
    """Collect hashtags from paragraph and list blocks, de-duplicated in order."""
    hashtags: list[str] = []
    seen: set[str] = set()
    for block in blocks:
        if block.kind not in ("paragraph", "list"):
            continue
        cleaned = _INLINE_CODE_RE.sub(" ", block.text)
        cleaned = _MARKDOWN_LINK_RE.sub(" ", cleaned)
        cleaned = _AUTOLINK_RE.sub(" ", cleaned)
        cleaned = _ANY_WIKILINK_RE.sub(" ", cleaned)
        for match in _HASHTAG_RE.finditer(cleaned):
            tag = match.group(1)
            if tag not in seen:
                seen.add(tag)
                hashtags.append(tag)
    return hashtags


register(MarkdownExtractor())
register(TextExtractor())
