"""Common intermediate representation (IR) returned by every extractor.

PLAN.md §7.2: each extractor turns one source document into an
:class:`IRDocument` made of ordered :class:`IRBlock`s plus metadata, so the
chunker (task 19) and the graph builders are format-agnostic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from typing import Any, Literal

BlockKind = Literal["heading", "paragraph", "list", "table", "code", "image_caption", "page_break"]
"""Kinds of blocks every extractor can emit."""

LinkKind = Literal["wikilink", "markdown", "url"]
"""Kinds of links collected for the structural graph (task 39)."""

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")


@dataclass
class IRBlock:
    """One block of content in document order.

    Attributes:
        kind: The block kind (one of :data:`BlockKind`).
        text: Markdown source for the block (for headings, the heading text
            without the ``#`` markers).
        level: Heading level 1-6, only for ``kind="heading"``.
        page: 1-based PDF page or PPTX slide number; None for md/txt.
    """

    kind: BlockKind
    text: str
    level: int | None = None
    page: int | None = None


@dataclass
class Link:
    """One link found in the document.

    Attributes:
        target: Wikilink target, relative path or URL, exactly as written.
        kind: How the link was written in the source.
        text: Visible link text, when the source has one.
    """

    target: str
    kind: LinkKind
    text: str | None = None


@dataclass
class IRDocument:
    """Extraction result for one document version.

    Attributes:
        blocks: Content blocks in document order.
        title: Resolved title (task 17 resolves it for md/txt).
        frontmatter: Parsed frontmatter; values may be dates or lists.
        links: Markdown links, URLs and wikilinks in document order.
        hashtags: Inline hashtags without ``#``, original case, de-duplicated
            in order of first appearance.
        language: ``"pt"``, ``"en"`` or None when undetected.
        doc_date: Resolved document date, timezone-aware UTC.
        extractor: Name of the extractor that produced this document.
        extractor_version: Version of that extractor.
        warnings: Non-fatal extraction notes (e.g. ``"needs_ocr"``).
    """

    blocks: list[IRBlock]
    title: str | None = None
    frontmatter: dict[str, Any] = field(default_factory=dict)
    links: list[Link] = field(default_factory=list)
    hashtags: list[str] = field(default_factory=list)
    language: str | None = None
    doc_date: datetime | None = None
    extractor: str = ""
    extractor_version: str = ""
    warnings: list[str] = field(default_factory=list)

    def plain_text(self) -> str:
        """Return every block text joined with blank lines.

        Returns:
            The concatenation of all block texts separated by ``\n\n``; an
            empty string when the document has no blocks.
        """
        return "\n\n".join(block.text for block in self.blocks)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe dict of the document.

        Returns:
            A dict safe for ``json.dumps``: datetimes (including values nested
            in the frontmatter) become ISO strings, everything else is kept.
        """
        return {
            "blocks": [
                {"kind": block.kind, "text": block.text, "level": block.level, "page": block.page}
                for block in self.blocks
            ],
            "title": self.title,
            "frontmatter": _json_safe(self.frontmatter),
            "links": [
                {"target": link.target, "kind": link.kind, "text": link.text} for link in self.links
            ],
            "hashtags": list(self.hashtags),
            "language": self.language,
            "doc_date": self.doc_date.isoformat() if self.doc_date else None,
            "extractor": self.extractor,
            "extractor_version": self.extractor_version,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> IRDocument:
        """Rebuild a document from :meth:`to_json` output.

        Args:
            data: The dict produced by :meth:`to_json`. Missing keys fall back
                to the field defaults. ISO strings for ``doc_date`` and for
                frontmatter date values are parsed back into ``datetime`` /
                ``date`` objects so a round trip preserves the values.

        Returns:
            The rebuilt :class:`IRDocument`.
        """
        blocks = [
            IRBlock(
                kind=block["kind"],
                text=block["text"],
                level=block.get("level"),
                page=block.get("page"),
            )
            for block in data.get("blocks", [])
        ]
        links = [
            Link(target=link["target"], kind=link["kind"], text=link.get("text"))
            for link in data.get("links", [])
        ]
        doc_date = _parse_datetime(data.get("doc_date"))
        return cls(
            blocks=blocks,
            title=data.get("title"),
            frontmatter=_restore_dates(data.get("frontmatter") or {}),
            links=links,
            hashtags=list(data.get("hashtags", [])),
            language=data.get("language"),
            doc_date=doc_date,
            extractor=data.get("extractor", ""),
            extractor_version=data.get("extractor_version", ""),
            warnings=list(data.get("warnings", [])),
        )


def _json_safe(value: Any) -> Any:
    """Recursively convert a frontmatter value into JSON-safe data."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _restore_dates(value: Any) -> Any:
    """Recursively parse ISO date strings (written by :func:`_json_safe`) back."""
    if isinstance(value, dict):
        return {str(key): _restore_dates(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_dates(item) for item in value]
    if isinstance(value, str):
        if _ISO_DATE_RE.fullmatch(value):
            try:
                return date.fromisoformat(value)
            except ValueError:
                return value
        if _ISO_DATETIME_RE.match(value):
            parsed = _parse_datetime(value)
            if parsed is not None:
                return parsed
    return value


def _parse_datetime(value: Any) -> datetime | None:
    """Parse a JSON value into a timezone-aware UTC datetime, or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed
