"""PDF extractor built on pymupdf4llm (PLAN.md §7.2).

Per-page Markdown via ``pymupdf4llm.helpers.pymupdf_rag.to_markdown``,
promotion of numbered ALL-CAPS headings that pymupdf4llm misses (manuals
typeset their headings at body size), a largest-font title fallback, and
detection of image-only PDFs: below the text threshold an optional OCR pass
(``extract/ocr.py``, enabled with ``extract.pdf_ocr``) re-runs extraction on
the OCR'd copy; otherwise the IR carries the ``needs_ocr`` warning.

PyMuPDF is AGPL-3.0 licensed; every PyMuPDF import stays inside this module
so the backend can be swapped later (PLAN.md §16).
"""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pymupdf

from mynotes.config import get_settings
from mynotes.extract.dates import resolve_doc_date
from mynotes.extract.ir import IRBlock, IRDocument, Link
from mynotes.extract.markdown import markdown_to_blocks
from mynotes.extract.ocr import ocr_available, ocr_pdf
from mynotes.extract.registry import register
from mynotes.extract.titles import resolve_title
from mynotes.text.language import detect_language

logger = logging.getLogger(__name__)

MIN_EXTRACTABLE_TEXT_CHARS = 50
"""Below this many extractable characters the PDF is almost certainly an
image-only scan: without OCR (or with OCR disabled) the document would yield
no chunks."""

TITLE_FALLBACK_PAGES = 2
"""Number of leading pages scanned for the largest-font line used as the
document title when the Markdown has no H1 (cover pages are often images)."""

_MARKDOWN_HEADING = re.compile(r"(?m)^#{1,6}\s+\S")
# "1. PIEZAS Y COMPONENTES", "2.3 MONTAGEM": numbered, ALL-CAPS section
# titles typeset at body size. pymupdf4llm picks headings by font size, so
# manuals that style their headings like this (Cecotec, ...) come out with no
# headings at all. A top-level number must carry a dot ("1.") so running
# footers like "4 BAMBA IONICARE RADIANCE 5" never qualify.
_NUMBERED_CAPS_HEADING = re.compile(
    r"^(?P<num>\d{1,2}(?:\.\d{1,2})+\.?|\d{1,2}\.)\s+"
    r"(?P<title>[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ0-9 ,'’&/()\-]{2,70})$"  # noqa: RUF001
)
_PDF_DATE_RE = re.compile(
    r"^(?:D:)?"
    r"(?P<year>\d{4})"
    r"(?P<month>\d{2})?"
    r"(?P<day>\d{2})?"
    r"(?P<hour>\d{2})?"
    r"(?P<minute>\d{2})?"
    r"(?P<second>\d{2})?"
    r"(?:(?P<sign>[+\-Zz])(?P<off_hour>\d{2})?'?(?P<off_minute>\d{2})?'?)?"
    r"$"
)
"""PDF 32000-1 §7.9.4 date syntax: D:YYYYMMDDHHmmSSOHH'mm', all but the year
optional; the apostrophes around the UTC offset may be omitted."""


def _load_to_markdown() -> Callable[..., object]:
    """Return ``pymupdf4llm.to_markdown`` with the layout backend disabled.

    Newer PyMuPDF builds expose a layout backend that triggers ONNX/GPU
    probing even though only Markdown text is needed; clobbering
    ``pymupdf._get_layout`` and silencing ``pymupdf._warn_layout_once`` before
    importing keeps the process lightweight.

    Returns:
        The ``to_markdown`` function.

    Raises:
        ImportError: If ``pymupdf4llm`` is not installed.
    """
    try:
        pymupdf._get_layout = None  # type: ignore[attr-defined]
        pymupdf._warn_layout_once = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        from pymupdf4llm.helpers.pymupdf_rag import to_markdown
    except ImportError as exc:
        raise ImportError(
            "pymupdf4llm is required for Markdown PDF extraction. "
            "Install it with `uv add pymupdf4llm`."
        ) from exc
    return to_markdown


def _extract_page_number(page_chunk: dict, fallback: int) -> int:
    """Return a 1-based page number for a pymupdf4llm page chunk.

    Looks at every plausible key name (``page``, ``page_number``,
    ``page_num``, ``page_no``, ``page_index``, ``number``) in both the chunk
    and its ``metadata`` sub-dict to cope with version skew across pymupdf4llm
    releases. Treats ``0`` as a 1-indexed offset (returns 1), passes any other
    positive integer through unchanged, and otherwise falls back to
    ``fallback``.

    Args:
        page_chunk: One element returned by ``to_markdown(..., page_chunks=True)``.
        fallback: 1-based page number to use when no candidate field contains
            a usable integer.

    Returns:
        A 1-based page number.
    """
    metadata = page_chunk.get("metadata") if isinstance(page_chunk, dict) else None
    candidates: list[object] = []

    if isinstance(metadata, dict):
        candidates.extend(
            [
                metadata.get("page"),
                metadata.get("page_number"),
                metadata.get("page_num"),
                metadata.get("page_no"),
                metadata.get("page_index"),
            ]
        )

    if isinstance(page_chunk, dict):
        candidates.extend(
            [
                page_chunk.get("page"),
                page_chunk.get("page_number"),
                page_chunk.get("page_num"),
                page_chunk.get("page_no"),
                page_chunk.get("number"),
            ]
        )

    for candidate in candidates:
        if isinstance(candidate, int) and candidate >= 0:
            return candidate + 1 if candidate == 0 else candidate

    return fallback


def _to_pages(page_chunks: object) -> list[tuple[int, str]]:
    """Normalize ``to_markdown`` output to ``(page_number, markdown)`` pairs.

    Args:
        page_chunks: Whatever ``to_markdown`` returned: a plain string when
            ``page_chunks`` was not honoured, a list of chunk dicts (current
            pymupdf4llm), or a list of strings (very old releases).

    Returns:
        ``(page_number, markdown)`` tuples in document order; empty when
        nothing was returned.
    """
    if isinstance(page_chunks, str):
        return [(1, page_chunks)]
    pages: list[tuple[int, str]] = []
    for index, chunk in enumerate(page_chunks, start=1):  # type: ignore[arg-type]
        if isinstance(chunk, dict):
            page_number = _extract_page_number(chunk, index)
            markdown = chunk.get("text", "") or chunk.get("markdown", "")
        else:
            page_number = index
            markdown = str(chunk or "")
        pages.append((page_number, markdown))
    return pages


def _largest_font_line(
    document: pymupdf.Document, max_pages: int = TITLE_FALLBACK_PAGES
) -> str | None:
    """Return the text line set in the largest font on the first pages.

    Used as the document title when the Markdown has no H1: manuals whose
    headings are typeset at body size never get a ``#`` heading from
    pymupdf4llm, but their cover still shows the product name in the biggest
    type on the page. Ties keep the first line encountered; lines with fewer
    than three letters (page numbers, bullets) are ignored. The PDF metadata
    title is deliberately not used; in practice it holds values like
    ``823245pt.indd`` or ``My Document``.

    Args:
        document: The open PyMuPDF document.
        max_pages: Number of leading pages scanned for the line.

    Returns:
        The largest-font line of the first page that has one, or None when
        those pages carry no text.
    """
    for page_index in range(min(max_pages, document.page_count)):
        best_size = 0.0
        best_text: str | None = None
        try:
            blocks = document[page_index].get_text("dict")["blocks"]
        except Exception:
            continue
        for block in blocks:
            for line in block.get("lines", []):
                spans = [span for span in line.get("spans", []) if span["text"].strip()]
                if not spans:
                    continue
                text = re.sub(r"\s+", " ", " ".join(span["text"].strip() for span in spans))
                if len(re.sub(r"[^A-Za-zÀ-ÿ]", "", text)) < 3:
                    continue
                size = max(float(span["size"]) for span in spans)
                if size > best_size:
                    best_size, best_text = size, text
        if best_text:
            return best_text
    return None


def _pages_from_file(path: Path) -> list[tuple[int, str]]:
    """Run pymupdf4llm over one PDF file and normalize to (page, markdown)."""
    to_markdown = _load_to_markdown()
    document = pymupdf.open(path)
    try:
        page_chunks = to_markdown(document, page_chunks=True)
    finally:
        document.close()
    return _to_pages(page_chunks)


def _promote_numbered_headings(
    markdown_pages: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    """Turn numbered ALL-CAPS section titles into Markdown headings when
    pymupdf4llm found no headings anywhere in the document.

    Gated on the whole document: if any page already has a ``#`` heading the
    font-size detection worked and the text is returned unchanged.
    ``1. TITLE`` becomes ``## 1. TITLE``, ``1.2 TITLE`` becomes
    ``### 1.2 TITLE`` (one level per dotted component). Bold wrappers around
    the line are tolerated.

    Args:
        markdown_pages: ``(page_number, markdown)`` pairs.

    Returns:
        The same pairs, with promoted heading lines when no page had a
        Markdown heading before.
    """
    if any(_MARKDOWN_HEADING.search(markdown or "") for _, markdown in markdown_pages):
        return markdown_pages

    promoted: list[tuple[int, str]] = []
    for page_number, markdown in markdown_pages:
        lines = []
        for line in (markdown or "").splitlines():
            core = line.strip().strip("*").strip()
            match = _NUMBERED_CAPS_HEADING.match(core)
            if match:
                depth = match.group("num").rstrip(".").count(".")
                level = min(2 + depth, 6)
                lines.append(f"{'#' * level} {core}")
            else:
                lines.append(line)
        promoted.append((page_number, "\n".join(lines)))
    return promoted


def _parse_pdf_date(value: str | None) -> datetime | None:
    """Parse a PDF metadata date string such as ``D:20260917123000+01'00'``.

    Args:
        value: Raw ``modDate``/``creationDate`` value from the PDF metadata;
            may be None or empty.

    Returns:
        A timezone-aware UTC datetime, or None when the value does not match
        the PDF date syntax or has an invalid calendar or offset value.
    """
    if not value:
        return None
    match = _PDF_DATE_RE.fullmatch(value.strip())
    if match is None:
        return None
    try:
        parsed = datetime(
            int(match["year"]),
            int(match["month"] or 1),
            int(match["day"] or 1),
            int(match["hour"] or 0),
            int(match["minute"] or 0),
            int(match["second"] or 0),
        )
        sign = match["sign"]
        if sign is None or sign in ("Z", "z"):
            return parsed.replace(tzinfo=UTC)
        offset = timedelta(hours=int(match["off_hour"] or 0), minutes=int(match["off_minute"] or 0))
        local = parsed.replace(tzinfo=timezone(-offset if sign == "-" else offset))
    except ValueError:
        return None
    return local.astimezone(UTC)


class PdfExtractor:
    """Extractor for PDF files (``.pdf``) via pymupdf4llm."""

    name = "pymupdf4llm"
    version = "1"
    formats = ("pdf",)

    def extract(self, path: Path) -> IRDocument:
        """Extract a PDF into the common IR, one block list per page.

        Opens the document once and always closes it. When the whole document
        holds fewer than :data:`MIN_EXTRACTABLE_TEXT_CHARS` characters of text
        (an image-only scan) and OCR is enabled (``extract.pdf_ocr``) and
        available, the file is OCR'd to a temporary copy and extraction re-runs
        on it (warning ``ocr_applied``); a failing OCR pass yields an empty IR
        with ``ocr_failed: <message>``; OCR disabled or unavailable keeps the
        previous ``needs_ocr`` behaviour.

        Args:
            path: The PDF file to read.

        Returns:
            The IR document with per-page blocks, links, resolved title
            (heading, else largest-font line, else humanized file name),
            document date (file name, else PDF metadata, else mtime) and
            language.
        """
        to_markdown = _load_to_markdown()
        document = pymupdf.open(path)
        try:
            largest_font_line = _largest_font_line(document)
            metadata_date = _parse_pdf_date(
                document.metadata.get("modDate") or document.metadata.get("creationDate")
            )
            page_chunks = to_markdown(document, page_chunks=True)
        finally:
            document.close()

        pages = _to_pages(page_chunks)
        if not pages:
            logger.warning("No Markdown pages returned for %s", path)

        warnings: list[str] = []
        blocks: list[IRBlock] = []
        links: list[Link] = []
        extracted_chars = sum(len((markdown or "").strip()) for _, markdown in pages)
        skip_blocks = False
        if extracted_chars < MIN_EXTRACTABLE_TEXT_CHARS:
            ocr_result = self._ocr_pages(path)
            if ocr_result is None:
                logger.warning(
                    "%s has only %d characters of extractable text across %d page(s); "
                    "likely an image-only or scanned PDF",
                    path.name,
                    extracted_chars,
                    len(pages),
                )
                warnings.append("needs_ocr")
                skip_blocks = True
            else:
                pages, ocr_warning = ocr_result
                warnings.append(ocr_warning)
                skip_blocks = ocr_warning != "ocr_applied"
        if not skip_blocks:
            for page_number, markdown in _promote_numbered_headings(pages):
                page_blocks, page_links = markdown_to_blocks(markdown, page=page_number)
                blocks.extend(page_blocks)
                links.extend(page_links)

        result = IRDocument(
            blocks=blocks,
            frontmatter={},
            links=links,
            hashtags=[],
            extractor=self.name,
            extractor_version=self.version,
            warnings=warnings,
        )
        result.language = detect_language(result.plain_text()[:5000])
        result.title = resolve_title(
            frontmatter={},
            blocks=blocks,
            filename=path.name,
            largest_font_line=largest_font_line,
        )
        result.doc_date = resolve_doc_date(
            frontmatter={},
            filename=path.name,
            metadata_date=metadata_date,
            mtime_ns=path.stat().st_mtime_ns,
        )
        return result

    def _ocr_pages(self, path: Path) -> tuple[list[tuple[int, str]], str] | None:
        """OCR ``path`` and extract its pages, when enabled and available.

        Args:
            path: The original PDF file.

        Returns:
            ``(pages, warning)`` with ``warning="ocr_applied"`` on success or
            ``warning="ocr_failed: <message>"`` (with no pages) on failure;
            None when OCR is disabled in the settings or ocrmypdf/tesseract
            are not installed (the caller keeps the ``needs_ocr`` behaviour).
        """
        settings = get_settings()
        if not settings.extract.pdf_ocr or not ocr_available():
            return None
        try:
            ocr_path = ocr_pdf(path)
        except Exception as exc:
            logger.warning("OCR of %s failed: %s", path, exc)
            return ([], f"ocr_failed: {exc}")
        try:
            try:
                return (_pages_from_file(ocr_path), "ocr_applied")
            finally:
                shutil.rmtree(ocr_path.parent, ignore_errors=True)
        except Exception as exc:
            logger.warning("Extraction of the OCR'd %s failed: %s", path, exc)
            return ([], f"ocr_failed: {exc}")


register(PdfExtractor())
