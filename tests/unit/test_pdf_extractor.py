"""Unit tests for the PDF extractor (pymupdf4llm).

The fixture PDFs are generated at test time with PyMuPDF into
``tests/fixtures/docs/`` (git-ignored), so no binary files are committed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pymupdf
import pytest

from mynotes.extract.ir import IRBlock, IRDocument
from mynotes.extract.markdown import markdown_to_blocks
from mynotes.extract.pdf import (
    PdfExtractor,
    _extract_page_number,
    _largest_font_line,
    _parse_pdf_date,
    _promote_numbered_headings,
)
from mynotes.extract.registry import get_extractor

DOCS = Path(__file__).resolve().parents[1] / "fixtures" / "docs"


def _write_pdf(
    path: Path,
    pages: list[list[tuple[str, float]]],
    metadata: dict[str, str] | None = None,
) -> None:
    """Write a small PDF with one ``insert_text`` line per tuple (text, size)."""
    document = pymupdf.open()
    try:
        if metadata:
            document.set_metadata(metadata)
        for lines in pages:
            page = document.new_page()
            y = 100.0
            for text, size in lines:
                page.insert_text((72, y), text, fontsize=size, fontname="helv")
                y += size + 38.0
        document.save(path)
    finally:
        document.close()


@pytest.fixture(scope="session")
def pdf_docs() -> Path:
    """Generate the fixture PDFs once per test session."""
    DOCS.mkdir(parents=True, exist_ok=True)
    _write_pdf(
        DOCS / "titled.pdf",
        [
            [
                ("Manual AquaBreeze", 24.0),
                ("Guarde este manual para consulta futura.", 11.0),
                ("O aparelho destina-se exclusivamente a uso doméstico.", 11.0),
            ],
            [
                ("Características do produto.", 11.0),
                ("Limpe o aparelho antes da primeira utilização.", 11.0),
            ],
        ],
        metadata={"modDate": "D:20260917123000+01'00'"},
    )
    _write_pdf(
        DOCS / "numbered.pdf",
        [
            [
                ("1. PIEZAS Y COMPONENTES", 11.0),
                ("En este manual se describen las piezas principales del aparato.", 11.0),
                ("2.3 MONTAGEM", 11.0),
                ("Siga os passos indicados na sequencia apresentada.", 11.0),
            ]
        ],
    )
    _write_pdf(DOCS / "empty.pdf", [[]])
    return DOCS


@pytest.fixture(scope="module")
def titled(pdf_docs: Path) -> IRDocument:
    return PdfExtractor().extract(pdf_docs / "titled.pdf")


@pytest.fixture(scope="module")
def numbered(pdf_docs: Path) -> IRDocument:
    return PdfExtractor().extract(pdf_docs / "numbered.pdf")


@pytest.fixture(scope="module")
def empty(pdf_docs: Path) -> IRDocument:
    return PdfExtractor().extract(pdf_docs / "empty.pdf")


def test_titled_pdf_title_pages_language_and_date(titled: IRDocument) -> None:
    assert "AquaBreeze" in (titled.title or "")
    assert {block.page for block in titled.blocks} == {1, 2}
    page_two = [block for block in titled.blocks if block.page == 2]
    assert [block.text for block in page_two] == [
        "Características do produto.",
        "Limpe o aparelho antes da primeira utilização.",
    ]
    assert titled.language == "pt"
    assert titled.doc_date == datetime(2026, 9, 17, 11, 30, tzinfo=UTC)
    assert titled.extractor == "pymupdf4llm"
    assert titled.extractor_version == "1"
    assert titled.frontmatter == {}
    assert titled.hashtags == []
    assert titled.warnings == []


def test_numbered_pdf_promotes_caps_headings(numbered: IRDocument) -> None:
    headings = [block for block in numbered.blocks if block.kind == "heading"]
    assert [(block.text, block.level, block.page) for block in headings] == [
        ("1. PIEZAS Y COMPONENTES", 2, 1),
        ("2.3 MONTAGEM", 3, 1),
    ]
    assert numbered.title == "1. PIEZAS Y COMPONENTES"


def test_empty_pdf_needs_ocr(empty: IRDocument) -> None:
    assert empty.blocks == []
    assert "needs_ocr" in empty.warnings
    assert empty.language is None
    assert empty.title == "Empty"


def test_document_is_closed_after_extraction(
    pdf_docs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[pymupdf.Document] = []
    closed: list[pymupdf.Document] = []
    real_open = pymupdf.open
    real_close = pymupdf.Document.close

    def tracking_open(*args: object, **kwargs: object) -> pymupdf.Document:
        document = real_open(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(document)
        return document

    def counting_close(document: pymupdf.Document) -> None:
        closed.append(document)
        real_close(document)

    monkeypatch.setattr(pymupdf, "open", tracking_open)
    monkeypatch.setattr(pymupdf.Document, "close", counting_close)

    PdfExtractor().extract(pdf_docs / "titled.pdf")

    assert len(opened) == 1
    assert closed == opened
    assert opened[0].is_closed


def test_registry_serves_pdf_format() -> None:
    extractor = get_extractor("pdf")
    assert extractor.name == "pymupdf4llm"
    assert extractor.version == "1"
    assert extractor.formats == ("pdf",)


def test_largest_font_line_ignores_short_and_small_lines(pdf_docs: Path) -> None:
    document = pymupdf.open(pdf_docs / "titled.pdf")
    try:
        assert _largest_font_line(document) == "Manual AquaBreeze"
    finally:
        document.close()


def test_promote_numbered_headings_positive_cases() -> None:
    pages = [(1, "**2.3 MONTAGEM**\n\n1. PIEZAS Y COMPONENTES\n\n1.2.3 INSTALACAO\n")]
    promoted = _promote_numbered_headings(pages)
    assert (
        promoted[0][1] == "### 2.3 MONTAGEM\n\n## 1. PIEZAS Y COMPONENTES\n\n#### 1.2.3 INSTALACAO"
    )


def test_promote_numbered_headings_ignores_running_footer() -> None:
    pages = [(1, "4 BAMBA IONICARE RADIANCE 5\n\nCorpo do texto do manual.\n")]
    promoted = _promote_numbered_headings(pages)
    assert promoted[0][1] == "4 BAMBA IONICARE RADIANCE 5\n\nCorpo do texto do manual."
    assert not any(line.startswith("#") for line in promoted[0][1].splitlines())


def test_promote_numbered_headings_gated_by_existing_heading() -> None:
    pages = [(1, "# Real Heading\n\n1. PIEZAS Y COMPONENTES\n")]
    assert _promote_numbered_headings(pages) == pages


def test_extract_page_number_candidates_and_fallback() -> None:
    assert _extract_page_number({"metadata": {"page": 3}, "text": "x"}, 9) == 3
    assert _extract_page_number({"page": 0, "text": "x"}, 9) == 1
    assert _extract_page_number({"page_number": 5}, 9) == 5
    assert _extract_page_number({"metadata": {}, "text": "x"}, 7) == 7


def test_parse_pdf_date_with_offset_utc_and_bare_date() -> None:
    assert _parse_pdf_date("D:20260917123000+01'00'") == datetime(2026, 9, 17, 11, 30, tzinfo=UTC)
    assert _parse_pdf_date("D:20260917123000-02'00'") == datetime(2026, 9, 17, 14, 30, tzinfo=UTC)
    assert _parse_pdf_date("D:20260917123000Z") == datetime(2026, 9, 17, 12, 30, tzinfo=UTC)
    assert _parse_pdf_date("20260917") == datetime(2026, 9, 17, tzinfo=UTC)


def test_parse_pdf_date_ignores_unparsable_values() -> None:
    assert _parse_pdf_date("D:2026134000000") is None
    assert _parse_pdf_date("823245pt.indd") is None
    assert _parse_pdf_date("") is None
    assert _parse_pdf_date(None) is None


def test_markdown_to_blocks_stamps_page_and_collects_links() -> None:
    blocks, links = markdown_to_blocks("# Título\n\nParágrafo com [[Alvo]].\n", page=3)

    assert blocks[0] == IRBlock(kind="heading", text="Título", level=1, page=3)
    assert blocks[1].kind == "paragraph"
    assert blocks[1].page == 3
    assert [(link.target, link.kind) for link in links] == [("Alvo", "wikilink")]


def test_markdown_to_blocks_without_page_keeps_page_none() -> None:
    blocks, _ = markdown_to_blocks("# X\n\nTexto.\n")
    assert all(block.page is None for block in blocks)
