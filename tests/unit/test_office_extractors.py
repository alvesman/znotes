"""Unit tests for the markitdown Office/HTML/EPUB extractor (task 36).

The fixtures are generated at test time into the git-ignored
``tests/fixtures/docs/`` directory (see ``tests/fixtures/make_office.py``);
no binary files are committed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.fixtures import make_office

from mynotes.extract.ir import IRDocument
from mynotes.extract.office import (
    OfficeExtractor,
    _clean_metadata_title,
    _is_pure_links,
    _split_slides,
)
from mynotes.extract.registry import get_extractor

DOCS = make_office.DOCS


@pytest.fixture(scope="module")
def office_docs() -> Path:
    """Generate every office fixture once per module."""
    DOCS.mkdir(parents=True, exist_ok=True)
    make_office.make_docx()
    make_office.make_pptx()
    make_office.make_xlsx()
    make_office.make_html()
    make_office.make_epub()
    return DOCS


@pytest.fixture(scope="module")
def docx(office_docs: Path) -> IRDocument:
    return OfficeExtractor().extract(office_docs / "sample.docx")


@pytest.fixture(scope="module")
def pptx(office_docs: Path) -> IRDocument:
    return OfficeExtractor().extract(office_docs / "sample.pptx")


@pytest.fixture(scope="module")
def xlsx(office_docs: Path) -> IRDocument:
    return OfficeExtractor().extract(office_docs / "sample.xlsx")


@pytest.fixture(scope="module")
def html(office_docs: Path) -> IRDocument:
    return OfficeExtractor().extract(office_docs / "sample.html")


@pytest.fixture(scope="module")
def epub(office_docs: Path) -> IRDocument:
    return OfficeExtractor().extract(office_docs / "sample.epub")


def test_docx_headings_table_and_title(docx: IRDocument) -> None:
    assert docx.title == "Proposta Alfa"
    headings = [(block.text, block.level) for block in docx.blocks if block.kind == "heading"]
    assert headings == [("Proposta Alfa", 1), ("Orçamento", 2)]
    tables = [block for block in docx.blocks if block.kind == "table"]
    assert len(tables) == 1
    assert "Item" in tables[0].text and "Filtro" in tables[0].text
    assert all(block.page is None for block in docx.blocks)
    assert docx.extractor == "markitdown"
    assert docx.extractor_version == "1"
    assert docx.language == "pt"


def test_docx_core_properties_date_from_zip(docx: IRDocument) -> None:
    assert docx.doc_date == datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


def test_docx_metadata_title_placeholder_rules() -> None:
    assert _clean_metadata_title("Proposta Alfa") == "Proposta Alfa"
    assert _clean_metadata_title(" 823245pt.indd ") is None
    assert _clean_metadata_title("proposal_v2.docx") is None
    for placeholder in ("Document", "DOCUMENTO", "Presentation", "Apresentação"):
        assert _clean_metadata_title(placeholder) is None
    assert _clean_metadata_title("") is None
    assert _clean_metadata_title(None) is None


def test_pptx_slides_notes_and_title(pptx: IRDocument) -> None:
    assert pptx.title == "Visão"
    assert {block.page for block in pptx.blocks} == {1, 2, 3}
    slide_titles = [
        block.text
        for block in pptx.blocks
        if block.kind == "heading" and block.page in (1, 2, 3) and block.text != "Notes:"
    ]
    assert slide_titles == ["Visão", "Plano", "Riscos"]
    notes = [
        block
        for block in pptx.blocks
        if block.page == 2 and block.kind == "paragraph" and "orçamento" in block.text
    ]
    assert notes, "speaker notes must stay as blocks of slide 2"


def test_pptx_split_slides_uses_markers() -> None:
    markdown = (
        "<!-- Slide number: 1 -->\n# A\ntext a\n\n"
        "<!-- Slide number: 2 -->\n# B\n### Notes:\nnote b\n"
    )
    parts = _split_slides(markdown)
    assert [page for page, _ in parts] == [1, 2]
    assert "text a" in parts[0][1]
    assert "note b" in parts[1][1]


def test_pptx_split_slides_keeps_preamble_unpaged() -> None:
    parts = _split_slides("loose text\n\n<!-- Slide number: 1 -->\n# A\n")
    assert parts[0][0] is None
    assert "loose text" in parts[0][1]
    assert parts[1][0] == 1


def test_xlsx_sheet_headings_and_tables(xlsx: IRDocument) -> None:
    headings = [block for block in xlsx.blocks if block.kind == "heading"]
    assert [(block.text, block.level) for block in headings] == [("Custos", 2), ("Prazos", 2)]
    tables = [block for block in xlsx.blocks if block.kind == "table"]
    assert len(tables) == 2
    assert "Item" in tables[0].text and "Fase" in tables[1].text
    assert all(block.page is None for block in xlsx.blocks)


def test_html_strips_script_and_nav_but_keeps_heading(html: IRDocument) -> None:
    joined = "\n".join(block.text for block in html.blocks)
    assert "x()" not in joined
    assert any(block.kind == "heading" and block.text == "Página" for block in html.blocks)
    assert any(block.kind == "paragraph" and "Olá mundo" in block.text for block in html.blocks)
    assert not any("[Início]" in block.text or "Alpha" in block.text for block in html.blocks)


def test_is_pure_links_matches_only_link_spans() -> None:
    assert _is_pure_links("[Início](/inicio) [Sobre](/sobre)")
    assert _is_pure_links("* [Alpha](/a)\n* [Beta](/b)\n")
    assert _is_pure_links("[](https://x)[](https://y)")
    assert not _is_pure_links("Veja [aqui](https://x) os detalhes")
    assert not _is_pure_links("Parágrafo comum sem links")
    assert not _is_pure_links("| a | b |")


def test_epub_chapters_become_headings(epub: IRDocument) -> None:
    headings = [block.text for block in epub.blocks if block.kind == "heading"]
    assert "Capítulo Um" in headings
    assert "Capítulo Dois" in headings
    assert any(
        block.kind == "paragraph" and "primeiro capítulo" in block.text for block in epub.blocks
    )


def test_legacy_doc_not_registered() -> None:
    with pytest.raises(KeyError):
        get_extractor("unsupported")
    with pytest.raises(KeyError):
        get_extractor("doc")


def test_registry_serves_office_formats() -> None:
    extractor = get_extractor("docx")
    assert extractor.name == "markitdown"
    assert extractor.version == "1"
    assert extractor.formats == ("docx", "pptx", "xlsx", "html", "epub")
    for fmt in ("pptx", "xlsx", "html", "epub"):
        assert get_extractor(fmt) is extractor


def test_resolve_title_metadata_title_order() -> None:
    from mynotes.extract.ir import IRBlock
    from mynotes.extract.titles import resolve_title

    assert (
        resolve_title(frontmatter={}, blocks=[], filename="a.docx", metadata_title="Meta Título")
        == "Meta Título"
    )
    assert (
        resolve_title(
            frontmatter={},
            blocks=[IRBlock(kind="heading", text="Primeiro", level=1)],
            filename="a.docx",
            first_slide_title="Slide",
            metadata_title="Meta",
        )
        == "Primeiro"
    )
    assert (
        resolve_title(frontmatter={}, blocks=[], filename="a.docx", first_slide_title="Slide")
        == "Slide"
    )
    assert resolve_title(frontmatter={}, blocks=[], filename="meu_arquivo.docx") == "Meu arquivo"
