"""Unit tests for the Markdown and plain-text extractors."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from mynotes.extract.markdown import MarkdownExtractor, TextExtractor

DOCS = Path(__file__).resolve().parents[1] / "fixtures" / "docs"


@pytest.fixture(scope="module")
def note_pt() -> object:
    return MarkdownExtractor().extract(DOCS / "note_pt.md")


@pytest.fixture(scope="module")
def note_en() -> object:
    return MarkdownExtractor().extract(DOCS / "note_en.md")


@pytest.fixture(scope="module")
def plain() -> object:
    return TextExtractor().extract(DOCS / "plain.txt")


def test_note_pt_block_kinds_and_levels_in_order(note_pt) -> None:  # type: ignore[no-untyped-def]
    kinds = [block.kind for block in note_pt.blocks]
    levels = [block.level for block in note_pt.blocks if block.kind == "heading"]

    assert kinds == [
        "heading",
        "paragraph",
        "heading",
        "list",
        "heading",
        "table",
        "code",
        "paragraph",
    ]
    assert levels == [1, 2, 2]
    table = next(block for block in note_pt.blocks if block.kind == "table")
    assert "|" in table.text
    heading = note_pt.blocks[0]
    assert heading.text == "Ideia Alfa"


def test_note_pt_frontmatter_title_and_date(note_pt) -> None:  # type: ignore[no-untyped-def]
    assert note_pt.title == "Ideia Alfa"
    assert note_pt.frontmatter["tags"] == ["alpha"]
    assert note_pt.doc_date == datetime(2026, 3, 10, tzinfo=UTC)
    assert note_pt.extractor == "markdown"
    assert note_pt.extractor_version == "1"
    assert note_pt.warnings == []


def test_note_pt_hashtags_and_links(note_pt) -> None:  # type: ignore[no-untyped-def]
    assert note_pt.hashtags == ["projeto-x"]

    targets = {(link.target, link.kind, link.text) for link in note_pt.links}
    assert ("Plano Beta", "wikilink", "plano") in targets
    assert ("https://example.com", "url", "exemplo") in targets


def test_note_pt_language(note_pt) -> None:  # type: ignore[no-untyped-def]
    assert note_pt.language == "pt"


def test_note_en_language_and_h1_title(note_en) -> None:  # type: ignore[no-untyped-def]
    assert note_en.language == "en"
    assert note_en.title == "Market research"
    assert note_en.frontmatter == {}
    headings = [block for block in note_en.blocks if block.kind == "heading"]
    assert [block.level for block in headings] == [1, 2]


def test_plain_txt_paragraphs(plain) -> None:  # type: ignore[no-untyped-def]
    assert [block.kind for block in plain.blocks] == ["paragraph", "paragraph"]
    assert plain.blocks[0].text.startswith("Primeiro parágrafo")
    assert plain.blocks[1].text.startswith("Segundo parágrafo")
    assert plain.title == "Primeiro parágrafo do ficheiro de texto simples com algumas frases."
    assert plain.extractor == "text"


def test_markdown_ir_round_trip(note_pt) -> None:  # type: ignore[no-untyped-def]
    from mynotes.extract.ir import IRDocument

    assert IRDocument.from_json(note_pt.to_json()) == note_pt


def test_invalid_utf8_adds_warning(tmp_path: Path) -> None:
    path = tmp_path / "quebrado.md"
    path.write_bytes("texto com acentuaç".encode() + b"\xff\xfe")

    document = MarkdownExtractor().extract(path)

    assert document.warnings == ["invalid_utf8"]
    assert "acentuaç" in document.plain_text()


def test_wikilink_without_text_and_hashtag_in_list(tmp_path: Path) -> None:
    path = tmp_path / "nota.md"
    path.write_text(
        "# Nota\n\n"
        "- item com [[Alvo Simples]]\n"
        "- marcado com #Etiqueta-1\n\n"
        "Referência #1 e #2026 não são tags; `#codigo` também não.\n",
        encoding="utf-8",
    )

    document = MarkdownExtractor().extract(path)

    assert [(link.target, link.text) for link in document.links if link.kind == "wikilink"] == [
        ("Alvo Simples", None)
    ]
    assert document.hashtags == ["Etiqueta-1"]
