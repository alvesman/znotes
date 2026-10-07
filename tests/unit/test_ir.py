"""Unit tests for the extraction IR and the extractor registry."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from mynotes.extract.ir import IRBlock, IRDocument, Link
from mynotes.extract.registry import get_extractor, register


def test_plain_text_joins_blocks_with_blank_lines() -> None:
    document = IRDocument(
        blocks=[
            IRBlock(kind="heading", text="Título", level=1),
            IRBlock(kind="paragraph", text="Primeiro parágrafo."),
            IRBlock(kind="paragraph", text="Segundo parágrafo."),
        ]
    )

    assert document.plain_text() == "Título\n\nPrimeiro parágrafo.\n\nSegundo parágrafo."
    assert IRDocument(blocks=[]).plain_text() == ""


def test_to_json_converts_dates_to_iso_strings() -> None:
    document = IRDocument(
        blocks=[IRBlock(kind="heading", text="Nota", level=1, page=3)],
        frontmatter={"date": date(2026, 3, 10), "when": datetime(2026, 3, 10, 12, 30)},
        doc_date=datetime(2026, 3, 10, 12, 0, tzinfo=UTC),
    )

    data = document.to_json()

    json.dumps(data)  # must not raise
    assert data["frontmatter"]["date"] == "2026-03-10"
    assert data["frontmatter"]["when"] == "2026-03-10T12:30:00"
    assert data["doc_date"] == "2026-03-10T12:00:00+00:00"
    assert data["blocks"] == [{"kind": "heading", "text": "Nota", "level": 1, "page": 3}]


def test_from_json_round_trips_documents() -> None:
    document = IRDocument(
        blocks=[
            IRBlock(kind="heading", text="Ideia", level=1),
            IRBlock(kind="paragraph", text="Texto."),
            IRBlock(kind="code", text="```\ncodigo\n```", page=2),
        ],
        title="Ideia",
        frontmatter={"title": "Ideia", "tags": ["alpha"], "date": date(2026, 3, 10)},
        links=[
            Link(target="Outra", kind="wikilink", text="outra"),
            Link(target="https://x.y", kind="url"),
        ],
        hashtags=["projeto-x"],
        language="pt",
        doc_date=datetime(2026, 3, 10, tzinfo=UTC),
        extractor="markdown",
        extractor_version="1",
        warnings=["invalid_utf8"],
    )

    assert IRDocument.from_json(document.to_json()) == document


def test_from_json_fills_defaults_for_missing_keys() -> None:
    document = IRDocument.from_json({"blocks": [{"kind": "paragraph", "text": "olá"}]})

    assert document.blocks == [IRBlock(kind="paragraph", text="olá")]
    assert document.title is None
    assert document.frontmatter == {}
    assert document.links == []
    assert document.doc_date is None
    assert document.extractor == ""
    assert document.warnings == []


def test_get_extractor_returns_registered_extractors() -> None:
    markdown = get_extractor("md")
    text = get_extractor("txt")

    assert markdown.name == "markdown"
    assert markdown.version == "1"
    assert "md" in markdown.formats
    assert text.name == "text"


def test_get_extractor_unknown_format_raises_key_error() -> None:
    with pytest.raises(KeyError, match="no extractor for format 'rtf'"):
        get_extractor("rtf")


def test_register_adds_extractor_for_all_formats() -> None:
    class FakeExtractor:
        name = "fake"
        version = "1"
        formats = ("fake1", "fake2")

        def extract(self, path: Path) -> IRDocument:  # pragma: no cover - trivial
            return IRDocument(blocks=[])

    fake = FakeExtractor()
    register(fake)

    assert get_extractor("fake1") is fake
    assert get_extractor("fake2") is fake
