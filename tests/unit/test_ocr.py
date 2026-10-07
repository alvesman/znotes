"""Unit tests for the OCR fallback (task 36): availability, ocrmypdf run and
the PDF extractor wiring (ocr_applied / ocr_failed / needs_ocr).

The real-OCR test is skipped when ocrmypdf or tesseract is not installed. The
PDF wiring is exercised with a fake ``ocr_pdf`` so it runs everywhere.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pymupdf
import pytest
from tests.fixtures.make_office import make_image_only_pdf

from mynotes.config import Settings
from mynotes.extract.ocr import ocr_available, ocr_pdf
from mynotes.extract.pdf import PdfExtractor

DOCS = Path(__file__).resolve().parents[1] / "fixtures" / "docs"


def test_ocr_available_returns_bool() -> None:
    assert isinstance(ocr_available(), bool)


def test_ocr_pdf_without_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mynotes.extract.ocr.shutil.which", lambda name: None)
    assert ocr_available() is False
    with pytest.raises(FileNotFoundError):
        ocr_pdf(Path("anything.pdf"))


def test_ocr_pdf_requires_both_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mynotes.extract.ocr.shutil.which",
        lambda name: "tesseract" if name == "tesseract" else None,
    )
    assert ocr_available() is False


@pytest.mark.skipif(not ocr_available(), reason="ocrmypdf/tesseract not installed")
def test_ocr_pdf_recognizes_words(tmp_path: Path) -> None:
    source = make_image_only_pdf(tmp_path / "image_only.pdf")
    output = ocr_pdf(source, timeout_s=300)
    try:
        assert output.exists() and output != source
        document = pymupdf.open(output)
        try:
            text = document[0].get_text()
        finally:
            document.close()
        assert "Filtro" in text or "exaustor" in text
    finally:
        shutil.rmtree(output.parent, ignore_errors=True)


def _settings_with_ocr(enabled: bool) -> Settings:
    return Settings(
        extract={"pdf_ocr": enabled},
        models={
            "embedding": {"dim": 8, "base_url": "http://embed.test/v1"},
            "reranker": {"url": "http://rerank.test/v1/rerank"},
            "llm": {"base_url": "http://llm.test/v1"},
        },
    )


def _text_pdf(path: Path) -> Path:
    document = pymupdf.open()
    try:
        page = document.new_page()
        page.insert_text((72, 100), "Conteudo real do pdf em portugues.", fontsize=14)
        document.save(path)
    finally:
        document.close()
    return path


@pytest.fixture
def image_only(tmp_path: Path) -> Path:
    return make_image_only_pdf(tmp_path / "image_only.pdf")


def test_extract_applies_ocr_when_enabled(
    image_only: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ocr_source = _text_pdf(tmp_path / "ocr_output.pdf")

    def fake_ocr_pdf(path: Path, languages: str = "por+eng", timeout_s: int = 600) -> Path:
        staged = tmp_path / "mynotes-ocr-fake"
        staged.mkdir(exist_ok=True)
        return shutil.copy(ocr_source, staged / "ocr_output.pdf")

    monkeypatch.setattr("mynotes.extract.pdf.get_settings", lambda: _settings_with_ocr(True))
    monkeypatch.setattr("mynotes.extract.pdf.ocr_available", lambda: True)
    monkeypatch.setattr("mynotes.extract.pdf.ocr_pdf", fake_ocr_pdf)

    result = PdfExtractor().extract(image_only)
    assert result.warnings == ["ocr_applied"]
    assert result.blocks, "OCR'd content must produce blocks"
    assert any("Conteudo real" in block.text for block in result.blocks)
    assert not list(tmp_path.glob("mynotes-ocr-fake/*")), "temp copy must be deleted"


def test_extract_ocr_failure_yields_empty_ir(
    image_only: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_ocr_pdf(path: Path, languages: str = "por+eng", timeout_s: int = 600) -> Path:
        raise RuntimeError("boom")

    monkeypatch.setattr("mynotes.extract.pdf.get_settings", lambda: _settings_with_ocr(True))
    monkeypatch.setattr("mynotes.extract.pdf.ocr_available", lambda: True)
    monkeypatch.setattr("mynotes.extract.pdf.ocr_pdf", failing_ocr_pdf)

    result = PdfExtractor().extract(image_only)
    assert result.blocks == []
    assert result.warnings == ["ocr_failed: boom"]


def test_extract_keeps_needs_ocr_when_ocr_disabled(image_only: Path) -> None:
    result = PdfExtractor().extract(image_only)
    assert result.blocks == []
    assert result.warnings == ["needs_ocr"]


def test_extract_keeps_needs_ocr_when_tools_missing(
    image_only: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("mynotes.extract.pdf.get_settings", lambda: _settings_with_ocr(True))
    monkeypatch.setattr("mynotes.extract.pdf.ocr_available", lambda: False)
    result = PdfExtractor().extract(image_only)
    assert result.warnings == ["needs_ocr"]
    assert result.blocks == []
