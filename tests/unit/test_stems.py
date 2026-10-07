"""Unit tests for file-name stem parsing (task 34)."""

from __future__ import annotations

from datetime import UTC, datetime

from mynotes.versioning.stems import StemInfo, parse_stem


def info(stem: str, version: int | None, date: datetime | None, ext: str) -> StemInfo:
    """Build the expected StemInfo for one table row."""
    return StemInfo(stem=stem, version_number=version, date=date, ext=ext)


def test_spec_table() -> None:
    """The examples from the task specification parse exactly as listed."""
    assert parse_stem("Proposta_v2.docx") == info("proposta", 2, None, "docx")
    assert parse_stem("proposta final.docx") == info("proposta", None, None, "docx")
    assert parse_stem("spec-2026-09.md") == info(
        "spec", None, datetime(2026, 9, 1, tzinfo=UTC), "md"
    )
    assert parse_stem("plano (1).pdf") == info("plano", 1, None, "pdf")
    assert parse_stem("Plano Alfa rev3.pptx") == info("plano alfa", 3, None, "pptx")
    assert parse_stem("notes.md") == info("notes", None, None, "md")


def test_version_markers() -> None:
    """v/ver/rev/r markers, versao/version words and (n) are captured."""
    assert parse_stem("idea_v2.md").version_number == 2
    assert parse_stem("Relatório_v02.pdf").version_number == 2
    assert parse_stem("relatorio_v2_rev3.md").version_number == 3
    assert parse_stem("spec rev3.md").version_number == 3
    assert parse_stem("spec r7.md").version_number == 7
    assert parse_stem("versao 3.docx").version_number == 3
    assert parse_stem("Version 3.docx").version_number == 3
    assert parse_stem("versao3.docx").version_number == 3
    assert parse_stem("plano (1).pdf").version_number == 1
    assert parse_stem("plano (12) (3).pdf").version_number == 12
    assert parse_stem("artigo(v4).docx").version_number == 4


def test_four_digit_numbers_after_v_are_not_versions() -> None:
    """``v2026`` is a year, not a version number."""
    assert parse_stem("relatorio_v2026.docx").version_number is None
    assert parse_stem("relatorio r2026.md").version_number is None
    assert parse_stem("versao 2026.docx").version_number == 2026


def test_noise_words_are_removed() -> None:
    """The noise words disappear as whole tokens, not inside words."""
    for word in (
        "final",
        "draft",
        "rascunho",
        "copy",
        "copia",
        "old",
        "antigo",
        "novo",
        "new",
        "latest",
        "wip",
    ):
        assert parse_stem(f"relatorio {word}.md").stem == "relatorio", word
    assert parse_stem("goldfish.md").stem == "goldfish"
    assert parse_stem("copia antigo do relatorio.md").stem == "do relatorio"


def test_dates_are_captured_and_removed() -> None:
    """Every filename date pattern is stripped from the stem."""
    parsed = parse_stem("plan-2026-09-17.md")
    assert parsed.stem == "plan"
    assert parsed.date == datetime(2026, 9, 17, tzinfo=UTC)
    assert parse_stem("20260917.txt").stem == ""
    assert parse_stem("relatorio_2026_09.docx").stem == "relatorio"
    assert parse_stem("2026-13-99.md").date is None


def test_case_accents_and_separators_are_folded() -> None:
    """The stem is a normalize_key key with separators as single spaces."""
    assert parse_stem("Conclusão_Final.DOCX").stem == "conclusao"
    assert parse_stem("  Ação--Técnica _v2.pdf").stem == "acao tecnica"
    assert parse_stem("ação técnica.docx").stem == parse_stem("ACAO TECNICA.docx").stem


def test_paths_and_extensions() -> None:
    """Only the file name counts; the extension is lowercase without the dot."""
    parsed = parse_stem("folder/sub/Ideia_v3.MD")
    assert parsed.stem == "ideia"
    assert parsed.version_number == 3
    assert parsed.ext == "md"
    assert parse_stem("noext").ext == ""


def test_empty_and_degenerate_names() -> None:
    """A name made only of markers yields an empty stem."""
    assert parse_stem("2026-09-17.md").stem == ""
    assert parse_stem("versao 3.docx").stem == ""
    assert parse_stem("notes.md").stem == "notes"
