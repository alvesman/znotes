"""Unit tests for title and document-date resolution."""

from __future__ import annotations

from datetime import UTC, date, datetime

from mynotes.extract.dates import date_from_filename, resolve_doc_date
from mynotes.extract.ir import IRBlock
from mynotes.extract.titles import humanize_filename, resolve_title


def block(kind: str, text: str, level: int | None = None) -> IRBlock:
    return IRBlock(kind=kind, text=text, level=level)


def test_frontmatter_title_wins_over_everything() -> None:
    title = resolve_title(
        frontmatter={"title": "Do frontmatter"},
        blocks=[block("heading", "H1 grande", 1)],
        filename="outro-nome.md",
        largest_font_line="Fonte maior",
        first_slide_title="Slide um",
    )

    assert title == "Do frontmatter"


def test_first_level1_heading_used_without_frontmatter_title() -> None:
    title = resolve_title(
        frontmatter={},
        blocks=[block("heading", "Sumário", 2), block("heading", "Ideia Principal", 1)],
        filename="outro-nome.md",
    )

    assert title == "Ideia Principal"


def test_too_short_h1_is_skipped() -> None:
    title = resolve_title(
        frontmatter={},
        blocks=[block("heading", "**i**", 1)],
        filename="outro-nome.md",
        largest_font_line="Linha do PDF",
    )

    assert title == "Linha do PDF"


def test_short_h1_after_stripping_emphasis_is_skipped() -> None:
    title = resolve_title(
        frontmatter={},
        blocks=[block("heading", "__ab__", 1)],
        filename="outro-nome.md",
        first_slide_title="Primeiro slide",
    )

    assert title == "Primeiro slide"


def test_filename_fallback_is_humanized() -> None:
    title = resolve_title(frontmatter={}, blocks=[], filename="my_idea-v2.md")

    assert title == "My idea v2"


def test_humanize_filename() -> None:
    assert humanize_filename("my_idea-v2.md") == "My idea v2"
    assert humanize_filename("plano_final.docx") == "Plano final"
    assert humanize_filename("PDF notes.txt") == "PDF notes"


def test_date_from_filename_patterns() -> None:
    assert date_from_filename("plan-2026-09-17.pdf") == datetime(2026, 9, 17, tzinfo=UTC)
    assert date_from_filename("relatorio_2026_09_17.docx") == datetime(2026, 9, 17, tzinfo=UTC)
    assert date_from_filename("notas.2026.09.17.md") == datetime(2026, 9, 17, tzinfo=UTC)
    assert date_from_filename("backup20260917.txt") == datetime(2026, 9, 17, tzinfo=UTC)
    assert date_from_filename("spec-2026-09.md") == datetime(2026, 9, 1, tzinfo=UTC)
    assert date_from_filename("resumo_2026-09.pdf") == datetime(2026, 9, 1, tzinfo=UTC)


def test_date_from_filename_rejects_invalid_candidates() -> None:
    assert date_from_filename("report_202609.pdf") is None
    assert date_from_filename("x-20261340.md") is None
    assert date_from_filename("nada.pdf") is None
    assert date_from_filename("velho-1989-12-31.md") is None
    assert date_from_filename("futuro-2101-01-01.md") is None
    assert date_from_filename("fevereiro-2026-02-30.md") is None


def test_date_from_filename_returns_first_valid_match() -> None:
    assert date_from_filename("2026-02-30-depois-2026-03-15.md") == datetime(
        2026, 3, 15, tzinfo=UTC
    )


def test_resolve_doc_date_prefers_updated_then_date_then_created() -> None:
    resolved = resolve_doc_date(
        frontmatter={
            "updated": "2026-05-02T10:00:00+02:00",
            "date": "2026-04-01",
            "created": date(2026, 3, 1),
        },
        filename="plan-2026-09-17.pdf",
        metadata_date=None,
        mtime_ns=None,
    )

    assert resolved == datetime(2026, 5, 2, 8, 0, tzinfo=UTC)


def test_resolve_doc_date_falls_back_to_date_key() -> None:
    resolved = resolve_doc_date(
        frontmatter={"date": datetime(2026, 4, 1, 12, 0)},
        filename="plan-2026-09-17.pdf",
        metadata_date=None,
        mtime_ns=None,
    )

    assert resolved == datetime(2026, 4, 1, 12, 0, tzinfo=UTC)


def test_resolve_doc_date_falls_back_to_created_key() -> None:
    resolved = resolve_doc_date(
        frontmatter={"created": date(2026, 3, 1)},
        filename="nada.pdf",
        metadata_date=None,
        mtime_ns=None,
    )

    assert resolved == datetime(2026, 3, 1, tzinfo=UTC)


def test_resolve_doc_date_order_filename_metadata_mtime() -> None:
    from_filename = resolve_doc_date(
        frontmatter={},
        filename="plan-2026-09-17.pdf",
        metadata_date=datetime(2026, 1, 1, tzinfo=UTC),
        mtime_ns=1,
    )
    from_metadata = resolve_doc_date(
        frontmatter={},
        filename="nada.pdf",
        metadata_date=datetime(2026, 1, 1, 12, 0),
        mtime_ns=(datetime(2020, 6, 1, tzinfo=UTC).timestamp() * 1_000_000_000).__int__(),
    )
    mtime_ns = int(datetime(2020, 6, 1, 12, 30, tzinfo=UTC).timestamp() * 1_000_000_000)
    from_mtime = resolve_doc_date(
        frontmatter={}, filename="nada.pdf", metadata_date=None, mtime_ns=mtime_ns
    )

    assert from_filename == datetime(2026, 9, 17, tzinfo=UTC)
    assert from_metadata == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    assert from_mtime == datetime(2020, 6, 1, 12, 30, tzinfo=UTC)


def test_resolve_doc_date_unparseable_frontmatter_is_skipped() -> None:
    resolved = resolve_doc_date(
        frontmatter={"updated": "ontem", "date": "2026-04-01"},
        filename="nada.pdf",
        metadata_date=None,
        mtime_ns=None,
    )

    assert resolved == datetime(2026, 4, 1, tzinfo=UTC)


def test_resolve_doc_date_none_without_sources() -> None:
    assert (
        resolve_doc_date(frontmatter={}, filename="nada.pdf", metadata_date=None, mtime_ns=None)
        is None
    )


def test_resolve_doc_date_assumes_naive_values_are_utc() -> None:
    resolved = resolve_doc_date(
        frontmatter={"updated": datetime(2026, 5, 2, 10, 0, tzinfo=UTC)},
        filename="nada.pdf",
        metadata_date=None,
        mtime_ns=None,
    )

    assert resolved.tzinfo is not None
    assert resolved.utcoffset().total_seconds() == 0
