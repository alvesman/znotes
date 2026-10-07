"""Unit tests for the query analysis (task 28, PLAN.md §9 step 1)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mynotes.extract.markdown import _HASHTAG_RE as _MARKDOWN_HASHTAG_RE
from mynotes.retrieval.analyze import HASHTAG_RE, analyze_query
from mynotes.retrieval.types import SearchFilters


def test_inline_filters_are_parsed_and_removed() -> None:
    analysis = analyze_query('custo do projeto #Projeto-X in:"Projetos/Alfa" since:2026-01')

    assert analysis.raw == 'custo do projeto #Projeto-X in:"Projetos/Alfa" since:2026-01'
    assert analysis.text == "custo do projeto"
    assert analysis.filters.tags == ["projeto-x"]
    assert analysis.filters.folders == ["Projetos/Alfa"]
    assert analysis.filters.since == datetime(2026, 1, 1, tzinfo=UTC)
    assert analysis.intent == "current"
    assert analysis.filters.include_superseded is False
    assert analysis.as_of is None


def test_history_keyword_sets_intent() -> None:
    analysis = analyze_query("qual era o orçamento antes?")

    assert analysis.intent == "history"
    assert analysis.filters.include_superseded is True
    assert "antes" in analysis.text


def test_history_keyword_is_accent_and_case_insensitive() -> None:
    analysis = analyze_query("VERSÃO ANTERIOR do plano")

    assert analysis.intent == "history"
    assert analysis.text == "VERSÃO ANTERIOR do plano"


def test_as_of_sets_history_intent_and_date() -> None:
    analysis = analyze_query("as-of:2026-03 budget")

    assert analysis.intent == "history"
    assert analysis.as_of == datetime(2026, 3, 1, tzinfo=UTC)
    assert analysis.text == "budget"
    assert analysis.filters.include_superseded is True


def test_hash_number_is_not_a_tag() -> None:
    analysis = analyze_query("issue #1 open")

    assert analysis.filters.tags == []
    assert analysis.text == "issue #1 open"


def test_until_covers_the_whole_day_or_month() -> None:
    day = analyze_query("relatorio until:2026-01-15")
    month = analyze_query("relatorio until:2026-01")

    assert day.filters.until == datetime(2026, 1, 15, 23, 59, 59, 999999, tzinfo=UTC)
    assert month.filters.until == datetime(2026, 1, 31, 23, 59, 59, 999999, tzinfo=UTC)


def test_history_token_alone_sets_intent() -> None:
    analysis = analyze_query("frigorifico history:")

    assert analysis.intent == "history"
    assert analysis.text == "frigorifico"


def test_format_token_keeps_the_value() -> None:
    analysis = analyze_query("contrato format:pdf")

    assert analysis.filters.formats == ["pdf"]
    assert analysis.text == "contrato"


def test_unparseable_dates_stay_in_the_text() -> None:
    analysis = analyze_query("since:bananas relatorio")

    assert analysis.filters.since is None
    assert analysis.text == "since:bananas relatorio"


def test_base_filters_are_merged_with_inline_filters() -> None:
    base = SearchFilters(
        tags=["trabalho"],
        formats=["md"],
        since=datetime(2025, 1, 1, tzinfo=UTC),
        acl_prefixes=["Private"],
        workspace_id=7,
    )

    analysis = analyze_query("custo #projeto-x since:2026-01", base)

    assert analysis.filters.tags == ["trabalho", "projeto-x"]
    assert analysis.filters.formats == ["md"]
    assert analysis.filters.since == datetime(2026, 1, 1, tzinfo=UTC)
    assert analysis.filters.acl_prefixes == ["Private"]
    assert analysis.filters.workspace_id == 7


def test_base_include_superseded_survives_current_intent() -> None:
    analysis = analyze_query("custo", SearchFilters(include_superseded=True))

    assert analysis.intent == "current"
    assert analysis.filters.include_superseded is True


def test_history_intent_takes_old_versions_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mynotes.config import Settings

    settings = Settings()
    settings.versions.index_old_versions = True
    monkeypatch.setattr("mynotes.retrieval.analyze.get_settings", lambda: settings)

    analysis = analyze_query("versoes do plano")

    assert analysis.intent == "history"
    assert analysis.filters.include_old_versions is True

    settings.versions.index_old_versions = False
    off = analyze_query("versoes do plano")
    assert off.filters.include_old_versions is False


def test_duplicate_tags_and_folders_collapse() -> None:
    analysis = analyze_query("#projeto-x #Projeto-X in:Projetos in:projetos")

    assert analysis.filters.tags == ["projeto-x"]
    assert analysis.filters.folders == ["Projetos", "projetos"]


def test_language_is_detected_on_the_cleaned_text() -> None:
    analysis = analyze_query("What is the budget for the storage engine project?")

    assert analysis.language == "en"


def test_short_queries_have_no_language() -> None:
    assert analyze_query("custo").language is None


def test_hashtag_regex_matches_the_markdown_extractor() -> None:
    assert HASHTAG_RE.pattern == _MARKDOWN_HASHTAG_RE.pattern


def test_date_tokens_accept_a_space_after_the_colon() -> None:
    """``as-of: 2025-09-20`` (the natural spelling) must work like ``as-of:2025-09-20``."""
    analysis = analyze_query("Qual era o orcamento do Projeto Coa? as-of: 2025-09-20")
    assert analysis.intent == "history"
    assert analysis.as_of is not None and analysis.as_of.date().isoformat() == "2025-09-20"
    assert analysis.text == "Qual era o orcamento do Projeto Coa?"

    spaced = analyze_query("relatorio since: 2026-01 until: 2026-03")
    assert spaced.filters.since is not None and spaced.filters.since.month == 1
    assert spaced.filters.until is not None and spaced.filters.until.month == 3
    assert spaced.text == "relatorio"


def test_date_tokens_without_a_date_are_left_in_the_text() -> None:
    analysis = analyze_query("as-of: tomorrow budget")
    assert analysis.intent == "current" and analysis.as_of is None
    assert analysis.text == "as-of: tomorrow budget"


def test_as_of_bounds_the_document_date() -> None:
    """``as-of`` also acts as an ``until`` at the end of that day (or month)."""
    day = analyze_query("orcamento as-of:2025-09-20")
    assert day.filters.until is not None
    assert day.filters.until.isoformat().startswith("2025-09-20T23:59:59")
    month = analyze_query("orcamento as-of:2025-09")
    assert month.filters.until is not None
    assert month.filters.until.isoformat().startswith("2025-09-30T23:59:59")
    tighter = analyze_query("orcamento until:2025-06-30 as-of:2025-09-20")
    assert tighter.filters.until is not None and tighter.filters.until.month == 6


def test_quoted_phrases_are_extracted_in_order_without_filter_tokens() -> None:
    from mynotes.retrieval.analyze import analyze_query, quoted_phrases

    assert quoted_phrases('na nota "Diario de reunioes - Alva" e em “Plano B”') == (
        "Diario de reunioes - Alva",
        "Plano B",
    )
    assert quoted_phrases("sem aspas") == ()
    analysis = analyze_query('in:"projects/alfa" o que diz a nota "Plano Alfa"?')
    assert analysis.quoted == ("Plano Alfa",)


def test_named_dates_in_all_supported_forms_and_only_for_current_intent() -> None:
    from datetime import date

    from mynotes.retrieval.analyze import analyze_query, named_dates

    assert named_dates("Na reunião de 2026-01-12 e em 19/01/2026") == (
        date(2026, 1, 12),
        date(2026, 1, 19),
    )
    assert named_dates("realizada em 10 de março de 2025") == (date(2025, 3, 10),)
    assert named_dates("for January 7, 2022 and 8 January 2022") == (
        date(2022, 1, 7),
        date(2022, 1, 8),
    )
    assert named_dates("em junho de 2025, v2.10.2026, 2026-13-40") == ()
    assert analyze_query("Na reunião de 2026-01-12 quem decidiu?").named_dates == (
        date(2026, 1, 12),
    )
    # in an as-of question the written date is a cutoff, not a note
    assert (
        analyze_query("Qual era o prazo no estado de 2026-06-05? as-of: 2026-06-05").named_dates
        == ()
    )


def test_change_questions() -> None:
    from mynotes.retrieval.analyze import analyze_query

    assert analyze_query("Em que reunião o deploy passou a DEMO?").asks_change
    assert analyze_query("When did the status change to done? It changed to done").asks_change
    assert not analyze_query("Qual é o estado atual do deploy?").asks_change
