"""Unit tests for the quoted-title lock (reorder path)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mynotes.retrieval import title_lock
from mynotes.retrieval.analyze import analyze_query
from mynotes.retrieval.types import SearchHit


def _hit(rank: int, document_id: int, title: str, path: str) -> SearchHit:
    return SearchHit(
        chunk_id=f"c{rank}",
        document_id=document_id,
        version_id=document_id * 10,
        rank=rank,
        score=1.0 / rank,
        text=f"texto {rank}",
        title=title,
        section=None,
        heading_path="",
        path=path,
        page=None,
        slide=None,
        doc_date=datetime(2026, 1, 1, tzinfo=UTC),
        doc_status="current",
        is_latest=True,
        is_table=False,
        format="md",
        language="pt",
        tags=[],
    )


def _hits() -> list[SearchHit]:
    return [
        _hit(1, 1, "Diario de reunioes - Xarrama", "projects/xarrama/diario-de-reunioes.md"),
        _hit(2, 2, "Diario de reunioes - Alva", "projects/alva/diario-de-reunioes.md"),
        _hit(3, 1, "Diario de reunioes - Xarrama", "projects/xarrama/diario-de-reunioes.md"),
        _hit(4, 3, "Stanza", "tools/stanza.md"),
        _hit(5, 2, "Diario de reunioes - Alva", "projects/alva/diario-de-reunioes.md"),
    ]


def test_quoted_title_moves_that_document_first_keeping_order() -> None:
    analysis = analyze_query('Em que linguagem esta o plugin da nota "Diário de reuniões - Alva"?')
    out = title_lock.lock_quoted_titles(_hits(), analysis)
    assert [h.chunk_id for h in out] == ["c2", "c5", "c1", "c3", "c4"]
    assert [h.rank for h in out] == [1, 2, 3, 4, 5]
    assert out[0].sources == {"title": 1} and "title" not in out[2].sources


def test_file_stem_counts_as_a_title() -> None:
    analysis = analyze_query('o que diz "stanza"?')
    out = title_lock.lock_quoted_titles(_hits(), analysis)
    assert out[0].chunk_id == "c4"


def test_no_quote_or_no_match_leaves_the_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(title_lock, "_load_quoted_documents", lambda quoted, analysis: [])
    hits = _hits()
    assert [
        h.chunk_id for h in title_lock.lock_quoted_titles(hits, analyze_query("sem aspas"))
    ] == [
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
    ]
    out = title_lock.lock_quoted_titles(hits, analyze_query('a nota "Inexistente"'))
    assert [h.chunk_id for h in out] == ["c1", "c2", "c3", "c4", "c5"]


def test_missing_document_is_loaded_and_put_first(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = [_hit(1, 9, "Plano Alfa", "projects/alfa/plano.md")]
    monkeypatch.setattr(title_lock, "_load_quoted_documents", lambda quoted, analysis: loaded)
    out = title_lock.lock_quoted_titles(_hits(), analyze_query('a nota "Plano Alfa"'))
    assert out[0].document_id == 9 and out[0].sources == {"title": 1}
    assert len(out) == 6


def test_named_date_anchors_and_loads_missing_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    hits = [
        _hit(1, 1, "Reuniao Atlas - 2026-05-11", "meetings/atlas/2026-05-11.md"),
        _hit(2, 2, "Reuniao Boreal - 2026-01-12", "meetings/boreal/2026-01-12.md"),
    ]
    loaded = [_hit(1, 9, "Reuniao Atlas - 2026-01-12", "meetings/atlas/2026-01-12.md")]
    monkeypatch.setattr(
        title_lock, "_load_dated_documents", lambda dates, analysis, exclude: loaded
    )
    out = title_lock.lock_named_dates(hits, analyze_query("Na reuniao de 2026-01-12 quem decidiu?"))
    assert [h.document_id for h in out] == [2, 9, 1]
    assert out[0].sources.get("date") == 1 and out[1].sources == {"date": 1}
    assert "date" not in out[2].sources


def test_front_anchored_moves_strong_anchors_only() -> None:
    hits = _hits()
    for hit, score in zip(hits, [0.9, 0.85, 0.8, 0.7, 0.1], strict=True):
        hit.rerank_score = score
    hits[2].sources = {"hybrid": 3, "date": 1}  # strong anchor: moves first
    hits[4].sources = {"hybrid": 5, "date": 1}  # 0.1 < 0.2 x 0.9: stays
    out = title_lock.front_anchored(hits)
    assert [h.chunk_id for h in out] == ["c3", "c1", "c2", "c4", "c5"]


def test_expand_series_marks_the_folder_and_appends_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hits = _hits()
    sibling = _hit(9, 8, "Diario de reunioes - Xarrama (old)", "projects/xarrama/diario-2025.md")
    sibling.sources = {"series": 1}  # as the real loader marks it
    monkeypatch.setattr(
        title_lock, "_load_folder_documents", lambda folder, analysis, exclude: [sibling]
    )
    out = title_lock.expand_series(hits, analyze_query("Em que reuniao o deploy passou a DEMO?"))
    assert out[-1] is sibling
    assert all("series" in h.sources for h in out if h.path.startswith("projects/xarrama/"))
    assert all("series" not in h.sources for h in out if not h.path.startswith("projects/xarrama/"))


def test_expand_series_does_nothing_for_other_questions() -> None:
    out = title_lock.expand_series(_hits(), analyze_query("Qual e o estado do deploy?"))
    assert [h.chunk_id for h in out] == ["c1", "c2", "c3", "c4", "c5"]
    assert all("series" not in h.sources for h in out)
