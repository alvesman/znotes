"""Unit tests for context assembly (task 29, PLAN.md §9 step 6).

A fake :class:`ChunkStore` and the :class:`WhitespaceCounter` keep the tests
offline; the budget arithmetic below uses the counter's ``ceil(words * 1.3)``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mynotes.chunk.tokenizer import WhitespaceCounter
from mynotes.config import Settings
from mynotes.retrieval.analyze import QueryAnalysis
from mynotes.retrieval.assemble import (
    ChunkRow,
    ContextBundle,
    assemble_context,
    register_extra_block,
    unregister_extra_block,
)
from mynotes.retrieval.pipeline import RetrievalResult
from mynotes.retrieval.types import SearchFilters, SearchHit

_COUNTER = WhitespaceCounter()


def _tokens(text: str) -> int:
    return _COUNTER.count(text)


def _settings(monkeypatch: pytest.MonkeyPatch, **retrieval: int) -> Settings:
    settings = Settings(
        retrieval={"max_docs": 4, "max_chunks": 10, "context_tokens": 3500, **retrieval}
    )
    monkeypatch.setattr("mynotes.retrieval.assemble.get_settings", lambda: settings)
    return settings


def _hit(
    rank: int,
    *,
    document_id: int = 1,
    version_id: int = 11,
    text: str = "corpo do trecho",
    title: str | None = "Nota",
    section: str | None = None,
    page: int | None = None,
    slide: int | None = None,
    is_table: bool = False,
    doc_status: str = "current",
    is_latest: bool = True,
    doc_date: datetime | None = datetime(2026, 1, 1, tzinfo=UTC),
    path: str | None = None,
) -> SearchHit:
    return SearchHit(
        chunk_id=f"chunk-{rank}",
        document_id=document_id,
        version_id=version_id,
        rank=rank,
        score=0.0,
        text=text,
        title=title,
        section=section,
        heading_path="",
        path=path or f"notes/doc{document_id}.md",
        page=page,
        slide=slide,
        doc_date=doc_date,
        doc_status=doc_status,
        is_latest=is_latest,
        is_table=is_table,
        format="md",
        language=None,
        tags=[],
    )


def _result(*hits: SearchHit) -> RetrievalResult:
    analysis = QueryAnalysis(
        raw="pergunta",
        text="pergunta",
        filters=SearchFilters(),
        intent="current",
        as_of=None,
        language=None,
    )
    return RetrievalResult(analysis=analysis, hits=list(hits), candidates=len(hits), relevant=True)


class FakeStore:
    """In-memory ChunkStore keyed like the DbChunkStore queries."""

    def __init__(self) -> None:
        self.sections: dict[tuple[int, int, str], list[ChunkRow]] = {}
        self.pages: dict[tuple[int, int, int | None, int | None], list[ChunkRow]] = {}
        self.section_calls: list[tuple[int, int, str]] = []
        self.page_calls: list[tuple[int, int, int | None, int | None]] = []

    def section_chunks(self, document_id: int, version_id: int, section: str) -> list[ChunkRow]:
        self.section_calls.append((document_id, version_id, section))
        return list(self.sections.get((document_id, version_id, section), []))

    def page_chunks(
        self, document_id: int, version_id: int, page: int | None, slide: int | None
    ) -> list[ChunkRow]:
        self.page_calls.append((document_id, version_id, page, slide))
        return list(self.pages.get((document_id, version_id, page, slide), []))


def _row(
    ordinal: int,
    text: str,
    *,
    page: int | None = None,
    slide: int | None = None,
    section: str | None = None,
    token_count: int = 30,
) -> ChunkRow:
    return ChunkRow(
        ordinal=ordinal,
        text=text,
        page=page,
        slide=slide,
        section=section,
        token_count=token_count,
    )


def test_empty_result_yields_an_empty_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch)

    bundle = assemble_context(_result())

    assert bundle == ContextBundle(
        text="", citations=[], tokens=0, has_table=False, extra_blocks=[]
    )


def test_document_cap_keeps_only_the_first_max_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch)
    hits = [_hit(rank, document_id=rank, text=f"texto {rank}", page=rank) for rank in range(1, 7)]

    bundle = assemble_context(_result(*hits))

    assert {citation.document_id for citation in bundle.citations} == {1, 2, 3, 4}
    assert "texto 5" not in bundle.text
    assert "texto 6" not in bundle.text


def test_chunk_cap_limits_the_selected_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch)
    hits = [
        _hit(rank, document_id=1, text=f"texto numero {rank}", page=rank) for rank in range(1, 16)
    ]

    bundle = assemble_context(_result(*hits))

    assert len(bundle.citations) == 10


def test_section_expansion_orders_seed_outward(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch)
    store = FakeStore()
    store.sections[(1, 11, "Setup")] = [
        _row(0, "primeira linha", page=1, section="Setup"),
        _row(1, "segunda linha", page=1, section="Setup"),
        _row(2, "alvo central", page=1, section="Setup"),
        _row(3, "quarta linha", page=2, section="Setup"),
        _row(4, "quinta linha", page=2, section="Setup"),
    ]
    hit = _hit(1, section="Setup", text="alvo central", page=1)

    bundle = assemble_context(_result(hit), store=store)

    positions = [
        bundle.text.index(mark)
        for mark in (
            "alvo central",
            "segunda linha",
            "primeira linha",
            "quarta linha",
            "quinta linha",
        )
    ]
    assert positions == sorted(positions), "seed outward inside each page group"
    assert "p.2" in bundle.text, "cross-page rows keep their own page attribution"
    assert store.section_calls == [(1, 11, "Setup")]


def test_section_expansion_stops_at_the_token_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch)
    store = FakeStore()
    store.sections[(1, 11, "Setup")] = [
        _row(0, "esquerda", token_count=400),
        _row(1, "alvo", token_count=400),
        _row(2, "direita", token_count=400),
    ]
    hit = _hit(1, section="Setup", text="alvo", page=1)

    bundle = assemble_context(_result(hit), store=store)

    assert "alvo" in bundle.text
    assert "esquerda" not in bundle.text
    assert "direita" not in bundle.text


def test_page_expansion_when_the_hit_has_no_section(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch)
    store = FakeStore()
    store.pages[(1, 11, 4, None)] = [
        _row(0, "paragrafo um", page=4),
        _row(1, "alvo da busca", page=4),
        _row(2, "paragrafo tres", page=4),
    ]
    hit = _hit(1, text="alvo da busca", page=4)

    bundle = assemble_context(_result(hit), store=store)

    assert (
        bundle.text.index("alvo da busca")
        < bundle.text.index("paragrafo um")
        < bundle.text.index("paragrafo tres")
    )


def test_heading_only_seed_falls_back_to_page_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings(monkeypatch)
    store = FakeStore()
    store.sections[(1, 11, "Introducao")] = [_row(3, "Introducao", token_count=5)]
    store.pages[(1, 11, 1, None)] = [
        _row(2, "paragrafo anterior da pagina", page=1),
        _row(3, "Introducao", page=1, token_count=5),
    ]
    hit = _hit(1, section="Introducao", text="Introducao", page=1)

    bundle = assemble_context(_result(hit), store=store)

    assert "paragrafo anterior da pagina" in bundle.text
    assert store.page_calls == [(1, 11, 1, None)]


def test_dedup_keeps_overlapping_expansion_text_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch)
    store = FakeStore()
    seed_text = "O frigorifico esta avariado e precisa de revisao tecnica completa antes do verao."
    neighbour_text = "vizinho distinto da mesma secao traz o complemento do procedimento escrito"
    store.sections[(1, 11, "Frigorifico")] = [
        _row(4, seed_text, page=1, section="Frigorifico"),
        _row(5, neighbour_text, page=1, section="Frigorifico"),
    ]
    first = _hit(1, section="Frigorifico", text=seed_text, page=1)
    second = _hit(2, text=neighbour_text, page=1)

    bundle = assemble_context(_result(first, second), store=store)

    assert bundle.text.count("O frigorifico esta avariado") == 1
    assert bundle.text.count("vizinho distinto da mesma secao") == 1


def test_dedup_by_document_and_ordinal_across_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings(monkeypatch)
    store = FakeStore()
    store.sections[(1, 11, "Alpha")] = [
        _row(5, "semente da versao antiga com texto suficiente para o teste de expansao"),
        _row(7, "colisao de ordinal na versao antiga com texto suficiente para contar"),
    ]
    store.sections[(1, 12, "Beta")] = [
        _row(6, "semente da versao nova com texto suficiente para o teste de expansao", page=2),
        _row(7, "colisao de ordinal na versao nova com texto suficiente para contar", page=2),
    ]
    old_hit = _hit(1, version_id=11, section="Alpha", text=store.sections[(1, 11, "Alpha")][0].text)
    new_hit = _hit(
        2, version_id=12, section="Beta", text=store.sections[(1, 12, "Beta")][0].text, page=2
    )

    bundle = assemble_context(_result(old_hit, new_hit), store=store)

    assert "colisao de ordinal na versao antiga" in bundle.text
    assert "colisao de ordinal na versao nova" not in bundle.text


def test_headers_and_citations_follow_first_appearance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings(monkeypatch)
    top = _hit(
        1,
        document_id=1,
        title="Manual Frigorifico",
        text="texto do manual atual",
        page=3,
        doc_date=datetime(2026, 3, 1, tzinfo=UTC),
    )
    archived = _hit(
        2,
        document_id=2,
        title="Spec Antiga",
        text="texto do documento arquivado",
        slide=4,
        doc_status="archived",
        is_latest=False,
        doc_date=datetime(2025, 12, 31, tzinfo=UTC),
    )
    plain = _hit(3, document_id=3, title=None, text="texto sem titulo nem data", doc_date=None)

    bundle = assemble_context(_result(top, archived, plain))
    headers = [line for line in bundle.text.splitlines() if line.startswith("[")]

    assert headers == [
        "[1] Manual Frigorifico — notes/doc1.md — p.3 — 2026-03-01",
        "[2] Spec Antiga — notes/doc2.md — slide 4 — 2025-12-31 — historical",
        "[3] notes/doc3.md",
    ]
    assert [citation.n for citation in bundle.citations] == [1, 2, 3]
    assert [citation.is_current for citation in bundle.citations] == [True, False, True]
    assert bundle.citations[0].document_id == 1
    assert bundle.citations[0].path == "notes/doc1.md"
    assert bundle.citations[1].page is None and bundle.citations[1].slide == 4


def test_budget_drops_lowest_ranked_snippets_and_keeps_the_top_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings(monkeypatch, context_tokens=60)
    top = _hit(
        1,
        text="primeira frase do topo. segunda frase do topo. terceira frase do topo.",
        page=1,
    )
    other = _hit(2, document_id=2, text=" ".join(f"palavra{i}" for i in range(60)), page=1)

    bundle = assemble_context(_result(top, other))

    assert "palavra" not in bundle.text
    assert "primeira frase do topo" in bundle.text
    assert "terceira frase do topo" in bundle.text
    assert bundle.tokens <= 60
    assert [citation.n for citation in bundle.citations] == [1]


def test_budget_truncates_the_last_snippet_at_a_sentence_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings(monkeypatch, context_tokens=24)
    top = _hit(
        1,
        text=(
            "primeira frase completa aqui. segunda frase completa aqui. "
            "terceira frase completa aqui."
        ),
        page=1,
    )

    bundle = assemble_context(_result(top))

    assert "primeira frase completa aqui." in bundle.text
    assert "segunda frase completa aqui." in bundle.text
    assert "terceira frase" not in bundle.text
    assert bundle.text.endswith("aqui.")
    assert bundle.tokens <= 24


def test_registered_extra_block_is_appended_and_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings(monkeypatch)

    class GraphFacts:
        name = "graph_facts"

        def __init__(self) -> None:
            self.calls = 0

        def block(self, result: object, selected: object) -> str:
            self.calls += 1
            return "EntidadeA — liga_para — EntidadeB (2026-05-01)"

    provider = GraphFacts()
    register_extra_block(provider)
    try:
        bundle = assemble_context(_result(_hit(1, text="conteudo principal da nota")))
    finally:
        unregister_extra_block("graph_facts")

    assert provider.calls == 1
    assert bundle.text.endswith("### Graph facts\nEntidadeA — liga_para — EntidadeB (2026-05-01)")
    assert bundle.extra_blocks == ["EntidadeA — liga_para — EntidadeB (2026-05-01)"]
    assert bundle.tokens == _tokens(bundle.text)
    assert bundle.tokens <= 3500


def test_extra_blocks_are_capped_at_a_share_of_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings(monkeypatch, context_tokens=100)

    class BigNotes:
        name = "version_notes"

        def block(self, result: object, selected: object) -> str:
            return " ".join(f"nota{i}" for i in range(60))

    register_extra_block(BigNotes())
    try:
        bundle = assemble_context(_result(_hit(1, text="conteudo principal da nota")))
    finally:
        unregister_extra_block("version_notes")

    assert bundle.extra_blocks, "the truncated block is still appended"
    assert _tokens(bundle.extra_blocks[0]) <= int(100 * 0.2)
    assert bundle.extra_blocks[0].startswith("nota0")
    assert "conteudo principal da nota" in bundle.text
    assert bundle.tokens <= 100


def test_failing_extra_block_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch)

    class Boom:
        name = "boom"

        def block(self, result: object, selected: object) -> str:
            raise RuntimeError("boom")

    register_extra_block(Boom())
    try:
        bundle = assemble_context(_result(_hit(1, text="conteudo principal da nota")))
    finally:
        unregister_extra_block("boom")

    assert "boom" not in bundle.text
    assert bundle.extra_blocks == []


def test_has_table_from_the_hit_flag_or_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch)

    flagged = assemble_context(_result(_hit(1, is_table=True, text="tabela em outro formato")))
    markdown = assemble_context(
        _result(_hit(1, text="colunas:\n\n| x | y |\n| --- | --- |\n| 1 | 2 |"))
    )
    plain = assemble_context(_result(_hit(1, text="texto sem tabela nenhuma aqui")))

    assert flagged.has_table is True
    assert markdown.has_table is True
    assert plain.has_table is False


class TestScoreGapRule:
    """``retrieval.extra_docs_min_ratio``: close runners-up join the context."""

    @staticmethod
    def _scored(scores: list[float]) -> list[SearchHit]:
        from dataclasses import replace

        return [
            replace(_hit(rank, document_id=rank, text=f"texto {rank}"), rerank_score=score)
            for rank, score in enumerate(scores, start=1)
        ]

    def test_off_by_default_keeps_the_flat_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _settings(monkeypatch)
        bundle = assemble_context(_result(*self._scored([1.0, 0.9, 0.9, 0.9, 0.9, 0.9])))
        assert {c.document_id for c in bundle.citations} == {1, 2, 3, 4}

    def test_admits_close_documents_up_to_extra_max(self) -> None:
        from mynotes.retrieval.assemble import _allowed_documents

        hits = self._scored([1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.52])
        assert _allowed_documents(hits, 4, extra_max=2, min_ratio=0.5) == {1, 2, 3, 4, 5, 6}
        assert _allowed_documents(hits, 4, extra_max=1, min_ratio=0.5) == {1, 2, 3, 4, 5}

    def test_first_document_below_the_ratio_ends_the_extension(self) -> None:
        from mynotes.retrieval.assemble import _allowed_documents

        hits = self._scored([1.0, 0.9, 0.8, 0.7, 0.3, 0.6])
        assert _allowed_documents(hits, 4, extra_max=2, min_ratio=0.5) == {1, 2, 3, 4}

    def test_falls_back_to_fused_score_without_rerank(self) -> None:
        from mynotes.retrieval.assemble import _allowed_documents

        hits = [_hit(rank, document_id=rank) for rank in range(1, 7)]
        for hit, score in zip(hits, [0.05, 0.04, 0.04, 0.04, 0.04, 0.01], strict=True):
            hit.score = score
        assert _allowed_documents(hits, 4, extra_max=2, min_ratio=0.5) == {1, 2, 3, 4, 5}

    def test_via_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _settings(monkeypatch)
        settings.retrieval.extra_docs_min_ratio = 0.5
        bundle = assemble_context(_result(*self._scored([1.0, 0.9, 0.8, 0.7, 0.6, 0.2])))
        assert {c.document_id for c in bundle.citations} == {1, 2, 3, 4, 5}
