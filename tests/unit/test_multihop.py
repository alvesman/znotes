"""Unit tests for the bridge-following pieces of graph expansion (option A)."""

from __future__ import annotations

from datetime import UTC, datetime

from fakes import FakeReranker
from mynotes.config import Settings
from mynotes.graph.expand import Subgraph, score_entities
from mynotes.graph.extract import ChunkGroup, merge_small_groups
from mynotes.graph.resolve import entity_norm
from mynotes.graph.store import variant_groups
from mynotes.retrieval.analyze import analyze_query
from mynotes.retrieval.graph_expand import (
    asks_fact_about_subject,
    graph_expansion_active,
    is_chain_question,
    is_deep_chain_question,
    mention_seed_weight,
    preferred_entity_types,
    type_factor,
)
from mynotes.retrieval.rerank import promote_bridge_hit, rerank_hits
from mynotes.retrieval.types import SearchHit


class TestMentionSeedWeight:
    def test_top_document_and_rare_entity_weigh_most(self) -> None:
        assert mention_seed_weight(0, 2) > mention_seed_weight(1, 2) > mention_seed_weight(3, 2)
        assert mention_seed_weight(0, 2) > mention_seed_weight(0, 20) > mention_seed_weight(0, 200)
        assert 0 < mention_seed_weight(5, 500) < mention_seed_weight(0, 0) <= 0.9

    def test_rare_tool_in_top_document_beats_common_entity(self) -> None:
        tool = mention_seed_weight(0, 3)  # the plugin named by the bridge note
        team = mention_seed_weight(0, 60)  # "equipa", mentioned everywhere
        assert tool > 2 * team


class TestHubDamping:
    def test_specific_neighbour_outscores_a_hub(self) -> None:
        # Seed 1 touches the specific node 2 and the hub 3; the hub also touches 4..9.
        edges = [(1, 2, 10, 0.8), (1, 3, 11, 0.8)] + [(3, n, 20 + n, 0.8) for n in range(4, 10)]
        sub = Subgraph(nodes={n: f"e{n}" for n in range(1, 10)}, edges=edges)
        scores = score_entities(sub, {1: 1.0}, method="one_hop")
        assert scores[2] > scores[3]
        assert scores[1] == 1.0


def _hit(chunk_id: str, text: str, hint: str | None = None) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id=hash(chunk_id) % 1000,
        version_id=1,
        rank=1,
        score=0.01,
        text=text,
        title=None,
        section=None,
        heading_path="",
        path=f"notes/{chunk_id}.md",
        page=None,
        slide=None,
        doc_date=datetime(2026, 1, 1, tzinfo=UTC),
        doc_status="current",
        is_latest=True,
        is_table=False,
        format="md",
        language=None,
        tags=[],
        query_hint=hint,
    )


class TestHintedRerank:
    def test_second_hop_note_wins_only_with_its_hint(self) -> None:
        query = "quem mantem o plugin usado para a vista de etiquetas na nota X"
        bridge = _hit("bridge", "para a vista de etiquetas da nota X usamos o plugin Pagewright")
        tool_plain = _hit("tool1", "a ferramenta Pagewright e mantida por Rita Figueiredo")
        tool_hinted = _hit(
            "tool2", "a ferramenta Pagewright e mantida por Rita Figueiredo", "Pagewright"
        )
        ranked = rerank_hits(query, [bridge, tool_plain, tool_hinted], FakeReranker(), top=10)
        by_id = {hit.chunk_id: hit for hit in ranked}
        assert by_id["tool2"].rerank_score > by_id["tool1"].rerank_score
        assert ranked[0].chunk_id == "bridge"  # the bridge still matches the question best
        assert [hit.chunk_id for hit in ranked].index("tool2") < [
            hit.chunk_id for hit in ranked
        ].index("tool1")

    def test_hint_never_lowers_a_score(self) -> None:
        query = "quem mantem o plugin Pagewright"
        plain = _hit("a", "a ferramenta Pagewright e mantida por Rita")
        hinted = _hit("b", "a ferramenta Pagewright e mantida por Rita", "zzz")  # useless hint
        rerank_hits(query, [plain, hinted], FakeReranker(), top=10)
        assert hinted.rerank_score >= plain.rerank_score - 1e-9


def _group(section: str, tokens: int, *, table: bool = False) -> ChunkGroup:
    from uuid import uuid4

    return ChunkGroup(
        chunk_ids=[uuid4()],
        section=section,
        text=f"{section} " + "x " * tokens,
        content_hash=section,
        token_count=tokens,
        is_table=table,
    )


class TestMergeSmallGroups:
    def test_short_sections_merge_until_extractable(self) -> None:
        groups = [
            _group("Title", 8),
            _group("Context", 26),
            _group("Tool", 29),
            _group("Procedure", 32),
        ]
        merged = merge_small_groups(groups, min_tokens=40, max_tokens=1500)
        assert [g.token_count for g in merged] == [34 + 29, 32] or [
            g.token_count for g in merged
        ] == [63, 32]
        assert merged[0].section == "Title / Context / Tool"
        assert len(merged[0].chunk_ids) == 3

    def test_large_groups_and_tables_are_untouched(self) -> None:
        groups = [_group("Intro", 300), _group("Table", 10, table=True), _group("Notes", 500)]
        merged = merge_small_groups(groups, min_tokens=40, max_tokens=1500)
        assert [(g.section, g.token_count, g.is_table) for g in merged] == [
            ("Intro", 300, False),
            ("Table", 10, True),
            ("Notes", 500, False),
        ]

    def test_small_tail_joins_the_previous_small_run_only(self) -> None:
        groups = [
            _group("A", 10),
            _group("B", 10),
            _group("C", 10),
            _group("D", 10),
            _group("E", 10),
        ]
        merged = merge_small_groups(groups, min_tokens=40, max_tokens=1500)
        assert [g.token_count for g in merged] == [40, 10]


class TestChainQuestion:
    def test_chain_shapes_are_detected(self) -> None:
        for q in (
            'Quem mantem o plugin usado para a vista de nuvem de etiquetas na nota "Vista"?',
            "What is the licence of the tool that Project Tua uses for the weekly note review?",
            "Em que linguagem esta escrita a ferramenta em que assenta o modulo de exportacao?",
            "Que ideia e proposta nos casos de uso do plugin que a nota 'X' usa?",
            "Which language is the tool that the export module of Zambujal is built on written in?",
        ):
            assert is_chain_question(q), q

    def test_direct_questions_are_not(self) -> None:
        for q in (
            "Quem mantem o plugin Pagewright?",
            "Qual e o orcamento aprovado do Projeto Coa?",
            "What is the default port of Snipster?",
            "Qual e o valor do parametro grant_type nos pedidos cURL?",
        ):
            assert not is_chain_question(q), q


class TestEntityNorm:
    def test_type_words_and_articles_are_stripped(self) -> None:
        assert entity_norm("the Kanbanist plugin") == "kanbanist"
        assert entity_norm("Kanbanist") == "kanbanist"
        assert entity_norm("ferramenta Tagloom") == "tagloom"
        assert entity_norm("Tagloom (extensão)") == entity_norm("Tagloom")
        assert entity_norm("plugin") == "plugin"  # nothing would remain
        assert entity_norm("Export module") == "export module"  # not a generic type word
        assert entity_norm("Rita Ferramenta", "Person") == "rita ferramenta"  # surnames stay
        assert entity_norm("Rita Ferramenta", "Technology") == "rita"


class TestGraphExpansionAuto:
    def test_auto_only_for_chain_questions(self) -> None:
        settings = Settings()
        assert settings.retrieval.graph_expansion == "auto"
        chain = analyze_query("Who maintains the tool that Project Tua uses for the weekly review?")
        direct = analyze_query("Who maintains the Snipster plugin?")
        assert graph_expansion_active(settings, chain) is True
        assert graph_expansion_active(settings, direct) is False
        settings.retrieval.graph_expansion = True
        assert graph_expansion_active(settings, direct) is True
        settings.retrieval.graph_expansion = False
        assert graph_expansion_active(settings, chain) is False


class TestPromoteBridgeHit:
    def _ranked(self) -> list[SearchHit]:
        hits = [
            _hit("bridge", "bridge note", None),
            _hit("other1", "look-alike 1", None),
            _hit("other2", "look-alike 2", None),
            _hit("other3", "look-alike 3", None),
            _hit("other4", "look-alike 4", None),
            _hit("tool", "the tool note", "Pagewright"),
        ]
        for i, hit in enumerate(hits):
            hit.document_id = i + 1
            hit.rerank_score = 0.9 - 0.1 * i
            hit.rank = i + 1
        return hits

    def test_best_hinted_hit_moves_to_second_place(self) -> None:
        hits = self._ranked()
        out = promote_bridge_hit(hits, max_docs=4)
        assert [h.chunk_id for h in out][:3] == ["bridge", "tool", "other1"]
        assert [h.rank for h in out] == [1, 2, 3, 4, 5, 6]

    def test_nothing_moves_when_already_in_context_or_too_weak(self) -> None:
        hits = self._ranked()
        hits[5].rerank_score = 0.02  # below min_score
        assert [h.chunk_id for h in promote_bridge_hit(hits, max_docs=4)][1] == "other1"
        hits = self._ranked()
        assert [h.chunk_id for h in promote_bridge_hit(hits, max_docs=6)][
            1
        ] == "other1"  # in context


class TestVariantGroups:
    def test_type_word_variants_group_under_the_bare_name(self) -> None:
        rows = [
            (10, "Technology", "kanbanist plugin"),
            (12, "Technology", "kanbanist"),
            (15, "Technology", "the kanbanist tool"),
            (20, "Technology", "tagloom plugin"),  # no bare twin: lowest id keeps
            (21, "Technology", "ferramenta tagloom"),
            (30, "Person", "kanbanist"),  # other type: separate
            (31, "Person", "rita ferramenta"),  # a surname, not a type word
            (32, "Person", "rita"),
            (40, "Technology", "snipster"),  # alone
        ]
        assert variant_groups(rows) == [(12, [10, 15]), (20, [21])]


class TestDeepChain:
    def test_component_chains_only(self) -> None:
        assert is_deep_chain_question(
            "Qual e a licenca da ferramenta em que assenta o modulo de exportacao do Projeto Raia?"
        )
        assert is_deep_chain_question(
            "Which language is the tool that the export module of Zambujal is built on written in?"
        )
        assert not is_deep_chain_question(
            "What is the licence of the tool that Project Tua uses for the weekly note review?"
        )
        assert not is_deep_chain_question('Quem mantem o plugin usado na nota "Vista"?')


class TestPreferredTypes:
    def test_subject_nouns_map_to_entity_types(self) -> None:
        assert preferred_entity_types("Quem mantem o plugin usado na nota X?") == {
            "Technology",
            "Person",
            "Organization",
        }
        assert preferred_entity_types("What is the licence of the tool that Project Tua uses?") == {
            "Technology",
            "Project",
        }
        assert preferred_entity_types("Qual e o orcamento aprovado?") == frozenset()
        preferred = preferred_entity_types("the tool that the service of Project Tua is built on")
        assert type_factor("Technology", preferred) == 1.5
        assert type_factor("Concept", preferred) == 0.5
        assert type_factor("Person", preferred) == 1.0


class TestHintedBeyondTop:
    def test_hinted_hits_past_the_window_are_still_reranked(self) -> None:
        query = "quem mantem o plugin usado na nota X Pagewright"
        hits = [_hit(f"h{i}", "texto sem relacao nenhuma", None) for i in range(5)]
        tool = _hit("tool", "a ferramenta Pagewright e mantida por Rita", "Pagewright")
        for i, hit in enumerate([*hits, tool]):
            hit.document_id = i + 1
        ranked = rerank_hits(query, [*hits, tool], FakeReranker(), top=3)
        assert "tool" in [hit.chunk_id for hit in ranked]  # not dropped with the tail
        assert ranked[0].chunk_id == "tool"


class TestFactQuestions:
    def test_fact_questions_are_detected(self) -> None:
        for q in (
            "Quem mantem o plugin Dataview usado pela vista tagcloud das notas?",
            "What is the licence of Tagloom?",
            "Em que linguagem esta escrita a ferramenta Codexa?",
            "Which repository hosts the Snipster plugin?",
        ):
            assert asks_fact_about_subject(q), q

    def test_other_questions_are_not(self) -> None:
        for q in (
            "Que tarefas estao planeadas para a semana W41Y2025?",
            "Qual e o prazo de entrega do Projeto Alfa?",
        ):
            assert not asks_fact_about_subject(q), q


class TestBridgePromotionTypesAndSlots:
    @staticmethod
    def _hits() -> list[SearchHit]:
        hits = [
            _hit("top", "the bridge note"),
            _hit("other1", "look-alike 1"),
            _hit("other2", "look-alike 2"),
            _hit("other3", "look-alike 3"),
            _hit("org", "a team note", "equipa de plataforma"),
            _hit("tool", "the tool note", "Stanza"),
            _hit("tool2", "the tool note, later section", "Stanza"),
        ]
        for i, (hit, doc, score) in enumerate(
            zip(hits, [1, 2, 3, 4, 5, 6, 6], [0.99, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2], strict=True)
        ):
            hit.document_id, hit.rerank_score, hit.rank = doc, score, i + 1
        hits[4].query_hint_type = "Organization"
        hits[5].query_hint_type = "Technology"
        hits[6].query_hint_type = "Technology"
        return hits

    def test_preferred_type_beats_a_higher_score(self) -> None:
        out = promote_bridge_hit(
            self._hits(), max_docs=4, preferred_types=frozenset({"Technology"})
        )
        assert [h.chunk_id for h in out][:2] == ["top", "tool"]

    def test_second_slot_only_for_a_preferred_type(self) -> None:
        # without a preferred type, the second slot stays empty: one weak hint is enough
        out = promote_bridge_hit(self._hits(), max_docs=4, slots=2)
        assert [h.chunk_id for h in out][:3] == ["top", "org", "other1"]
        hits = self._hits()
        hits[6].document_id = 7  # a second tool note in its own document
        out = promote_bridge_hit(
            hits, max_docs=4, slots=2, preferred_types=frozenset({"Technology"})
        )
        assert [h.chunk_id for h in out][:3] == ["top", "tool", "tool2"]
        assert [h.rank for h in out] == list(range(1, 8))

    def test_bridge_types_ignore_the_answer_type(self) -> None:
        from mynotes.retrieval.graph_expand import bridge_entity_types

        assert bridge_entity_types("Who maintains the tool that the export module uses?") == {
            "Technology"
        }
        assert bridge_entity_types("Quem coordena o Projeto Alfa?") == frozenset()

    def test_without_preference_the_score_decides(self) -> None:
        out = promote_bridge_hit(self._hits(), max_docs=4)
        assert [h.chunk_id for h in out][:2] == ["top", "org"]


class TestChangeCluster:
    def test_near_tied_hits_go_earliest_first_and_the_rest_follows(self) -> None:
        from mynotes.retrieval.rerank import order_change_cluster

        dates = ["2026-02-16", "2026-02-09", "2026-01-26", "2026-02-02", "2026-05-18"]
        scores = [0.9534, 0.949, 0.9503, 0.9444, 0.8474]
        hits = []
        for i, (day, score) in enumerate(zip(dates, scores, strict=True)):
            hit = _hit(f"w{i}", f"week {day}")
            hit.doc_date = datetime.fromisoformat(day).replace(tzinfo=UTC)
            hit.rerank_score, hit.rank = score, i + 1
            hits.append(hit)
        out = order_change_cluster(hits)
        assert [h.doc_date.date().isoformat() for h in out] == [
            "2026-01-26",
            "2026-02-02",
            "2026-02-09",
            "2026-02-16",
            "2026-05-18",
        ]
        assert [h.rank for h in out] == [1, 2, 3, 4, 5]


class TestChangeClusterRule:
    @staticmethod
    def _notes(rows: list[tuple[str, float]]) -> list[SearchHit]:
        hits = []
        for i, (day, score) in enumerate(rows):
            hit = _hit(f"n{i}", f"note {day}")
            hit.document_id = i + 1
            hit.path = f"meetings/series/{day}.md"
            hit.doc_date = datetime.fromisoformat(day).replace(tzinfo=UTC)
            hit.rerank_score, hit.rank = score, i + 1
            hit.sources = {"hybrid": i + 1, "series": 1}
            hits.append(hit)
        return hits

    def test_first_run_of_near_equal_scores(self) -> None:
        from mynotes.retrieval.rerank import change_cluster

        hits = self._notes(
            [
                ("2026-01-05", 0.68),
                ("2026-01-26", 0.775),
                ("2026-02-02", 0.756),
                ("2026-04-06", 0.45),
            ]
        )
        assert {h.doc_date.date().isoformat() for h in change_cluster(hits)} == {
            "2026-01-26",
            "2026-02-02",
        }

    def test_a_lone_top_note_that_is_not_the_newest_merges_with_the_next_run(self) -> None:
        from mynotes.retrieval.rerank import order_change_cluster

        hits = self._notes(
            [
                ("2026-05-11", 0.848),  # one note of the *next* value, scored above the rest
                ("2026-04-27", 0.802),
                ("2026-03-30", 0.778),
                ("2026-03-23", 0.766),
                ("2026-03-16", 0.462),  # the old value
                ("2026-05-18", 0.469),
            ]
        )
        out = order_change_cluster(hits)
        assert [h.doc_date.date().isoformat() for h in out][:2] == ["2026-03-23", "2026-03-16"]

    def test_a_lone_newest_note_is_a_change_in_the_latest_week(self) -> None:
        from mynotes.retrieval.rerank import change_cluster

        hits = self._notes([("2026-05-18", 0.9), ("2026-05-11", 0.6), ("2026-05-04", 0.59)])
        assert [h.doc_date.date().isoformat() for h in change_cluster(hits)] == ["2026-05-18"]


def test_resolved_query_replaces_the_bridge_description_with_the_entity() -> None:
    from mynotes.retrieval.rerank import resolved_query

    assert (
        resolved_query(
            "What is the licence of the tool that the notification service of Project Tua "
            "is built on?",
            "Memoranda",
        )
        == "What is the licence of the Memoranda?"
    )
    assert (
        resolved_query('Qual é o porto do plugin usado na nota "Diário - Xarrama"?', "Clipmark")
        == "Qual é o porto do Clipmark?"
    )
    assert resolved_query("Quem coordena o Projeto Alfa?", "Ana") is None
    assert resolved_query("Plugin usado na nota X?", "Clipmark") is None


def test_promotion_leaves_a_well_ranked_bridge_note_alone() -> None:
    hits = [
        _hit("top", "the component note"),
        _hit("tool", "the tool note", "Stanza"),
        _hit("other", "another tool note", "Pagewright"),
        _hit("x1", "look-alike"),
        _hit("weak1", "a team note", "equipa de plataforma"),
        _hit("weak2", "a tool mention", "Folios"),
    ]
    for i, (hit, score) in enumerate(
        zip(hits, [0.998, 0.988, 0.973, 0.5, 0.066, 0.065], strict=True)
    ):
        hit.document_id, hit.rerank_score, hit.rank = i + 1, score, i + 1
    hits[1].query_hint_type = hits[2].query_hint_type = hits[5].query_hint_type = "Technology"
    hits[4].query_hint_type = "Organization"
    out = promote_bridge_hit(hits, max_docs=4, preferred_types=frozenset({"Technology"}), slots=2)
    assert [h.chunk_id for h in out] == ["top", "tool", "other", "x1", "weak1", "weak2"]
    # one slot filled by the tool note: the free slot only takes another tool hint
    hits[2].query_hint = None
    out = promote_bridge_hit(hits, max_docs=4, preferred_types=frozenset({"Technology"}), slots=2)
    assert [h.chunk_id for h in out][:2] == ["top", "weak2"]
