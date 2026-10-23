"""Unit tests for KG extraction: section groups, skips, validators, prompt (task 40)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import uuid4

import pytest

from fakes import FakeChat
from mynotes.graph import extract as extract_module
from mynotes.graph.extract import (
    ChunkGroup,
    ChunkRowFull,
    ExtractedEntity,
    ExtractedRelation,
    Extraction,
    build_groups,
    build_kg_messages,
    extract_group,
    merge_extractions,
    should_skip,
    split_group,
)
from mynotes.graph.ontology import ENTITY_TYPES, EXTRACTOR_VERSION, RELATION_TYPES
from mynotes.llm.chat import ChatMessage
from mynotes.scan.hashing import hash_text


def row(
    text: str,
    *,
    section: str | None = "Ideia",
    is_table: bool = False,
    page: int | None = None,
    slide: int | None = None,
    tokens: int | None = None,
) -> ChunkRowFull:
    """A chunk row with an explicit token count (no tokenizer involved)."""
    return ChunkRowFull(
        id=uuid4(),
        text=text,
        section=section,
        is_table=is_table,
        page=page,
        slide=slide,
        token_count=tokens if tokens is not None else len(text.split()),
    )


def group(
    text: str,
    *,
    section: str | None = None,
    is_table: bool = False,
    tokens: int | None = None,
) -> ChunkGroup:
    """A hand-built group for the skip and prompt tests."""
    return ChunkGroup(
        chunk_ids=[uuid4()],
        section=section,
        text=text,
        content_hash=hash_text(text + "\x1f" + EXTRACTOR_VERSION),
        token_count=tokens if tokens is not None else len(text.split()),
        is_table=is_table,
    )


def test_build_groups_merges_same_section_up_to_budget() -> None:
    """Same-section chunks merge while the total fits; the next one starts a group."""
    chunks = [row("one", tokens=600), row("two", tokens=600), row("three", tokens=600)]

    groups = build_groups(chunks, max_tokens=1500)

    assert [group.chunk_ids for group in groups] == [
        [chunks[0].id, chunks[1].id],
        [chunks[2].id],
    ]
    assert [group.token_count for group in groups] == [1200, 600]
    assert groups[0].text == "one\n\ntwo"
    assert groups[0].content_hash == hash_text("one\n\ntwo" + "\x1f" + EXTRACTOR_VERSION)


def test_build_groups_split_on_section_change() -> None:
    """A section change always starts a new group, even under the budget."""
    chunks = [row("a", section="A", tokens=10), row("b", section="B", tokens=10)]

    groups = build_groups(chunks, max_tokens=1500)

    assert [(group.section, group.text) for group in groups] == [("A", "a"), ("B", "b")]


def test_build_groups_split_when_page_run_changes() -> None:
    """PDF chunks merge within a page and split across pages; slides behave the same."""
    chunks = [
        row("p1", section="A", page=1, tokens=10),
        row("p1b", section="A", page=1, tokens=10),
        row("p2", section="A", page=2, tokens=10),
    ]
    slides = [
        row("s1", section=None, slide=1, tokens=10),
        row("s2", section=None, slide=2, tokens=10),
    ]

    groups = build_groups(chunks, max_tokens=1500)
    slide_groups = build_groups(slides, max_tokens=1500)

    assert [group.chunk_ids for group in groups] == [
        [chunks[0].id, chunks[1].id],
        [chunks[2].id],
    ]
    assert [group.chunk_ids for group in slide_groups] == [[slides[0].id], [slides[1].id]]


def test_build_groups_tables_are_own_groups() -> None:
    """Every table chunk is a single-chunk group between the text groups."""
    chunks = [
        row("para one", section="A", tokens=100),
        row("| a | b |", section="A", is_table=True, tokens=50),
        row("| c | d |", section="A", is_table=True, tokens=50),
        row("para two", section="A", tokens=100),
    ]

    groups = build_groups(chunks, max_tokens=1500)

    assert len(groups) == 4
    assert [group.is_table for group in groups] == [False, True, True, False]
    assert all(len(group.chunk_ids) == 1 for group in groups[1:3])
    assert groups[1].chunk_ids == [chunks[1].id]
    assert groups[2].chunk_ids == [chunks[2].id]


def test_should_skip_short_and_toc_and_dotted_lines() -> None:
    """Short groups, TOC/legal sections and leader-dot text are skipped."""
    assert should_skip(group("x", tokens=39)) == "skip"
    assert should_skip(group("sumário do documento", section="Índice", tokens=100)) == "skip"
    assert should_skip(group("legal notes", section="Copyright", tokens=100)) == "skip"
    dotted = "\n".join(
        [
            "Introduction ......... 12",
            "Methods .............. 30",
            "Results .............. 55",
            "one line of body text",
        ]
    )
    assert should_skip(group(dotted, tokens=100)) == "skip"


def test_should_skip_boundary_lines_below_threshold() -> None:
    """30% dotted lines or fewer is not boilerplate."""
    lines = ["Introduction ......... 12", "real body line one", "real body line two", "third"]
    assert should_skip(group("\n".join(lines), tokens=100)) == "full"


def test_should_skip_table_and_full() -> None:
    """Table groups are entities_only; a normal group is full."""
    assert should_skip(group("| a | b |", section="Dados", is_table=True, tokens=100)) == (
        "entities_only"
    )
    assert should_skip(group("texto normal do projeto", section="Ideias", tokens=100)) == "full"


def test_unknown_entity_type_becomes_concept() -> None:
    """An entity type outside the ontology maps to Concept."""

    entity = ExtractedEntity(name="Alfa", type="Gadget")

    assert entity.type == "Concept"
    assert ExtractedEntity(name="Alfa", type="Risk").type == "Risk"


def test_unknown_relation_type_maps_and_keeps_raw_label() -> None:
    """An unknown relation type becomes related_to; the raw label is kept."""

    relation = ExtractedRelation(
        source="Alfa", target="Beta", type="financia", raw_label="financia"
    )

    assert relation.type == "related_to"
    assert relation.raw_label == "financia"
    assert ExtractedRelation(source="A", target="B", type="uses").type == "uses"


def test_relation_with_unknown_endpoint_is_dropped() -> None:
    """Relations whose source or target is not an entity name are dropped."""

    extraction = Extraction.model_validate(
        {
            "entities": [{"name": "Projeto Alfa", "type": "Project"}],
            "relations": [
                {"source": "Projeto Alfa", "target": "projeto alfa", "type": "part_of"},
                {"source": "Ghost", "target": "Projeto Alfa", "type": "uses"},
                {"source": "Projeto Alfa", "target": "Ghost", "type": "uses"},
            ],
        }
    )

    assert [(relation.source, relation.target) for relation in extraction.relations] == [
        ("Projeto Alfa", "projeto alfa")
    ]


def test_prompt_contains_context_and_known_entities() -> None:
    """The user message carries title, path, date, tags, section, entities and text."""
    kg_group = group("texto da seção sobre o projeto", section="Ideias", tokens=100)

    messages = build_kg_messages(
        kg_group,
        title="Plano Alfa",
        rel_path="projects/alfa/plano.md",
        doc_date=datetime(2026, 9, 1),
        tags=["project:alfa"],
        known_entities=[("Projeto Alfa", "Project"), ("Ana", "Person")],
        mode="full",
    )

    assert [message.role for message in messages] == ["system", "user"]
    system = messages[0].content
    user = messages[1].content
    assert all(entity_type in system for entity_type in ENTITY_TYPES)
    assert all(relation_type in system for relation_type in RELATION_TYPES)
    assert "Only facts stated in the text." in system
    assert "without indentation or line breaks" in system
    assert "at most 30 entities and 40 relations" in system
    assert "Plano Alfa" in user
    assert "projects/alfa/plano.md" in user
    assert "2026-09-01" in user
    assert "project:alfa" in user
    assert "Section: Ideias" in user
    assert "Projeto Alfa (Project)" in user
    assert "Ana (Person)" in user
    assert "texto da seção sobre o projeto" in user
    assert "Return entities only" not in user


def test_prompt_entities_only_instruction_and_known_entity_cap() -> None:
    """entities_only mode adds the instruction; at most 50 known entities are sent."""
    kg_group = group("tabela de dados", section="Dados", tokens=100)
    known = [(f"Entity {number}", "Concept") for number in range(60)]

    messages = build_kg_messages(
        kg_group,
        title=None,
        rel_path="notes/tabela.md",
        doc_date=None,
        tags=[],
        known_entities=known,
        mode="entities_only",
    )

    user = messages[1].content
    assert "Return entities only; relations must be an empty list." in user
    assert "Entity 49 (Concept)" in user
    assert "Entity 50 (Concept)" not in user
    assert "Title: (untitled)" in user


class FakeSession:
    """Just enough of a session for ``extract_group``: an empty cache."""

    def get(self, model: Any, key: Any) -> None:
        return None

    def execute(self, statement: Any) -> None:
        return None


def _entity_json(*names: tuple[str, str]) -> str:
    return json.dumps(
        {"entities": [{"name": name, "type": kind} for name, kind in names], "relations": []}
    )


def _cache_recorder(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict]]:
    written: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        extract_module,
        "_cache_result",
        lambda session, content_hash, model, result: written.append((content_hash, result)),
    )
    return written


def test_split_group_halves_at_paragraphs() -> None:
    """The split lands on a blank line, close to the middle, and keeps the ids."""
    kg_group = group("um dois\n\ntres quatro cinco\n\nseis\n\nsete oito", tokens=80)
    halves = split_group(kg_group)
    assert [half.text for half in halves] == ["um dois\n\ntres quatro cinco", "seis\n\nsete oito"]
    assert all(half.chunk_ids == kg_group.chunk_ids for half in halves)
    assert sum(half.token_count for half in halves) == 80
    assert len({half.content_hash for half in halves} | {kg_group.content_hash}) == 3


def test_split_group_falls_back_to_lines_or_gives_up() -> None:
    """Without blank lines the split uses line breaks; one paragraph is not split."""
    assert [half.text for half in split_group(group("a b\nc d\ne f"))] == ["a b\nc d", "e f"]
    single = group("uma só linha sem quebras")
    assert split_group(single) == [single]


def test_merge_extractions_dedupes_and_caps() -> None:
    """Entities repeat by name and type only once; relations by their key."""
    alpha = ExtractedEntity(name="Alpha", type="Project")
    beta = ExtractedEntity(name="Beta", type="Technology")
    relation = ExtractedRelation(source="Alpha", target="Beta", type="uses")
    merged = merge_extractions(
        [
            Extraction(entities=[alpha], relations=[]),
            Extraction(
                entities=[beta, ExtractedEntity(name="alpha", type="Project")], relations=[]
            ),
            Extraction(entities=[alpha, beta], relations=[relation, relation]),
        ]
    )
    assert [entity.name for entity in merged.entities] == ["Alpha", "Beta"]
    assert len(merged.relations) == 1
    many = [ExtractedEntity(name=f"E{i}", type="Concept") for i in range(60)]
    assert (
        len(
            merge_extractions(
                [Extraction(entities=many[:40]), Extraction(entities=many[40:])]
            ).entities
        )
        == 40
    )


def test_failed_group_is_extracted_in_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A truncated answer for the whole group leads to one call per half, merged and cached."""
    written = _cache_recorder(monkeypatch)
    kg_group = group("Alpha is the project.\n\nBeta is the tool it uses.", tokens=100)

    def answer(messages: Sequence[ChatMessage]) -> str:
        text = messages[1].content.split("Text:\n", 1)[1]
        if "Alpha" in text and "Beta" in text:
            return '{"entities": [{"name": "Alpha", "type": "Project"}, {"name": "Be'
        if "Alpha" in text:
            return _entity_json(("Alpha", "Project"))
        return _entity_json(("Beta", "Technology"))

    chat = FakeChat(answer)
    extraction = extract_group(FakeSession(), chat, kg_group, {"title": "t"}, "full")

    assert [(entity.name, entity.type) for entity in extraction.entities] == [
        ("Alpha", "Project"),
        ("Beta", "Technology"),
    ]
    assert len(chat.calls) == 4  # whole group + repair, then one call per half
    assert (
        "Alpha (Project)" in chat.calls[3][1].content
    )  # the second half knows the first's entities
    halves = split_group(kg_group)
    assert [content_hash for content_hash, _ in written] == [
        halves[0].content_hash,
        halves[1].content_hash,
        kg_group.content_hash,
    ]
    assert len(written[2][1]["entities"]) == 2
    assert "partial" not in written[2][1]


def test_group_with_a_failed_part_is_cached_as_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    """When one half keeps failing, the group keeps the other half's result, marked partial."""
    written = _cache_recorder(monkeypatch)
    kg_group = group("Alpha is the project.\n\nBeta fails.\n\nGamma fails too.", tokens=120)

    def answer(messages: Sequence[ChatMessage]) -> str:
        text = messages[1].content.split("Text:\n", 1)[1]
        if text.strip() == "Alpha is the project.":
            return _entity_json(("Alpha", "Project"))
        return "not json"

    extraction = extract_group(FakeSession(), FakeChat(answer), kg_group, {}, "full")

    assert [entity.name for entity in extraction.entities] == ["Alpha"]
    results = dict(written)
    assert results[kg_group.content_hash].get("partial") is True
    halves = split_group(kg_group)
    assert "partial" not in results[halves[0].content_hash]
    assert list(results[halves[1].content_hash]) == ["error"]  # both its quarters failed


def test_unsplittable_failure_is_cached_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """When even the smallest parts fail, the group caches an error and yields nothing."""
    written = _cache_recorder(monkeypatch)
    kg_group = group("\n\n".join(f"paragrafo {i}" for i in range(8)), tokens=200)
    chat = FakeChat(lambda messages: "not json at all")

    extraction = extract_group(FakeSession(), chat, kg_group, {}, "full")

    assert extraction.entities == [] and extraction.relations == []
    assert len(chat.calls) == 2 * (1 + 2 + 4)  # every node of the split tree, with its repair
    assert written[-1][0] == kg_group.content_hash
    assert all("error" in result for _, result in written)  # the 4 leaves and the group
    assert len(written) == 7  # 4 leaves, 2 halves whose parts all failed, the group
