"""Unit tests for KG extraction: section groups, skips, validators, prompt (task 40)."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from mynotes.graph.extract import (
    ChunkGroup,
    ChunkRowFull,
    ExtractedEntity,
    ExtractedRelation,
    Extraction,
    build_groups,
    build_kg_messages,
    should_skip,
)
from mynotes.graph.ontology import ENTITY_TYPES, EXTRACTOR_VERSION, RELATION_TYPES
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
