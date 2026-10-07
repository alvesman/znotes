"""Unit tests for the OpenSearch document builders (task 22).

The builders are pure helpers, so these tests need neither Postgres nor
OpenSearch; the path and heading rules of PLAN.md §6 are exercised directly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from mynotes.index.documents import chunk_source, heading_path_text
from mynotes.platform.paths import path_ancestors, path_ancestors_folded


def build_source(rel_path: str, heading_path: list[str]) -> dict:
    """A minimal ``_source`` with the path and heading fields under test."""
    return chunk_source(
        chunk_id=uuid4(),
        document_id=17,
        version_id=3,
        source_root_id=5,
        workspace_id=2,
        text="corpo do chunk",
        title="Relatório Trimestral",
        heading_path=heading_path,
        section="Resultados",
        tags=["trabalho", "area-tecnica"],
        format="md",
        language="pt",
        doc_status="current",
        rel_path=rel_path,
        doc_date=datetime(2026, 9, 1, 12, 0, tzinfo=UTC),
        is_latest=True,
        is_table=False,
        page=None,
        slide=None,
        ordinal=4,
        embedding=[0.5, -0.5],
    )


def test_path_ancestors_are_exact_folder_prefixes() -> None:
    """``path_ancestors`` keeps the on-disk case (ACL stays exact, PLAN §6.1)."""
    assert path_ancestors("Projetos/Ação/x.md") == ["Projetos", "Projetos/Ação"]


def test_path_ancestors_folded_are_normalized_folder_keys() -> None:
    """``path_ancestors_folded`` folds each prefix for user-facing filters."""
    assert path_ancestors_folded("Projetos/Ação/x.md") == ["projetos", "projetos/acao"]


def test_chunk_source_carries_exact_and_folded_paths() -> None:
    """The document carries the exact path plus both ancestor variants."""
    source = build_source("Projetos/Ação/x.md", ["Introdução", "Contexto"])
    assert source["path"] == "Projetos/Ação/x.md"
    assert source["path_ancestors"] == ["Projetos", "Projetos/Ação"]
    assert source["path_ancestors_folded"] == ["projetos", "projetos/acao"]


def test_chunk_source_root_level_file_has_no_ancestors() -> None:
    """A file directly under the root has empty ancestor lists."""
    source = build_source("notas.md", [])
    assert source["path_ancestors"] == []
    assert source["path_ancestors_folded"] == []


def test_heading_path_text_joins_with_angle_bracket() -> None:
    """The heading path is joined with ``" > "`` and the empty path is ``""``."""
    assert heading_path_text(["Introdução", "Contexto"]) == "Introdução > Contexto"
    assert heading_path_text([]) == ""
    source = build_source("x.md", ["Introdução", "Contexto"])
    assert source["heading_path_text"] == "Introdução > Contexto"


def test_chunk_source_field_types_match_the_mapping() -> None:
    """Ids become strings, doc_date becomes ISO and embeddings pass through."""
    source = build_source("x.md", ["Título"])
    assert isinstance(source["chunk_id"], str)
    assert isinstance(source["document_id"], str)
    assert isinstance(source["version_id"], str)
    assert isinstance(source["source_root_id"], str)
    assert isinstance(source["workspace_id"], str)
    assert source["doc_date"] == "2026-09-01T12:00:00+00:00"
    assert source["embedding"] == [0.5, -0.5]
    assert source["is_latest"] is True
    assert source["is_table"] is False
    assert source["page"] is None and source["slide"] is None
    assert source["ordinal"] == 4


def test_chunk_source_doc_date_none_stays_none() -> None:
    """A document without a date indexes ``doc_date`` as null."""
    source = chunk_source(
        chunk_id="c",
        document_id=1,
        version_id=1,
        source_root_id=1,
        workspace_id=1,
        text="t",
        title=None,
        heading_path=[],
        section=None,
        tags=[],
        format="txt",
        language=None,
        doc_status="current",
        rel_path="x.txt",
        doc_date=None,
        is_latest=True,
        is_table=False,
        page=None,
        slide=None,
        ordinal=0,
        embedding=[0.0],
    )
    assert source["doc_date"] is None
    assert source["title"] is None
    assert source["heading_path_text"] == ""
