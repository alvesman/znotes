"""Integration tests for the retrieval orchestrator (task 28).

The stack mirrors ``test_hybrid_search.py``: a temporary source root with the
same four-note corpus is scanned, drained through ``extract``/``chunk``,
embedded with a ``FakeEmbedder`` (dim 8), synced into per-test OpenSearch
aliases and searched through :func:`~mynotes.retrieval.pipeline.retrieve` with
a ``FakeReranker`` instead of the real reranker endpoint, so the whole
analyze → hybrid → expand → fuse → rerank path runs without a model endpoint.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from testcontainers.community.opensearch import OpenSearchContainer
from typer.testing import CliRunner

from fakes import FakeEmbedder, FakeReranker
from mynotes.cli import app
from mynotes.config import Settings
from mynotes.db.models import (
    Document,
    EmbeddingCache,
    File,
    IndexOutbox,
    Job,
    ScanRun,
    SourceRoot,
    Tag,
    TagAlias,
    Workspace,
)
from mynotes.db.session import reset_engine, session_scope
from mynotes.index.opensearch import ensure_indices, ensure_pipeline, get_client
from mynotes.index.sync import sync_once
from mynotes.jobs import pipeline
from mynotes.jobs.registry import JobContext
from mynotes.jobs.worker import Worker
from mynotes.retrieval.pipeline import register_expander, retrieve, unregister_expander
from mynotes.retrieval.types import SearchHit
from mynotes.scan.scanner import scan_root

pytestmark = pytest.mark.integration

_MTIME = 1_000_000_000_000_000_000
_JOB_TYPES = (
    "extract",
    "chunk",
    "embed",
    "apply_retention",
    "reindex_metadata",
    "delete_document",
    "version_detect",
    "tag_llm",
    "link_resolve",
    "kg_extract",
)

_KITCHEN_NOTE = """---
title: Kitchen Notes
date: 2026-07-10
---

# Kitchen

The kitchen inventory list and the maintenance calendar live in this folder.
"""

_PROJETO_X_NOTE = """---
title: Projeto X
date: 2026-08-20
---

# Projeto X

#projeto-x Começa em outubro com a migração do armazém de dados e um novo painel de indicadores.
"""

_ROADMAP_NOTE = """---
title: Roadmap
date: 2026-08-15
---

# Roadmap

The roadmap for the next quarter focuses on the storage engine and the search pipeline.
"""


def _frigorifico_note() -> str:
    """A PT note with one long section so the chunker yields several chunks."""
    paragraphs = "\n\n".join(
        f"O frigorífico número {number} da cozinha está com o termostato avariado "
        "e precisa de uma revisão técnica completa antes do verão chegar."
        for number in range(1, 121)
    )
    return (
        "---\ntitle: Manual do Frigorífico\ntags: [Trabalho]\ndate: 2026-09-01\n---\n\n"
        "# Frigorífico\n\n" + paragraphs + "\n"
    )


@dataclass
class Env:
    """The registered test root, the fakes and the test's aliases."""

    settings: Settings
    fake: FakeEmbedder
    reranker: FakeReranker
    root_id: int
    workspace_id: int
    root_path: Path
    chunks_alias: str


@pytest.fixture
def env(
    migrated_dsn: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    os_container: OpenSearchContainer,
) -> Iterator[Env]:
    """Container settings (dim 8, fast settle, grace 0) and per-test aliases."""
    settings = Settings(postgres={"dsn": migrated_dsn})
    config = os_container.get_config()
    suffix = uuid4().hex[:8]
    settings.opensearch.url = f"http://{config['host']}:{config['port']}"
    settings.opensearch.chunks_alias = f"t{suffix}_chunks"
    settings.opensearch.entities_alias = f"t{suffix}_entities"
    settings.models.embedding.dim = 8
    settings.scan.settle_seconds = 0
    settings.scan.delete_grace_seconds = 0
    settings.retrieval.min_rerank_score = 0.005
    for target in (
        "mynotes.config.get_settings",
        "mynotes.db.session.get_settings",
        "mynotes.scan.scanner.get_settings",
        "mynotes.jobs.worker.get_settings",
        "mynotes.index.sync.get_settings",
        "mynotes.index.opensearch.get_settings",
        "mynotes.retrieval.hybrid.get_settings",
        "mynotes.retrieval.analyze.get_settings",
        "mynotes.retrieval.pipeline.get_settings",
    ):
        monkeypatch.setattr(target, lambda: settings)
    fake = FakeEmbedder(dim=8)
    settings.models.embedding.model = fake.model
    monkeypatch.setattr("mynotes.retrieval.hybrid.get_embedder", lambda: fake)
    reranker = FakeReranker()
    monkeypatch.setattr("mynotes.retrieval.pipeline.get_reranker", lambda: reranker)
    reset_engine()
    with session_scope() as session:
        workspace = Workspace(name=f"retrieve-{suffix}")
        session.add(workspace)
        session.flush()
        root = SourceRoot(workspace_id=workspace.id, name="root", abs_path=str(tmp_path))
        session.add(root)
        session.flush()
        state = Env(
            settings=settings,
            fake=fake,
            reranker=reranker,
            root_id=root.id,
            workspace_id=workspace.id,
            root_path=tmp_path,
            chunks_alias=settings.opensearch.chunks_alias,
        )
    ensure_indices(get_client(), settings)
    ensure_pipeline(get_client(), settings)
    client = get_client()
    yield state
    for alias in (settings.opensearch.chunks_alias, settings.opensearch.entities_alias):
        try:
            physical = list(client.indices.get(index=f"{alias}_v*"))
        except Exception:
            physical = []
        for name in physical:
            client.indices.delete(index=name)
    with session_scope() as session:
        session.execute(delete(Job).where(Job.type.in_(_JOB_TYPES)))
        session.execute(delete(IndexOutbox).where(IndexOutbox.doc_kind == "chunk"))
        session.execute(delete(EmbeddingCache).where(EmbeddingCache.model == fake.model))
        session.execute(delete(File).where(File.source_root_id == state.root_id))
        session.execute(delete(Document).where(Document.source_root_id == state.root_id))
        session.execute(delete(ScanRun).where(ScanRun.source_root_id == state.root_id))
        session.execute(delete(SourceRoot).where(SourceRoot.id == state.root_id))
        session.execute(delete(TagAlias).where(TagAlias.workspace_id == state.workspace_id))
        session.execute(delete(Tag).where(Tag.workspace_id == state.workspace_id))
        session.execute(delete(Workspace).where(Workspace.id == state.workspace_id))
    reset_engine()


def write_file(path: Path, content: str, mtime_ns: int = _MTIME) -> None:
    """Create a file with a fixed mtime so scans are deterministic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))


def drain_extract_chunk() -> None:
    """Run the worker until no extract/chunk job is claimable."""
    worker = Worker(concurrency=1, poll_interval_s=0.05, types=["extract", "chunk"])
    while worker.run_once():
        pass


def refresh(env: Env) -> None:
    """Make the synced chunks visible to search."""
    get_client().indices.refresh(index=env.chunks_alias)


def ingest_corpus(env: Env) -> dict[str, int]:
    """Ingest the four-note corpus and return ``{rel_path: latest_version_id}``."""
    write_file(env.root_path / "folderA/frigorifico.md", _frigorifico_note())
    write_file(env.root_path / "folderA/kitchen.md", _KITCHEN_NOTE)
    write_file(env.root_path / "Projetos/projeto-x.md", _PROJETO_X_NOTE)
    write_file(env.root_path / "Projetos/roadmap.md", _ROADMAP_NOTE)
    scan_root(env.root_id)
    drain_extract_chunk()
    with session_scope() as session:
        rows = session.execute(
            select(File.rel_path, Document.latest_version_id)
            .join(Document, Document.current_file_id == File.id)
            .where(File.source_root_id == env.root_id)
        ).all()
        versions = {rel_path: version_id for rel_path, version_id in rows}
    for version_id in versions.values():
        pipeline.handle_embed(
            {"version_id": version_id},
            JobContext(worker_id="test", settings=env.settings, embedder=env.fake),
        )
    sync_once()
    refresh(env)
    return versions


def document_id_for(env: Env, rel_path: str) -> int:
    """The document currently stored for ``rel_path``."""
    with session_scope() as session:
        file_row = session.execute(
            select(File).where(File.source_root_id == env.root_id, File.rel_path == rel_path)
        ).scalar_one()
        assert file_row.document_id is not None
        return file_row.document_id


def mark_superseded(env: Env, document_id: int) -> None:
    """Flip a document to ``superseded`` and reindex it through the outbox."""
    with session_scope() as session:
        document = session.get(Document, document_id)
        assert document is not None
        document.status = "superseded"
    pipeline.handle_reindex_metadata(
        {"document_id": document_id},
        JobContext(worker_id="test", settings=env.settings, embedder=env.fake),
    )
    sync_once()
    refresh(env)


def _expander_hit(chunk_id: str, text: str) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id=9_999_999,
        version_id=9_999_999,
        rank=1,
        score=0.0,
        text=text,
        title=None,
        section=None,
        heading_path="",
        path="expandido/dummy.md",
        page=None,
        slide=None,
        doc_date=datetime(2026, 1, 1, tzinfo=UTC),
        doc_status="current",
        is_latest=True,
        is_table=False,
        format="md",
        language=None,
        tags=[],
    )


def test_retrieve_frigorifico_returns_expected_document_first(env: Env) -> None:
    """The frigorifico note wins after fusion and reranking."""
    ingest_corpus(env)

    result = retrieve("frigorifico")

    assert result.hits, "expected hits"
    assert result.hits[0].path == "folderA/frigorifico.md"
    assert result.analysis.text == "frigorifico"
    assert result.analysis.intent == "current"
    assert result.relevant is True
    assert result.candidates >= 1
    assert set(result.timings_ms) == {"analyze", "hybrid", "expand", "fuse", "rerank"}
    assert all(value >= 0 for value in result.timings_ms.values())
    assert [hit.rank for hit in result.hits] == list(range(1, len(result.hits) + 1))


def test_inline_filters_reach_the_hybrid_search(env: Env) -> None:
    """``#tag`` and ``in:`` in the query restrict the searched corpus."""
    ingest_corpus(env)

    result = retrieve('outubro #projeto-x in:"Projetos"')

    assert result.analysis.text == "outubro"
    assert result.analysis.filters.tags == ["projeto-x"]
    assert result.analysis.filters.folders == ["Projetos"]
    assert result.hits
    assert {hit.path for hit in result.hits} == {"Projetos/projeto-x.md"}


def test_history_intent_includes_superseded_documents(env: Env) -> None:
    """The history intent unlocks superseded documents."""
    ingest_corpus(env)
    roadmap_id = document_id_for(env, "Projetos/roadmap.md")
    mark_superseded(env, roadmap_id)

    result = retrieve("roadmap versoes")

    assert result.analysis.intent == "history"
    assert result.analysis.filters.include_superseded is True
    roadmap_hits = [hit for hit in result.hits if hit.path == "Projetos/roadmap.md"]
    assert roadmap_hits
    assert {hit.doc_status for hit in roadmap_hits} == {"superseded"}


def test_relevance_gate_uses_the_min_rerank_score(env: Env) -> None:
    """A high ``min_rerank_score`` reports ``relevant=False`` even with hits."""
    ingest_corpus(env)

    env.settings.retrieval.min_rerank_score = 0.99
    gated = retrieve("kitchen")

    assert gated.hits, "expected hits for 'kitchen'"
    assert all((hit.rerank_score or 0.0) < 0.99 for hit in gated.hits)
    assert gated.relevant is False

    env.settings.retrieval.min_rerank_score = 0.0
    ungated = retrieve("kitchen")
    assert ungated.relevant is True


def test_dummy_expander_hits_are_fused_with_their_source(env: Env) -> None:
    """A registered expander's candidates come back with ``sources["dummy"]``."""

    class DummyExpander:
        name = "dummy"

        def __init__(self) -> None:
            self.calls = 0

        def enabled(self, settings: Settings, analysis: object) -> bool:
            return True

        def expand(self, analysis: object, hybrid_hits: list[SearchHit]) -> list[SearchHit]:
            self.calls += 1
            return [_expander_hit("dummy-1", "O frigorifico esta avariado e precisa de revisao")]

    expander = DummyExpander()
    register_expander(expander)
    try:
        ingest_corpus(env)
        result = retrieve("frigorifico")
    finally:
        unregister_expander("dummy")

    assert expander.calls == 1
    dummy_hits = [hit for hit in result.hits if hit.chunk_id == "dummy-1"]
    assert len(dummy_hits) == 1
    assert dummy_hits[0].sources == {"dummy": 1}
    hybrid_hits = [hit for hit in result.hits if "hybrid" in hit.sources]
    assert hybrid_hits, "expected fused hybrid hits beside the dummy source"
    assert result.candidates >= 2
    assert result.relevant is True


def test_failing_expander_is_logged_and_skipped(env: Env) -> None:
    """An expander that raises must not break the retrieval."""

    class BoomExpander:
        name = "boom"

        def enabled(self, settings: Settings, analysis: object) -> bool:
            return True

        def expand(self, analysis: object, hybrid_hits: list[SearchHit]) -> list[SearchHit]:
            raise RuntimeError("boom")

    register_expander(BoomExpander())
    try:
        ingest_corpus(env)
        result = retrieve("frigorifico")
    finally:
        unregister_expander("boom")

    assert result.hits
    assert all("boom" not in hit.sources for hit in result.hits)


def test_cli_search_json_runs_the_full_pipeline(env: Env) -> None:
    """``mynotes search --json`` returns pipeline hits with rerank and sources."""
    ingest_corpus(env)

    result = CliRunner().invoke(app, ["search", "frigorifico", "--json", "-n", "5"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert isinstance(payload, list) and payload
    assert len(payload) <= 5
    first = payload[0]
    assert first["path"] == "folderA/frigorifico.md"
    assert first["rank"] == 1
    assert first["sources"]["hybrid"] >= 1
    assert first["rerank_score"] is not None
    assert SearchHit(**first)


def test_cli_search_raw_returns_plain_hybrid_hits(env: Env) -> None:
    """``--raw`` skips analysis, fusion and reranking."""
    ingest_corpus(env)

    result = CliRunner().invoke(app, ["search", "frigorifico", "--raw", "--json", "-n", "5"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload
    assert payload[0]["path"] == "folderA/frigorifico.md"
    assert payload[0]["rerank_score"] is None
    assert payload[0]["sources"] == {}
    assert SearchHit(**payload[0])


def test_cli_search_table_shows_rerank_and_sources_columns(env: Env) -> None:
    """The default table mode gained the Rerank and Sources columns."""
    ingest_corpus(env)

    result = CliRunner().invoke(app, ["search", "frigorifico", "-n", "3"], env={"COLUMNS": "300"})

    assert result.exit_code == 0, result.output
    assert "Rerank" in result.output
    assert "Sources" in result.output
