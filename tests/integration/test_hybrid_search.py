"""Integration tests for the hybrid search and the ``search`` CLI (task 24).

The setup mirrors ``test_outbox_sync.py``: a temporary source root with a
small corpus (four notes in two folders, PT and EN) is scanned, drained
through ``extract``/``chunk`` and embedded with a ``FakeEmbedder`` (dim 8);
``sync_once`` pushes the chunks into per-test OpenSearch aliases and the RRF
pipeline is upserted once per test (idempotent, cluster-global). Assertions
go through :func:`~mynotes.retrieval.hybrid.hybrid_search` and, for the CLI,
the ``CliRunner`` against the same stack, so the whole
scan → index → search path runs without a model endpoint.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select
from testcontainers.community.opensearch import OpenSearchContainer
from typer.testing import CliRunner

from fakes import FakeEmbedder, FakeReranker
from mynotes.cli import app
from mynotes.config import Settings
from mynotes.db.models import (
    Chunk,
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
from mynotes.retrieval.hybrid import hybrid_search
from mynotes.retrieval.types import SearchFilters, SearchHit
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
    """The registered test root, the fake embedder and the test's aliases."""

    settings: Settings
    fake: FakeEmbedder
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
    monkeypatch.setattr("mynotes.config.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.db.session.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.scan.scanner.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.jobs.worker.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.index.sync.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.index.opensearch.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.retrieval.hybrid.get_settings", lambda: settings)
    fake = FakeEmbedder(dim=8)
    settings.models.embedding.model = fake.model
    monkeypatch.setattr("mynotes.retrieval.hybrid.get_embedder", lambda: fake)
    # The search CLI now reranks through retrieve(); keep the task-24 tests
    # hermetic with the fake reranker instead of the real endpoint.
    monkeypatch.setattr("mynotes.retrieval.pipeline.get_reranker", lambda: FakeReranker())
    reset_engine()
    with session_scope() as session:
        workspace = Workspace(name=f"search-{suffix}")
        session.add(workspace)
        session.flush()
        root = SourceRoot(workspace_id=workspace.id, name="root", abs_path=str(tmp_path))
        session.add(root)
        session.flush()
        state = Env(
            settings=settings,
            fake=fake,
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
    assert set(versions) == {
        "folderA/frigorifico.md",
        "folderA/kitchen.md",
        "Projetos/projeto-x.md",
        "Projetos/roadmap.md",
    }
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


def chunk_count(env: Env, document_id: int) -> int:
    """The number of stored chunks of a document."""
    with session_scope() as session:
        return int(
            session.execute(
                select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
            ).scalar_one()
        )


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


def test_frigorifico_query_ranks_note_first_across_spellings(env: Env) -> None:
    """The Frigorífico note wins, and all spellings return the same top-3."""
    ingest_corpus(env)

    hits = hybrid_search("frigorifico", size=30)

    assert hits, "expected hits for 'frigorifico'"
    assert hits[0].path == "folderA/frigorifico.md"
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))
    top3 = [hit.chunk_id for hit in hits[:3]]
    assert len(top3) == 3
    for variant in ("FRIGORÍFICO", "Frigorífico"):
        again = hybrid_search(variant, size=30)
        assert [hit.chunk_id for hit in again[:3]] == top3


def test_tag_filter_returns_only_tagged_note(env: Env) -> None:
    """``tags=["projeto-x"]`` restricts the result to that note's chunks."""
    ingest_corpus(env)

    hits = hybrid_search("outubro", SearchFilters(tags=["projeto-x"]), size=10)

    assert hits, "expected hits for the tagged note"
    assert {hit.path for hit in hits} == {"Projetos/projeto-x.md"}


def test_folder_filter_matches_folded_prefix(env: Env) -> None:
    """Folder filters fold their values and hit ``path_ancestors_folded``."""
    ingest_corpus(env)

    for spelling in ("Projetos", "PROJETOS"):
        hits = hybrid_search("frigorifico", SearchFilters(folders=[spelling]), size=10)
        assert hits, f"expected hits with folders={spelling!r}"
        assert {hit.path for hit in hits} <= {"Projetos/projeto-x.md", "Projetos/roadmap.md"}

    hits = hybrid_search("frigorifico", SearchFilters(folders=["folderA"]), size=10)
    assert hits
    assert all(hit.path.startswith("folderA/") for hit in hits)


def test_superseded_document_needs_history_intent(env: Env) -> None:
    """A superseded document is excluded by default and returned with intent."""
    ingest_corpus(env)
    roadmap_id = document_id_for(env, "Projetos/roadmap.md")
    assert chunk_count(env, roadmap_id) > 0
    mark_superseded(env, roadmap_id)

    default = hybrid_search("roadmap", size=10)
    history = hybrid_search("roadmap", SearchFilters(include_superseded=True), size=10)

    assert all(hit.path != "Projetos/roadmap.md" for hit in default)
    roadmap_hits = [hit for hit in history if hit.path == "Projetos/roadmap.md"]
    assert roadmap_hits, "expected the superseded document with history intent"
    assert {hit.doc_status for hit in roadmap_hits} == {"superseded"}


def test_acl_prefixes_are_exact_and_case_sensitive(env: Env) -> None:
    """ACL matches the exact fields; a different case has no access at all."""
    ingest_corpus(env)

    allowed = hybrid_search("frigorifico", SearchFilters(acl_prefixes=["folderA"]), size=10)
    assert allowed
    assert all(hit.path.startswith("folderA/") for hit in allowed)

    denied = hybrid_search("frigorifico", SearchFilters(acl_prefixes=["FolderA"]), size=10)
    assert denied == []

    other_workspace = hybrid_search(
        "frigorifico", SearchFilters(workspace_id=env.workspace_id + 1), size=10
    )
    assert other_workspace == []
    own_workspace = hybrid_search(
        "frigorifico", SearchFilters(workspace_id=env.workspace_id), size=10
    )
    assert own_workspace


def test_dedup_keeps_at_most_two_hits_per_group(env: Env) -> None:
    """A document with many matching chunks in one section yields at most 2."""
    ingest_corpus(env)
    frigorifico_id = document_id_for(env, "folderA/frigorifico.md")
    stored = chunk_count(env, frigorifico_id)
    assert stored > 2, "the long section must produce more than 2 chunks"

    hits = hybrid_search("frigorifico", size=20)

    groups: dict[tuple, int] = {}
    for hit in hits:
        key = (hit.document_id, hit.page, hit.slide, hit.section)
        groups[key] = groups.get(key, 0) + 1
    assert groups, "expected hits"
    assert all(count <= 2 for count in groups.values())
    assert groups.get((frigorifico_id, None, None, "Frigorífico"), 0) <= 2


def test_cli_search_json_prints_valid_json(env: Env) -> None:
    """``mynotes search --json`` prints the hits as a JSON list."""
    ingest_corpus(env)

    result = CliRunner().invoke(app, ["search", "frigorifico", "--json", "-n", "5"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert isinstance(payload, list) and payload
    assert len(payload) <= 5
    first = payload[0]
    assert first["path"] == "folderA/frigorifico.md"
    assert first["rank"] == 1
    assert isinstance(first["document_id"], int)
    assert first["doc_date"] == "2026-09-01T00:00:00+00:00"
    assert SearchHit(**first)
