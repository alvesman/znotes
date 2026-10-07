"""Integration tests for version review and version-aware retrieval (task 35).

The stack mirrors ``test_version_detection.py``: a temporary source root with
the ``tests/fixtures/trees/versions/`` fixture tree is scanned, drained through
``extract``/``chunk``, embedded with a ``FakeEmbedder`` (dim 8), synced into
per-test OpenSearch aliases and linked by the ``version_detect`` handler. The
review functions then run against the real database and the retrieval tests
drive the full pipeline (with a ``FakeReranker``) plus the context assembly,
so the version expander and the version-notes provider run end to end.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import delete, select, update
from testcontainers.community.opensearch import OpenSearchContainer
from typer.testing import CliRunner

from fakes import FakeEmbedder, FakeReranker
from mynotes.cli import app
from mynotes.config import Settings
from mynotes.db.models import (
    DocRelation,
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
from mynotes.retrieval import version_expand
from mynotes.retrieval.assemble import assemble_context
from mynotes.retrieval.pipeline import retrieve
from mynotes.scan.scanner import scan_root
from mynotes.versioning.review import add_manual, confirm, list_relations, reject, swap

pytestmark = pytest.mark.integration

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "trees" / "versions"
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
    "kg_temporal_refresh",
)


@dataclass
class ReviewEnv:
    """The registered test root, the fakes and the test's aliases."""

    settings: Settings
    fake: FakeEmbedder
    reranker: FakeReranker
    root_id: int
    workspace_id: int
    root_path: Path
    chunks_alias: str


@pytest.fixture
def review_env(
    migrated_dsn: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    os_container: OpenSearchContainer,
) -> Iterator[ReviewEnv]:
    """Container settings (dim 8, fast settle) and per-test OpenSearch aliases."""
    settings = Settings(postgres={"dsn": migrated_dsn})
    config = os_container.get_config()
    suffix = uuid4().hex[:8]
    settings.opensearch.url = f"http://{config['host']}:{config['port']}"
    settings.opensearch.chunks_alias = f"t{suffix}_chunks"
    settings.opensearch.entities_alias = f"t{suffix}_entities"
    settings.models.embedding.dim = 8
    settings.scan.settle_seconds = 0
    settings.scan.delete_grace_seconds = 0
    settings.versions.min_interval_hours = 0.0
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
        "mynotes.retrieval.version_expand.get_settings",
    ):
        monkeypatch.setattr(target, lambda: settings)
    fake = FakeEmbedder(dim=8)
    settings.models.embedding.model = fake.model
    monkeypatch.setattr("mynotes.retrieval.hybrid.get_embedder", lambda: fake)
    reranker = FakeReranker()
    monkeypatch.setattr("mynotes.retrieval.pipeline.get_reranker", lambda: reranker)
    reset_engine()
    with session_scope() as session:
        workspace = Workspace(name=f"review-{suffix}")
        session.add(workspace)
        session.flush()
        root = SourceRoot(workspace_id=workspace.id, name="root", abs_path=str(tmp_path))
        session.add(root)
        session.flush()
        env = ReviewEnv(
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
    version_expand.register()
    client = get_client()
    yield env
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
        session.execute(delete(File).where(File.source_root_id == env.root_id))
        session.execute(delete(Document).where(Document.source_root_id == env.root_id))
        session.execute(delete(ScanRun).where(ScanRun.source_root_id == env.root_id))
        session.execute(delete(SourceRoot).where(SourceRoot.id == env.root_id))
        session.execute(delete(TagAlias).where(TagAlias.workspace_id == env.workspace_id))
        session.execute(delete(Tag).where(Tag.workspace_id == env.workspace_id))
        session.execute(delete(Workspace).where(Workspace.id == env.workspace_id))
    reset_engine()


def install(env: ReviewEnv, rel_path: str, source: str, mtime_ns: int = _MTIME) -> None:
    """Write a fixture file into the root with a fixed mtime."""
    path = env.root_path / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((_FIXTURES / source).read_text(encoding="utf-8"), encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))


def drain(*job_types: str) -> None:
    """Run the worker until no job of the given types is claimable."""
    worker = Worker(concurrency=1, poll_interval_s=0.05, types=list(job_types))
    while worker.run_once():
        pass


def latest_version_id(env: ReviewEnv, rel_path: str) -> int:
    """The latest version id of the document stored for ``rel_path``."""
    with session_scope() as session:
        file_row = session.execute(
            select(File).where(File.source_root_id == env.root_id, File.rel_path == rel_path)
        ).scalar_one()
        assert file_row.document_id is not None
        document = session.get(Document, file_row.document_id)
        assert document is not None and document.latest_version_id is not None
        return int(document.latest_version_id)


def document_id_of(env: ReviewEnv, rel_path: str) -> int:
    """The document id stored for ``rel_path``."""
    with session_scope() as session:
        file_row = session.execute(
            select(File).where(File.source_root_id == env.root_id, File.rel_path == rel_path)
        ).scalar_one()
        assert file_row.document_id is not None
        return int(file_row.document_id)


def document_row(env: ReviewEnv, document_id: int) -> Document:
    """A detached document row for assertions."""
    with session_scope() as session:
        document = session.get(Document, document_id)
        assert document is not None
        session.expunge(document)
        return document


def ingest_pair(env: ReviewEnv) -> tuple[int, int]:
    """Ingest idea.md + idea_v2.md and link them with an auto supersedes relation.

    Returns:
        The ``(old_doc, new_doc)`` document ids.
    """
    install(env, "idea.md", "idea.md")
    install(env, "idea_v2.md", "idea_v2.md", _MTIME + 1_000_000_000)
    scan_root(env.root_id)
    drain("extract", "chunk")
    for rel_path in ("idea.md", "idea_v2.md"):
        pipeline.handle_embed(
            {"version_id": latest_version_id(env, rel_path)},
            JobContext(worker_id="test", settings=env.settings, embedder=env.fake),
        )
    drain("apply_retention")
    drain("version_detect")
    drain("reindex_metadata")
    sync_once()
    get_client().indices.refresh(index=env.chunks_alias)
    return document_id_of(env, "idea.md"), document_id_of(env, "idea_v2.md")


def indexed_doc_statuses(env: ReviewEnv, document_id: int) -> set[str]:
    """The ``doc_status`` values indexed for one document, after a refresh."""
    client = get_client()
    client.indices.refresh(index=env.chunks_alias)
    body = {
        "query": {"term": {"document_id": str(document_id)}},
        "_source": ["doc_status"],
        "size": 100,
    }
    result = client.search(index=env.chunks_alias, body=body)
    return {hit["_source"]["doc_status"] for hit in result["hits"]["hits"]}


def auto_relation_id(env: ReviewEnv) -> int:
    """The single auto supersedes relation of the pair, as a detached id."""
    with session_scope() as session:
        views = list_relations(session, status=["auto"])
        assert len(views) == 1
        return views[0].id


def test_add_manual_confirms_and_applies_effects(review_env: ReviewEnv) -> None:
    """add_manual records a confirmed manual relation with the chain effects."""
    env = review_env
    install(env, "report.md", "report.md")
    install(env, "report_v2.md", "report_v2.md", _MTIME + 1_000_000_000)
    scan_root(env.root_id)
    drain("extract", "chunk")
    for rel_path in ("report.md", "report_v2.md"):
        pipeline.handle_embed(
            {"version_id": latest_version_id(env, rel_path)},
            JobContext(worker_id="test", settings=env.settings, embedder=env.fake),
        )
    drain("apply_retention")
    drain("reindex_metadata")
    sync_once()
    old_doc = document_id_of(env, "report.md")
    new_doc = document_id_of(env, "report_v2.md")

    with session_scope() as session:
        relation_id = add_manual(session, "report_v2.md", "report.md")
    drain("reindex_metadata")
    sync_once()

    with session_scope() as session:
        view = next(v for v in list_relations(session) if v.id == relation_id)
        assert view.status == "confirmed"
        assert view.method == "manual"
        assert view.confidence == 1.0
        assert view.newer.document_id == new_doc
        assert view.older.document_id == old_doc
    assert document_row(env, old_doc).status == "superseded"
    assert indexed_doc_statuses(env, old_doc) == {"superseded"}

    with pytest.raises(ValueError, match="same document"), session_scope() as session:
        add_manual(session, "report.md", "report.md")
    with pytest.raises(ValueError, match="no document found"), session_scope() as session:
        add_manual(session, "missing.md", "report.md")


def test_list_reject_and_confirm_round_trip(review_env: ReviewEnv) -> None:
    """list_relations shows the pair; reject/confirm toggle the old document."""
    env = review_env
    old_doc, new_doc = ingest_pair(env)
    assert document_row(env, old_doc).status == "superseded"

    with session_scope() as session:
        views = list_relations(session, status=["auto"])
    assert len(views) == 1
    view = views[0]
    assert view.id == auto_relation_id(env)
    assert view.newer.document_id == new_doc
    assert view.newer.path == "idea_v2.md"
    assert view.older.document_id == old_doc
    assert view.older.path == "idea.md"
    assert view.method == "name_content"
    assert view.evidence["order_reason"] == "version_number"
    assert view.evidence["name_signal"] is True
    assert "jaccard" in view.evidence and "cosine" in view.evidence

    relation_id = view.id
    with session_scope() as session:
        reject(session, relation_id)
    drain("reindex_metadata")
    sync_once()
    assert document_row(env, old_doc).status == "current"
    assert document_row(env, old_doc).superseded_by_document_id is None
    assert indexed_doc_statuses(env, old_doc) == {"current"}

    with session_scope() as session:
        confirm(session, relation_id)
    drain("reindex_metadata")
    sync_once()
    assert document_row(env, old_doc).status == "superseded"
    assert document_row(env, old_doc).superseded_by_document_id == new_doc
    assert indexed_doc_statuses(env, old_doc) == {"superseded"}


def test_swap_reverses_the_direction_and_confirms(review_env: ReviewEnv) -> None:
    """After a swap, idea.md is the successor and idea_v2.md is superseded."""
    env = review_env
    old_doc, new_doc = ingest_pair(env)
    relation_id = auto_relation_id(env)

    with session_scope() as session:
        swap(session, relation_id)
    drain("reindex_metadata")
    sync_once()

    with session_scope() as session:
        relation = session.get(DocRelation, relation_id)
        assert relation is not None
        assert relation.src_document_id == old_doc
        assert relation.dst_document_id == new_doc
        assert relation.status == "confirmed"
    assert document_row(env, old_doc).status == "current"
    assert document_row(env, new_doc).status == "superseded"
    assert document_row(env, new_doc).superseded_by_document_id == old_doc
    assert indexed_doc_statuses(env, new_doc) == {"superseded"}
    assert indexed_doc_statuses(env, old_doc) == {"current"}


def test_current_retrieval_prefers_the_head_and_notes_the_version(
    review_env: ReviewEnv,
) -> None:
    """Archived old file: head chunks rank first and the context carries the note."""
    env = review_env
    old_doc, _ = ingest_pair(env)
    with session_scope() as session:
        session.execute(update(Document).where(Document.id == old_doc).values(status="archived"))
    pipeline.handle_reindex_metadata(
        {"document_id": old_doc},
        JobContext(worker_id="test", settings=env.settings, embedder=env.fake),
    )
    sync_once()
    get_client().indices.refresh(index=env.chunks_alias)

    result = retrieve("projeto farol ideia inicial")

    assert result.hits
    new_ranks = [hit.rank for hit in result.hits if hit.path == "idea_v2.md"]
    old_ranks = [hit.rank for hit in result.hits if hit.path == "idea.md"]
    assert new_ranks and old_ranks
    assert min(new_ranks) < min(old_ranks)
    assert any(hit.sources.get("version") for hit in result.hits)

    bundle = assemble_context(result)
    assert "### Version notes" in bundle.text
    assert "supersedes" in bundle.text


def test_history_retrieval_includes_both_documents_and_marks_historical(
    review_env: ReviewEnv,
) -> None:
    """History intent returns both chain members; the context marks the old one."""
    env = review_env
    ingest_pair(env)

    result = retrieve("what changed in the idea before?")

    assert result.analysis.intent == "history"
    paths = {hit.path for hit in result.hits}
    assert "idea.md" in paths
    assert "idea_v2.md" in paths
    old_hits = [hit for hit in result.hits if hit.path == "idea.md"]
    assert {hit.doc_status for hit in old_hits} == {"superseded"}
    assert "### Version notes" in assemble_context(result).text


def test_history_context_marks_the_old_version_historical(review_env: ReviewEnv) -> None:
    """With history intent the context header of the old document says historical.

    The plan pair is used because plan-2026-03.md holds a line the successor
    dropped, so the old document's chunk survives the assembler's identical-
    prefix dedup and its attribution group is rendered.
    """
    env = review_env
    install(env, "plan-2026-03.md", "plan-2026-03.md")
    install(env, "plan-2026-09.md", "plan-2026-09.md", _MTIME + 1_000_000_000)
    scan_root(env.root_id)
    drain("extract", "chunk")
    for rel_path in ("plan-2026-03.md", "plan-2026-09.md"):
        pipeline.handle_embed(
            {"version_id": latest_version_id(env, rel_path)},
            JobContext(worker_id="test", settings=env.settings, embedder=env.fake),
        )
    drain("apply_retention")
    drain("version_detect")
    drain("reindex_metadata")
    sync_once()
    get_client().indices.refresh(index=env.chunks_alias)

    result = retrieve("o que mudou no módulo final antes?")

    assert result.analysis.intent == "history"
    paths = {hit.path for hit in result.hits}
    assert "plan-2026-03.md" in paths
    assert "plan-2026-09.md" in paths

    bundle = assemble_context(result)
    assert "### Version notes" in bundle.text
    assert 'supersedes "plan-2026-03.md"' in bundle.text
    historical = [
        line
        for line in bundle.text.splitlines()
        if line.startswith("[") and "plan-2026-03.md" in line
    ]
    assert historical
    assert all(line.endswith("historical") for line in historical)


def test_cli_versions_list_json(review_env: ReviewEnv) -> None:
    """``mynotes versions list --json`` prints the relations as JSON."""
    ingest_pair(review_env)

    result = CliRunner().invoke(app, ["versions", "list", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert isinstance(payload, list) and len(payload) == 1
    relation = payload[0]
    assert relation["status"] == "auto"
    assert relation["newer"]["path"] == "idea_v2.md"
    assert relation["older"]["path"] == "idea.md"
    assert relation["evidence"]["order_reason"] == "version_number"
