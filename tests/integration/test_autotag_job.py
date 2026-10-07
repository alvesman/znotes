"""Integration tests for the LLM auto-tagging job and its review CLI (task 38).

The setup mirrors ``test_tags.py``: a temporary source root is ingested with a
``FakeEmbedder`` (dim 8) and the ``tag_llm`` handler runs with a ``FakeChat``
returning fixed proposals, so no model endpoint is touched. One note carries a
frontmatter ``ML`` tag so the proposal ``ml`` already exists in the vocabulary
while ``gardening``/``rust`` do not.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from testcontainers.community.opensearch import OpenSearchContainer
from typer.testing import CliRunner

from fakes import FakeChat, FakeEmbedder
from mynotes.cli import app
from mynotes.config import FolderRule, Settings
from mynotes.db.models import (
    Chunk,
    Document,
    DocumentTag,
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
from mynotes.scan.scanner import scan_root
from mynotes.tags.autotag import decide_proposal

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

_FOLDER_RULES = [FolderRule(pattern="projects/{project}/**", tags=["project:{project}"])]

_NOTE = """---
title: Nota Alfa
date: 2026-09-01
---

# Ideia

Anotações sobre machine learning e busca híbrida no projeto.
"""

_NOTE_V2 = """---
title: Nota Alfa
date: 2026-09-03
---

# Ideia

Anotações revisadas sobre machine learning, jardineria e busca híbrida no projeto.
"""

_ML_NOTE = """---
title: Nota ML
tags: [ML]
date: 2026-09-02
---

# ML

Nota de outra pasta que coloca a tag ml no vocabulário aceito.
"""

_PROPOSALS_PATH = "projects/alpha/note.md"


@dataclass
class AutotagEnv:
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
) -> Iterator[AutotagEnv]:
    """Container settings (dim 8, fast settle, grace 0, folder rules) and aliases."""
    settings = Settings(postgres={"dsn": migrated_dsn})
    config = os_container.get_config()
    suffix = uuid4().hex[:8]
    settings.opensearch.url = f"http://{config['host']}:{config['port']}"
    settings.opensearch.chunks_alias = f"t{suffix}_chunks"
    settings.opensearch.entities_alias = f"t{suffix}_entities"
    settings.models.embedding.dim = 8
    settings.scan.settle_seconds = 0
    settings.scan.delete_grace_seconds = 0
    settings.tags.folder_rules = list(_FOLDER_RULES)
    fake = FakeEmbedder(dim=8)
    settings.models.embedding.model = fake.model
    monkeypatch.setattr("mynotes.config.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.db.session.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.scan.scanner.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.jobs.worker.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.index.sync.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.index.opensearch.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.llm.limits.get_settings", lambda: settings)
    reset_engine()
    with session_scope() as session:
        workspace = Workspace(name=f"autotag-{suffix}")
        session.add(workspace)
        session.flush()
        root = SourceRoot(workspace_id=workspace.id, name="root", abs_path=str(tmp_path))
        session.add(root)
        session.flush()
        settings.workspace = workspace.name
        state = AutotagEnv(
            settings=settings,
            fake=fake,
            root_id=root.id,
            workspace_id=workspace.id,
            root_path=tmp_path,
            chunks_alias=settings.opensearch.chunks_alias,
        )
    ensure_indices(get_client(), settings)
    ensure_pipeline(get_client(), settings)
    yield state
    client = get_client()
    for alias in (settings.opensearch.chunks_alias, settings.opensearch.entities_alias):
        try:
            physical = list(client.indices.get(index=f"{alias}_v*"))
        except Exception:
            physical = []
        for name in physical:
            client.indices.delete(index=name)
    with session_scope() as session:
        session.execute(delete(TagAlias).where(TagAlias.workspace_id == state.workspace_id))
        session.execute(delete(Tag).where(Tag.workspace_id == state.workspace_id))
        session.execute(delete(Job).where(Job.type.in_(_JOB_TYPES)))
        session.execute(delete(IndexOutbox).where(IndexOutbox.doc_kind == "chunk"))
        session.execute(delete(EmbeddingCache).where(EmbeddingCache.model == state.fake.model))
        session.execute(delete(File).where(File.source_root_id == state.root_id))
        session.execute(delete(Document).where(Document.source_root_id == state.root_id))
        session.execute(delete(ScanRun).where(ScanRun.source_root_id == state.root_id))
        session.execute(delete(SourceRoot).where(SourceRoot.id == state.root_id))
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


def document_and_version(env: AutotagEnv, rel_path: str) -> tuple[int, int]:
    """The document and latest version ids currently stored for ``rel_path``."""
    with session_scope() as session:
        file_row = session.execute(
            select(File).where(File.source_root_id == env.root_id, File.rel_path == rel_path)
        ).scalar_one()
        assert file_row.document_id is not None
        document = session.get(Document, file_row.document_id)
        assert document is not None and document.latest_version_id is not None
        return document.id, document.latest_version_id


def run_embed(env: AutotagEnv, version_id: int) -> None:
    """Run the embed handler with the fake embedder, like the worker would."""
    pipeline.handle_embed(
        {"version_id": version_id},
        JobContext(worker_id="test", settings=env.settings, embedder=env.fake),
    )


def ingest(env: AutotagEnv, rel_path: str, content: str, mtime_ns: int = _MTIME) -> tuple[int, int]:
    """Write a file, scan, drain extract/chunk and embed its latest version."""
    write_file(env.root_path / rel_path, content, mtime_ns)
    scan_root(env.root_id)
    drain_extract_chunk()
    document_id, version_id = document_and_version(env, rel_path)
    run_embed(env, version_id)
    return document_id, version_id


def run_tag_llm(env: AutotagEnv, version_id: int) -> None:
    """Run the tag_llm handler directly, like the worker would."""
    pipeline.handle_tag_llm(
        {"version_id": version_id},
        JobContext(worker_id="test", settings=env.settings, embedder=env.fake),
    )


def run_scan_handler(env: AutotagEnv, job_type: str, document_id: int) -> None:
    """Run the handler of one enqueued job type for a document."""
    handlers = {
        "reindex_metadata": pipeline.handle_reindex_metadata,
        "delete_document": pipeline.handle_delete_document,
    }
    with session_scope() as session:
        jobs = [
            job
            for job in session.execute(select(Job).where(Job.type == job_type)).scalars()
            if job.payload == {"document_id": document_id}
        ]
        payload = dict(jobs[0].payload) if jobs else None
    assert payload is not None, f"no {job_type} job for document {document_id}"
    handlers[job_type](
        payload, JobContext(worker_id="test", settings=env.settings, embedder=env.fake)
    )


def chat_with(proposals: list[tuple[str, float, str]]) -> FakeChat:
    """A FakeChat answering one fixed proposal set as JSON."""
    payload = {
        "tags": [
            {"tag": tag, "confidence": confidence, "reason": reason}
            for tag, confidence, reason in proposals
        ]
    }
    return FakeChat([json.dumps(payload)])


def patch_chat(monkeypatch: pytest.MonkeyPatch, chat: FakeChat) -> None:
    """Point the pipeline's ``get_chat`` at the fake."""
    monkeypatch.setattr("mynotes.jobs.pipeline.get_chat", lambda purpose="extract": chat)


def llm_rows(session: Session, document_id: int) -> dict[str, tuple[str, float, int]]:
    """The document's ``source='llm'`` rows as ``name -> (status, confidence, version)``."""
    rows = session.execute(
        select(
            Tag.name,
            DocumentTag.status,
            DocumentTag.confidence,
            DocumentTag.document_version_id,
        )
        .join(DocumentTag, DocumentTag.tag_id == Tag.id)
        .where(DocumentTag.document_id == document_id, DocumentTag.source == "llm")
    ).all()
    return {
        name: (status, round(float(confidence), 2), int(version_id))
        for name, status, confidence, version_id in rows
    }


def test_proposals_are_stored_and_auto_accepted(env: AutotagEnv, monkeypatch) -> None:
    """Vocabulary matches at high confidence are accepted; new/unsure ones proposed."""
    document_id, version_id = ingest(env, _PROPOSALS_PATH, _NOTE)
    ingest(env, "wiki/ml.md", _ML_NOTE)
    chat = chat_with(
        [
            ("ml", 0.95, "sobre machine learning"),
            ("gardening", 0.9, "jardim"),
            ("rust", 0.4, "talvez rust"),
        ]
    )
    patch_chat(monkeypatch, chat)

    run_tag_llm(env, version_id)

    assert len(chat.calls) == 1
    with session_scope() as session:
        assert llm_rows(session, document_id) == {
            "ml": ("accepted", 0.95, version_id),
            "gardening": ("proposed", 0.9, version_id),
            "rust": ("proposed", 0.4, version_id),
        }
        reasons = dict(
            session.execute(
                select(Tag.name, DocumentTag.reason)
                .join(DocumentTag, DocumentTag.tag_id == Tag.id)
                .where(DocumentTag.document_id == document_id, DocumentTag.source == "llm")
            ).all()
        )
        ml_ids = list(
            session.execute(
                select(Tag.id).where(Tag.workspace_id == env.workspace_id, Tag.name == "ml")
            ).scalars()
        )
        assert len(ml_ids) == 1, "the proposal must reuse the existing ml tag"
    assert reasons == {
        "ml": "sobre machine learning",
        "gardening": "jardim",
        "rust": "talvez rust",
    }
    with session_scope() as session:
        reindex_jobs = [
            job
            for job in session.execute(select(Job).where(Job.type == "reindex_metadata")).scalars()
            if job.payload == {"document_id": document_id} and job.status == "queued"
        ]
    assert reindex_jobs, "an auto-accepted tag must enqueue a metadata reindex"


def test_rerun_for_same_version_is_a_no_op(env: AutotagEnv, monkeypatch) -> None:
    """A second run for a version that already has LLM rows skips the model."""
    document_id, version_id = ingest(env, _PROPOSALS_PATH, _NOTE)
    ingest(env, "wiki/ml.md", _ML_NOTE)
    chat = chat_with([("ml", 0.95, "ml"), ("gardening", 0.9, "jardim"), ("rust", 0.4, "rust")])
    patch_chat(monkeypatch, chat)

    run_tag_llm(env, version_id)
    with session_scope() as session:
        before = llm_rows(session, document_id)
    run_tag_llm(env, version_id)

    assert len(chat.calls) == 1, "the idempotent skip must not call the model again"
    with session_scope() as session:
        assert llm_rows(session, document_id) == before


def test_new_version_drops_stale_proposals_and_keeps_decisions(
    env: AutotagEnv, monkeypatch
) -> None:
    """A new version re-proposes: decided rows persist, stale proposals vanish."""
    document_id, version_id = ingest(env, _PROPOSALS_PATH, _NOTE)
    ingest(env, "wiki/ml.md", _ML_NOTE)
    chat = FakeChat(
        [
            json.dumps(
                {
                    "tags": [
                        {"tag": "ml", "confidence": 0.95, "reason": "ml"},
                        {"tag": "gardening", "confidence": 0.9, "reason": "garden"},
                        {"tag": "rust", "confidence": 0.4, "reason": "rust"},
                        {"tag": "fishing", "confidence": 0.4, "reason": "fish"},
                    ]
                }
            ),
            json.dumps(
                {
                    "tags": [
                        {"tag": "gardening", "confidence": 0.95, "reason": "still garden"},
                        {"tag": "cooking", "confidence": 0.5, "reason": "recipes"},
                    ]
                }
            ),
        ]
    )
    patch_chat(monkeypatch, chat)
    run_tag_llm(env, version_id)
    with session_scope() as session:
        rust_id = session.execute(
            select(Tag.id).where(Tag.workspace_id == env.workspace_id, Tag.name == "rust")
        ).scalar_one()
        assert decide_proposal(session, document_id, rust_id, "rejected")

    write_file(env.root_path / _PROPOSALS_PATH, _NOTE_V2, _MTIME + 1)
    scan_root(env.root_id)
    drain_extract_chunk()
    _, second_version = document_and_version(env, _PROPOSALS_PATH)
    assert second_version != version_id

    run_tag_llm(env, second_version)

    with session_scope() as session:
        rows = llm_rows(session, document_id)
    assert rows == {
        "ml": ("accepted", 0.95, version_id),
        "gardening": ("accepted", 0.95, second_version),
        "cooking": ("proposed", 0.5, second_version),
        "rust": ("rejected", 0.4, version_id),
    }


def test_outside_background_window_requeues_without_attempt(env: AutotagEnv, monkeypatch) -> None:
    """A window excluding now makes the worker re-queue with a future run_after."""
    _document_id, version_id = ingest(env, _PROPOSALS_PATH, _NOTE)
    ingest(env, "wiki/ml.md", _ML_NOTE)
    chat = chat_with([("ml", 0.95, "ml")])
    patch_chat(monkeypatch, chat)
    start = datetime.now().astimezone() + timedelta(seconds=90)
    env.settings.models.llm.limits.background_window = (
        f"{start:%H:%M}-{start + timedelta(minutes=30):%H:%M}"
    )

    with session_scope() as session:
        job = session.execute(
            select(Job).where(Job.type == "tag_llm", Job.payload == {"version_id": version_id})
        ).scalar_one()
        job_id = job.id
        assert job.status == "queued" and job.attempts == 0

    before = datetime.now(UTC)
    assert Worker(concurrency=1, poll_interval_s=0.05, types=["tag_llm"]).run_once() is True

    with session_scope() as session:
        job = session.get(Job, job_id)
        assert job is not None
        assert job.status == "queued"
        assert job.attempts == 0, "RetryLater must not consume an attempt"
        assert job.last_error == "outside background_window"
        assert job.run_after > before
    assert chat.calls == [], "no model call outside the window"


def test_accepted_llm_tag_reaches_index_and_proposed_does_not(env: AutotagEnv, monkeypatch) -> None:
    """Synced chunk documents carry accepted LLM tags only."""
    document_id, version_id = ingest(env, _PROPOSALS_PATH, _NOTE)
    ml_document_id, _ml_version = ingest(env, "wiki/ml.md", _ML_NOTE)
    sync_once()
    chat = chat_with([("ml", 0.95, "ml"), ("gardening", 0.9, "garden"), ("rust", 0.4, "rust")])
    patch_chat(monkeypatch, chat)

    run_tag_llm(env, version_id)
    run_scan_handler(env, "reindex_metadata", document_id)
    sync_once()

    client = get_client()
    client.indices.refresh(index=env.chunks_alias)
    result = client.search(index=env.chunks_alias, body={"query": {"match_all": {}}, "size": 100})
    tags_by_doc: dict[str, set[str]] = {}
    for hit in result["hits"]["hits"]:
        tags_by_doc.setdefault(hit["_source"]["document_id"], set()).update(hit["_source"]["tags"])
    assert "ml" in tags_by_doc[str(document_id)]
    assert "gardening" not in tags_by_doc[str(document_id)]
    assert "rust" not in tags_by_doc[str(document_id)]
    assert "gardening" not in tags_by_doc.get(str(ml_document_id), set())


def test_review_cli_accepts_via_loop_and_rejects_by_command(env: AutotagEnv, monkeypatch) -> None:
    """``tags review`` accepts/skips interactively; ``tags reject`` is non-interactive."""
    document_id, version_id = ingest(env, _PROPOSALS_PATH, _NOTE)
    ingest(env, "wiki/ml.md", _ML_NOTE)
    chat = chat_with(
        [("ml", 0.95, "ml"), ("gardening", 0.9, "jardim"), ("rust", 0.4, "talvez rust")]
    )
    patch_chat(monkeypatch, chat)
    run_tag_llm(env, version_id)

    result = CliRunner().invoke(app, ["tags", "review"], input="a\nq\n")
    assert result.exit_code == 0, result.output
    with session_scope() as session:
        rows = llm_rows(session, document_id)
    assert rows["gardening"][0] == "accepted"
    assert rows["rust"][0] == "proposed"
    assert rows["ml"][0] == "accepted"

    result = CliRunner().invoke(app, ["tags", "reject", _PROPOSALS_PATH, "rust"])
    assert result.exit_code == 0, result.output
    with session_scope() as session:
        assert llm_rows(session, document_id)["rust"][0] == "rejected"

    result = CliRunner().invoke(app, ["tags", "review"])
    assert result.exit_code == 0, result.output
    assert "No proposed LLM tags to review." in result.output


def test_llm_tags_do_not_rechunk(env: AutotagEnv, monkeypatch) -> None:
    """LLM tags never join embed_text: no chunk job and no embed_text change."""
    document_id, version_id = ingest(env, _PROPOSALS_PATH, _NOTE)
    ingest(env, "wiki/ml.md", _ML_NOTE)
    with session_scope() as session:
        embed_texts_before = list(
            session.execute(
                select(Chunk.embed_text)
                .where(Chunk.document_id == document_id)
                .order_by(Chunk.ordinal)
            ).scalars()
        )
    assert embed_texts_before
    chat = chat_with([("ml", 0.95, "ml")])
    patch_chat(monkeypatch, chat)

    run_tag_llm(env, version_id)

    with session_scope() as session:
        embed_texts_after = list(
            session.execute(
                select(Chunk.embed_text)
                .where(Chunk.document_id == document_id)
                .order_by(Chunk.ordinal)
            ).scalars()
        )
        chunk_jobs = [
            job
            for job in session.execute(select(Job).where(Job.type == "chunk")).scalars()
            if job.payload == {"version_id": version_id} and job.status == "queued"
        ]
    assert embed_texts_after == embed_texts_before
    assert chunk_jobs == []
