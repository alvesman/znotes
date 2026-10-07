"""Integration tests for the extract/chunk pipeline handlers (task 20).

Each test registers a temporary directory as a ``source_root``, scans it with
the real scanner (task 16) and drains the queue with a ``Worker`` limited to
the ``extract`` and ``chunk`` types, then inspects the committed rows.
Handlers commit through ``session_scope``, so setup and assertions also go
through ``session_scope`` and every test cleans up its rows afterwards.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pymupdf
import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from mynotes.config import Settings
from mynotes.db.models import (
    Chunk,
    Document,
    DocumentVersion,
    File,
    Job,
    ScanRun,
    SourceRoot,
    Tag,
    TagAlias,
    Workspace,
)
from mynotes.db.session import session_scope
from mynotes.jobs import pipeline
from mynotes.jobs.registry import JobContext, Priority
from mynotes.jobs.worker import Worker
from mynotes.scan.scanner import scan_root

pytestmark = pytest.mark.integration

_MTIME = 1_000_000_000_000_000_000
_JOB_TYPES = (
    "extract",
    "chunk",
    "embed",
    "test_follow",
    "test_disabled",
    "version_detect",
    "tag_llm",
    "link_resolve",
    "kg_extract",
)
_WORKER_TYPES = ("extract", "chunk")

_FIXTURE_DOCS = Path(__file__).resolve().parents[1] / "fixtures" / "docs"
_NOTE_PT = (_FIXTURE_DOCS / "note_pt.md").read_text(encoding="utf-8")


@dataclass
class PipelineEnv:
    """The registered test root plus the settings the pipeline will read."""

    settings: Settings
    root_id: int
    workspace_id: int
    root_path: Path


@pytest.fixture
def pipeline_env(
    settings_with_db: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[PipelineEnv]:
    """Container settings with a fast settle time, a temp root and a patched worker."""
    settings_with_db.scan.settle_seconds = 0
    monkeypatch.setattr("mynotes.scan.scanner.get_settings", lambda: settings_with_db)
    monkeypatch.setattr("mynotes.jobs.worker.get_settings", lambda: settings_with_db)
    with session_scope() as session:
        workspace = Workspace(name=f"pipe-{uuid4().hex[:8]}")
        session.add(workspace)
        session.flush()
        root = SourceRoot(workspace_id=workspace.id, name="root", abs_path=str(tmp_path))
        session.add(root)
        session.flush()
        env = PipelineEnv(
            settings=settings_with_db,
            root_id=root.id,
            workspace_id=workspace.id,
            root_path=tmp_path,
        )
    yield env
    with session_scope() as session:
        session.execute(delete(Job).where(Job.type.in_(_JOB_TYPES)))
        session.execute(delete(File).where(File.source_root_id == env.root_id))
        session.execute(delete(Document).where(Document.source_root_id == env.root_id))
        session.execute(delete(ScanRun).where(ScanRun.source_root_id == env.root_id))
        session.execute(delete(SourceRoot).where(SourceRoot.id == env.root_id))
        session.execute(delete(TagAlias).where(TagAlias.workspace_id == env.workspace_id))
        session.execute(delete(Tag).where(Tag.workspace_id == env.workspace_id))
        session.execute(delete(Workspace).where(Workspace.id == env.workspace_id))


@pytest.fixture(autouse=True)
def _restore_follow_ups() -> Iterator[None]:
    """Keep the follow-up registry clean even when a test registration fails."""
    before = list(pipeline._FOLLOW_UPS)
    yield
    pipeline._FOLLOW_UPS[:] = before


def write_file(path: Path, content: str, mtime_ns: int = _MTIME) -> None:
    """Create a file with a fixed mtime so scans are deterministic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))


def write_pdf(path: Path, pages: list[list[tuple[str, float]]], mtime_ns: int = _MTIME) -> None:
    """Write a small PDF with one ``insert_text`` line per tuple (text, size)."""
    document = pymupdf.open()
    try:
        for lines in pages:
            page = document.new_page()
            y = 100.0
            for text, size in lines:
                page.insert_text((72, y), text, fontsize=size, fontname="helv")
                y += size + 38.0
        document.save(path)
    finally:
        document.close()
    os.utime(path, ns=(mtime_ns, mtime_ns))


def drain_queue(settings: Settings) -> None:
    """Run the worker until no ``extract``/``chunk`` job is claimable."""
    worker = Worker(concurrency=1, poll_interval_s=0.05, types=list(_WORKER_TYPES))
    while worker.run_once():
        pass


def rows(session: Session, model: Any, **filters: Any) -> list[Any]:
    """Fetch all rows of ``model`` matching the filters, ordered by id."""
    stmt = select(model).filter_by(**filters).order_by(model.id)
    return list(session.execute(stmt).scalars().all())


def documents_of(env: PipelineEnv, session: Session) -> list[Document]:
    """The documents created under the test root."""
    return rows(session, Document, source_root_id=env.root_id)


def single_version(env: PipelineEnv, session: Session) -> DocumentVersion:
    """The only document version of a one-file test root."""
    document = documents_of(env, session)[0]
    return session.execute(
        select(DocumentVersion).where(DocumentVersion.document_id == document.id)
    ).scalar_one()


def run_handler(handler: Any, payload: dict[str, Any], settings: Settings) -> None:
    """Run a pipeline handler directly, like the worker would."""
    handler(payload, JobContext(worker_id="test", settings=settings))


def test_scan_then_extract_and_chunk(pipeline_env: PipelineEnv) -> None:
    """Scanning a markdown and a PDF file, then draining, chunks both versions."""
    write_file(pipeline_env.root_path / "note_pt.md", _NOTE_PT)
    write_pdf(
        pipeline_env.root_path / "manual.pdf",
        [
            [
                ("Manual AquaBreeze", 24.0),
                ("Guarde este manual para consulta futura.", 11.0),
                ("O aparelho destina-se exclusivamente a uso doméstico.", 11.0),
            ],
            [
                ("Características do produto.", 11.0),
                ("Limpe o aparelho antes da primeira utilização.", 11.0),
            ],
        ],
    )

    summary = scan_root(pipeline_env.root_id)
    assert summary.errors == []
    drain_queue(pipeline_env.settings)

    with session_scope() as session:
        documents = documents_of(pipeline_env, session)
        assert len(documents) == 2
        versions = rows(session, DocumentVersion)
        assert len(versions) == 2
        assert all(item.status == "chunked" for item in versions)
        assert all(item.ir_json is not None for item in versions)
        assert all(item.extractor for item in versions)

        md_doc = next(item for item in documents if item.format == "md")
        assert md_doc.title == "Ideia Alfa"
        assert md_doc.title_norm == "ideia alfa"
        assert md_doc.doc_date is not None

        pdf_doc = next(item for item in documents if item.format == "pdf")
        assert pdf_doc.title is not None

        for document in (md_doc, pdf_doc):
            chunks = rows(session, Chunk, document_id=document.id)
            assert chunks, f"document {document.format} must have chunks"
            assert all(chunk.is_latest for chunk in chunks)

        active = rows(session, Job, type="extract") + rows(session, Job, type="chunk")
        assert all(job.status == "done" for job in active)

        embed_jobs = rows(session, Job, type="embed")
        assert len(embed_jobs) == 2
        assert {job.payload["version_id"] for job in embed_jobs} == {v.id for v in versions}
        assert all(job.status == "queued" for job in embed_jobs)
        assert all(job.priority == Priority.EMBED for job in embed_jobs)
        assert {job.dedupe_key for job in embed_jobs} == {f"embed:{v.id}" for v in versions}


def test_chunk_rerun_leaves_the_same_chunk_rows(pipeline_env: PipelineEnv) -> None:
    """Running the chunk handler twice for one version does not duplicate rows."""
    write_file(pipeline_env.root_path / "note_pt.md", _NOTE_PT)
    scan_root(pipeline_env.root_id)
    drain_queue(pipeline_env.settings)

    with session_scope() as session:
        version = single_version(pipeline_env, session)
        version_id = version.id
        before = session.execute(
            select(func.count()).select_from(Chunk).where(Chunk.document_version_id == version.id)
        ).scalar_one()
        assert before > 0

    run_handler(
        pipeline.handle_chunk,
        {"version_id": version_id},
        pipeline_env.settings,
    )

    with session_scope() as session:
        after = session.execute(
            select(func.count()).select_from(Chunk).where(Chunk.document_version_id == version_id)
        ).scalar_one()
    assert after == before


def test_new_version_flips_is_latest_on_old_chunks(pipeline_env: PipelineEnv) -> None:
    """After version 2 is chunked, its chunks are latest and version 1's are not."""
    write_file(
        pipeline_env.root_path / "note.md",
        "# Ideia Alfa\n\nprimeira versão do texto com bastante conteúdo em português.\n",
    )
    scan_root(pipeline_env.root_id)
    drain_queue(pipeline_env.settings)
    write_file(
        pipeline_env.root_path / "note.md",
        "# Ideia Alfa\n\nsegunda versão com um texto completamente diferente do anterior.\n",
        _MTIME + 5_000_000_000,
    )
    scan_root(pipeline_env.root_id)
    drain_queue(pipeline_env.settings)

    with session_scope() as session:
        document = documents_of(pipeline_env, session)[0]
        versions = rows(session, DocumentVersion, document_id=document.id)
        assert [item.version_no for item in versions] == [1, 2]
        v1_chunks = rows(session, Chunk, document_version_id=versions[0].id)
        v2_chunks = rows(session, Chunk, document_version_id=versions[1].id)
        assert v1_chunks and v2_chunks
        assert all(not chunk.is_latest for chunk in v1_chunks)
        assert all(chunk.is_latest for chunk in v2_chunks)


def test_extract_skips_when_file_changed_after_scan(pipeline_env: PipelineEnv) -> None:
    """A hash mismatch between scan and extract finishes quietly, version stays pending."""
    write_file(
        pipeline_env.root_path / "note.md",
        "# Primeira\n\nconteúdo original que o scanner verificou.\n",
    )
    scan_root(pipeline_env.root_id)
    with session_scope() as session:
        version = single_version(pipeline_env, session)
        version_id = version.id
        assert version.status == "pending"

    write_file(
        pipeline_env.root_path / "note.md",
        "# Segunda\n\nconteúdo modificado depois do scan do arquivo.\n",
    )
    drain_queue(pipeline_env.settings)

    with session_scope() as session:
        version = session.get(DocumentVersion, version_id)
        assert version is not None
        assert version.status == "pending"
        assert version.ir_json is None
        assert rows(session, Job, type="chunk") == []
        assert all(job.status == "done" for job in rows(session, Job, type="extract"))


def test_invalid_pdf_fails_the_version_without_retry(pipeline_env: PipelineEnv) -> None:
    """A .pdf holding plain text fails its version; the job completes, not retried."""
    write_file(pipeline_env.root_path / "broken.pdf", "isto não é um pdf, apenas texto simples\n")
    scan_root(pipeline_env.root_id)
    drain_queue(pipeline_env.settings)

    with session_scope() as session:
        version = single_version(pipeline_env, session)
        assert version.status == "failed"
        assert version.error
        assert version.ir_json is None
        extract_jobs = rows(session, Job, type="extract")
        assert [job.status for job in extract_jobs] == ["done"]
        assert rows(session, Job, type="chunk") == []


def test_registered_follow_up_is_enqueued_after_chunk(pipeline_env: PipelineEnv) -> None:
    """A registered follow-up is enqueued per version; a disabled one is not."""
    pipeline.register_follow_up(pipeline.FollowUp("test_follow", 99))
    pipeline.register_follow_up(
        pipeline.FollowUp("test_disabled", 99, enabled=lambda settings: False)
    )
    write_file(
        pipeline_env.root_path / "note.md",
        "# Ideia\n\nconteúdo para acionar o caminho completo do pipeline.\n",
    )
    scan_root(pipeline_env.root_id)
    drain_queue(pipeline_env.settings)

    with session_scope() as session:
        version = single_version(pipeline_env, session)
        follow_jobs = rows(session, Job, type="test_follow")
        assert len(follow_jobs) == 1
        job = follow_jobs[0]
        assert job.status == "queued"
        assert job.priority == 99
        assert job.dedupe_key == f"test_follow:{version.id}"
        assert job.payload == {"version_id": version.id}
        assert rows(session, Job, type="test_disabled") == []


def test_bulk_chunk_insert_writes_complete_rows(pipeline_env: PipelineEnv) -> None:
    """The executemany chunk insert (task 51) produces complete chunk rows.

    Chunks are inserted with a Core ``insert()`` executemany instead of one
    ORM object per chunk; this pins that the server-side defaults (uuidv7 id,
    ``created_at``) still apply and every draft field is persisted.
    """
    write_file(pipeline_env.root_path / "note_pt.md", _NOTE_PT)
    scan_root(pipeline_env.root_id)
    drain_queue(pipeline_env.settings)

    with session_scope() as session:
        version = single_version(pipeline_env, session)
        chunks = rows(session, Chunk, document_version_id=version.id)
        assert len(chunks) > 0
        ordinals = [chunk.ordinal for chunk in chunks]
        for chunk in chunks:
            assert chunk.created_at is not None
            assert chunk.content_hash
            assert chunk.text
            assert chunk.embed_text
            assert chunk.token_count > 0
            assert chunk.is_latest is True
            assert chunk.id is not None
    assert ordinals == list(range(len(chunks)))
