"""Integration tests for the maintenance handlers and backup (task 50).

Rows are inserted directly through ``session_scope`` (documents, versions,
chunks, cache rows, jobs, outbox), the maintenance functions run against the
real Postgres container, and every test cleans up its committed rows. The
backup test shells out to the local ``pg_dump`` against the testcontainer and
skips when it is not installed.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from fakes import FakeEmbedder
from mynotes.cli import app
from mynotes.config import Settings
from mynotes.db.models import (
    Chunk,
    Document,
    DocumentVersion,
    EmbeddingCache,
    File,
    IndexOutbox,
    Job,
    KgEntity,
    KgExtractionCache,
    KgMention,
    SourceRoot,
    Workspace,
    pack_vector,
)
from mynotes.db.session import reset_engine, session_scope
from mynotes.embed.cache import embed_key
from mynotes.graph.ontology import EXTRACTOR_VERSION
from mynotes.jobs.maintenance import clean_caches, prune_ops, purge_tombstones
from mynotes.scan.hashing import hash_text
from mynotes.text.normalize import normalize_key

pytestmark = pytest.mark.integration

runner = CliRunner()

_AGE_OLD = timedelta(days=60)
_AGE_FRESH = timedelta(hours=1)
_ORPHAN_KEY = hash_text("texto órfão que nunca foi um chunk")
_FRESH_ORPHAN_KEY = hash_text("órfão recente")
_KG_OLD_HASH = "kg-old-other-version"
_KG_CURRENT_HASH = "kg-old-current-version"
_JOB_TYPES = ("extract", "purge_tombstones", "clean_caches", "prune_ops")


@dataclass
class MaintenanceEnv:
    """The test workspace/root ids, the settings and the fake embedder."""

    settings: Settings
    fake: FakeEmbedder
    root_id: int
    workspace_id: int


@pytest.fixture
def maintenance_env(
    migrated_dsn: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[MaintenanceEnv]:
    """Settings pointing at the container plus a disposable workspace/root."""
    fake = FakeEmbedder(dim=8)
    settings = Settings(postgres={"dsn": migrated_dsn})
    settings.models.embedding.model = fake.model
    settings.models.embedding.dim = 8
    monkeypatch.setattr("mynotes.config.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.db.session.get_settings", lambda: settings)
    monkeypatch.setattr("mynotes.embed.client.get_settings", lambda: settings)
    # The session's first alembic run calls fileConfig (disable_existing_loggers
    # default), which disables every mynotes logger created at import time;
    # re-enable this module's logger so the prune test can assert its records.
    logging.getLogger("mynotes.jobs.maintenance").disabled = False
    reset_engine()
    with session_scope() as session:
        workspace = Workspace(name=f"maint-{uuid4().hex[:8]}")
        session.add(workspace)
        session.flush()
        root = SourceRoot(workspace_id=workspace.id, name="root", abs_path=str(tmp_path))
        session.add(root)
        session.flush()
        env = MaintenanceEnv(
            settings=settings,
            fake=fake,
            root_id=int(root.id),
            workspace_id=int(workspace.id),
        )
    yield env
    with session_scope() as session:
        entity_ids = select(KgEntity.id).where(KgEntity.workspace_id == env.workspace_id)
        session.execute(delete(KgMention).where(KgMention.entity_id.in_(entity_ids)))
        session.execute(delete(KgEntity).where(KgEntity.workspace_id == env.workspace_id))
        session.execute(
            delete(KgExtractionCache).where(
                KgExtractionCache.content_hash.in_([_KG_OLD_HASH, _KG_CURRENT_HASH])
            )
        )
        session.execute(delete(EmbeddingCache).where(EmbeddingCache.model == env.fake.model))
        session.execute(delete(Job).where(Job.type.in_(_JOB_TYPES)))
        session.execute(delete(IndexOutbox))
        session.execute(delete(File).where(File.source_root_id == env.root_id))
        session.execute(delete(Document).where(Document.source_root_id == env.root_id))
        session.execute(delete(SourceRoot).where(SourceRoot.id == env.root_id))
        session.execute(delete(Workspace).where(Workspace.id == env.workspace_id))
    reset_engine()


def add_document(
    session: Session,
    env: MaintenanceEnv,
    *,
    status: str = "current",
    age: timedelta | None = None,
    file_status: str = "active",
) -> tuple[int, list]:
    """Insert a document with one version, one chunk and a file row.

    Args:
        session: Open session/transaction.
        env: The test environment (workspace/root ids).
        status: The document status.
        age: When set, backdates ``created_at``/``updated_at`` by this much.
        file_status: The file row status.

    Returns:
        ``(document_id, [chunk_ids])``.
    """
    when = datetime.now(UTC) - age if age is not None else datetime.now(UTC)
    document = Document(
        workspace_id=env.workspace_id,
        source_root_id=env.root_id,
        format="md",
        status=status,
    )
    document.created_at = when
    document.updated_at = when
    session.add(document)
    session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version_no=1,
        content_hash=uuid4().hex,
        rel_path_at_version=f"doc-{document.id}.md",
        mtime_ns=1_000,
        status="chunked",
        is_latest=True,
    )
    version.created_at = when
    session.add(version)
    session.flush()
    document.latest_version_id = version.id
    chunk = Chunk(
        document_version_id=version.id,
        document_id=document.id,
        ordinal=0,
        content_hash=uuid4().hex,
        text="conteúdo do documento de teste",
        embed_text=f"embed {document.id}",
        token_count=8,
    )
    chunk.created_at = when
    session.add(chunk)
    session.flush()
    file_row = File(
        source_root_id=env.root_id,
        rel_path=f"doc-{document.id}.md",
        rel_path_norm=f"doc-{document.id}.md",
        size=10,
        mtime_ns=1_000,
        format="md",
        status=file_status,
        document_id=document.id,
    )
    session.add(file_row)
    session.flush()
    document.current_file_id = file_row.id
    return int(document.id), [chunk.id]


def add_entity_with_mention(
    session: Session, env: MaintenanceEnv, name: str, chunk_id, version_id: int
) -> int:
    """Insert an entity mentioned only in the given chunk; return its id."""
    entity = KgEntity(
        workspace_id=env.workspace_id,
        type="person",
        canonical_name=name,
        norm_name=normalize_key(name),
    )
    session.add(entity)
    session.flush()
    session.add(
        KgMention(
            entity_id=entity.id,
            chunk_id=chunk_id,
            document_version_id=version_id,
            surface_form=name,
        )
    )
    session.flush()
    return int(entity.id)


def _latest_version_id(session: Session, document_id: int) -> int:
    """The id of the document's single version row."""
    return int(
        session.execute(
            select(DocumentVersion.id).where(DocumentVersion.document_id == document_id)
        ).scalar_one()
    )


def test_purge_tombstones_removes_only_old_deleted_documents(
    maintenance_env: MaintenanceEnv,
) -> None:
    """A 60-day-old tombstone (and its rows) is purged; a 1-hour-old one stays."""
    env = maintenance_env
    with session_scope() as session:
        old_id, old_chunks = add_document(
            session, env, status="deleted", age=_AGE_OLD, file_status="deleted"
        )
        fresh_id, fresh_chunks = add_document(
            session, env, status="deleted", age=_AGE_FRESH, file_status="deleted"
        )
        stale_entity = add_entity_with_mention(
            session, env, "Entidade Velha", old_chunks[0], _latest_version_id(session, old_id)
        )
        live_entity = add_entity_with_mention(
            session, env, "Entidade Nova", fresh_chunks[0], _latest_version_id(session, fresh_id)
        )

    with session_scope() as session:
        counts = purge_tombstones(session, env.settings)
    assert counts == {"documents": 1, "files": 1, "entities_stale": 1}

    with session_scope() as session:
        assert session.get(Document, old_id) is None
        assert (
            session.scalar(
                select(func.count()).select_from(Chunk).where(Chunk.document_id == old_id)
            )
            == 0
        )
        assert (
            session.scalar(select(func.count()).select_from(File).where(File.document_id == old_id))
            == 0
        )
        assert session.get(Document, fresh_id) is not None
        assert (
            session.scalar(
                select(func.count()).select_from(File).where(File.document_id == fresh_id)
            )
            == 1
        )
        chunk_deletes = (
            session.execute(
                select(IndexOutbox).where(
                    IndexOutbox.doc_kind == "chunk", IndexOutbox.op == "delete"
                )
            )
            .scalars()
            .all()
        )
        assert {row.doc_id for row in chunk_deletes} == {str(c) for c in old_chunks}
        assert session.get(KgEntity, stale_entity).status == "stale"
        assert session.get(KgEntity, live_entity).status == "active"
        entity_deletes = (
            session.execute(
                select(IndexOutbox).where(
                    IndexOutbox.doc_kind == "entity", IndexOutbox.op == "delete"
                )
            )
            .scalars()
            .all()
        )
        assert {row.doc_id for row in entity_deletes} == {str(stale_entity)}

    with session_scope() as session:
        assert purge_tombstones(session, env.settings) == {
            "documents": 0,
            "files": 0,
            "entities_stale": 0,
        }


def test_clean_caches_removes_orphan_rows_only(maintenance_env: MaintenanceEnv) -> None:
    """Old orphaned cache rows go; live keys and current extractor versions stay."""
    env = maintenance_env
    with session_scope() as session:
        _, chunk_ids = add_document(session, env)
        embed_text = session.execute(
            select(Chunk.embed_text).where(Chunk.id == chunk_ids[0])
        ).scalar_one()
        live_key = embed_key(env.fake, embed_text)
        for key in (live_key, _ORPHAN_KEY):
            row = EmbeddingCache(
                model=env.fake.model, content_hash=key, dim=8, vector=pack_vector([0.5] * 8)
            )
            row.created_at = datetime.now(UTC) - _AGE_OLD
            session.add(row)
        session.add(
            EmbeddingCache(
                model=env.fake.model,
                content_hash=_FRESH_ORPHAN_KEY,
                dim=8,
                vector=pack_vector([0.5] * 8),
            )
        )
        for content_hash, version in (
            (_KG_OLD_HASH, "kg-v0"),
            (_KG_CURRENT_HASH, EXTRACTOR_VERSION),
        ):
            row = KgExtractionCache(
                content_hash=content_hash, extractor_version=version, model="m", result={}
            )
            row.created_at = datetime.now(UTC) - _AGE_OLD
            session.add(row)
        session.flush()

    with session_scope() as session:
        counts = clean_caches(session, env.settings, env.fake)
    assert counts == {"embedding_cache": 1, "kg_extraction_cache": 1}

    with session_scope() as session:
        remaining = set(
            session.execute(
                select(EmbeddingCache.content_hash).where(EmbeddingCache.model == env.fake.model)
            ).scalars()
        )
        assert remaining == {live_key, _FRESH_ORPHAN_KEY}
        kg_versions = set(
            session.execute(
                select(KgExtractionCache.extractor_version).where(
                    KgExtractionCache.content_hash.in_([_KG_OLD_HASH, _KG_CURRENT_HASH])
                )
            ).scalars()
        )
        assert kg_versions == {EXTRACTOR_VERSION}


def test_prune_ops_removes_old_finished_jobs_and_processed_outbox(
    maintenance_env: MaintenanceEnv, caplog: pytest.LogCaptureFixture
) -> None:
    """Old done/dead jobs and processed outbox rows go; queued/fresh rows stay."""
    env = maintenance_env
    with session_scope() as session:
        done_old = Job(type="extract", payload={}, status="done")
        done_old.finished_at = datetime.now(UTC) - _AGE_OLD
        dead_old = Job(type="extract", payload={}, status="dead", last_error="boom: endpoint 500")
        dead_old.finished_at = datetime.now(UTC) - _AGE_OLD
        queued = Job(type="extract", payload={}, status="queued")
        done_fresh = Job(type="extract", payload={}, status="done")
        done_fresh.finished_at = datetime.now(UTC) - _AGE_FRESH
        session.add_all([done_old, dead_old, queued, done_fresh])
        for doc_id in ("old", "fresh", "pending"):
            session.add(
                IndexOutbox(index_name="chunks", doc_kind="chunk", doc_id=doc_id, op="upsert")
            )
        session.flush()
        job_ids = {
            "done_old": done_old.id,
            "dead_old": dead_old.id,
            "queued": queued.id,
            "done_fresh": done_fresh.id,
        }
        session.execute(
            update(IndexOutbox)
            .where(IndexOutbox.doc_id == "old")
            .values(processed_at=datetime.now(UTC) - timedelta(days=10))
        )
        session.execute(
            update(IndexOutbox)
            .where(IndexOutbox.doc_id == "fresh")
            .values(processed_at=datetime.now(UTC) - _AGE_FRESH)
        )
        session.flush()

    with (
        caplog.at_level(logging.WARNING, logger="mynotes.jobs.maintenance"),
        session_scope() as session,
    ):
        counts = prune_ops(session, env.settings)
    assert counts == {"jobs": 2, "index_outbox": 1}
    assert "boom: endpoint 500" in caplog.text

    with session_scope() as session:
        # Other suites may leave unrelated job rows behind, so assert on the
        # rows this test inserted instead of global status counts.
        assert session.get(Job, job_ids["done_old"]) is None
        assert session.get(Job, job_ids["dead_old"]) is None
        assert session.get(Job, job_ids["queued"]) is not None
        assert session.get(Job, job_ids["done_fresh"]) is not None
        remaining = set(session.execute(select(IndexOutbox.doc_id)).scalars())
        assert remaining == {"fresh", "pending"}


def test_maintenance_run_dry_run_reports_without_deleting(
    maintenance_env: MaintenanceEnv,
) -> None:
    """``maintenance run --dry-run`` prints counts and leaves every row in place."""
    env = maintenance_env
    orphan_key = hash_text(f"linha de cache órfã {uuid4().hex}")
    with session_scope() as session:
        old_id, _ = add_document(
            session, env, status="deleted", age=_AGE_OLD, file_status="deleted"
        )
        orphan = EmbeddingCache(
            model=env.fake.model,
            content_hash=orphan_key,
            dim=8,
            vector=pack_vector([0.5] * 8),
        )
        orphan.created_at = datetime.now(UTC) - _AGE_OLD
        session.add(orphan)
        job = Job(type="extract", payload={}, status="done")
        job.finished_at = datetime.now(UTC) - _AGE_OLD
        session.add(job)
        session.add(IndexOutbox(index_name="chunks", doc_kind="chunk", doc_id="x", op="upsert"))
        session.flush()
        job_id = job.id
        session.execute(
            update(IndexOutbox)
            .where(IndexOutbox.doc_id == "x")
            .values(processed_at=datetime.now(UTC) - timedelta(days=10))
        )
        session.flush()

    result = runner.invoke(app, ["maintenance", "run", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "dry run" in result.output
    for operation in ("purge_tombstones", "clean_caches", "prune_ops"):
        assert operation in result.output
    # Other suites may leave unrelated rows behind, so only the counts of this
    # test's own data are guaranteed; the rows themselves must all survive.
    assert "documents=1" in result.output

    with session_scope() as session:
        assert session.get(Document, old_id) is not None
        assert (
            session.scalar(
                select(func.count())
                .select_from(EmbeddingCache)
                .where(EmbeddingCache.content_hash == orphan_key)
            )
            == 1
        )
        assert session.get(Job, job_id) is not None
        assert (
            session.scalar(
                select(func.count()).select_from(IndexOutbox).where(IndexOutbox.doc_id == "x")
            )
            == 1
        )


def test_backup_produces_nonempty_dump(migrated_dsn: str, tmp_path: Path) -> None:
    """``pg_dump -Fc`` against the testcontainer yields a valid custom archive."""
    if shutil.which("pg_dump") is None:
        pytest.skip("pg_dump is not installed; the compose fallback needs a compose stack")
    from mynotes.jobs.maintenance import create_backup

    settings = Settings(postgres={"dsn": migrated_dsn})
    out_dir = tmp_path / "backups"
    target = create_backup(settings, out_dir)
    assert target.parent == out_dir
    assert target.name.startswith("mynotes_") and target.name.endswith(".dump")
    data = target.read_bytes()
    assert data.startswith(b"PGDMP")
    assert len(data) > 1_000
