"""Integration tests for migration ``0001_corpus`` (corpus schema, Postgres 18)."""

from __future__ import annotations

import uuid

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from mynotes.db.models import Chunk, Document, DocumentVersion, File, SourceRoot, Workspace
from mynotes.db.session import session_scope

pytestmark = pytest.mark.integration

EXPECTED_TABLES = {
    "workspace",
    "source_root",
    "scan_run",
    "document",
    "file",
    "document_version",
    "chunk",
    "embedding_cache",
}


def _add_document_graph(session: Session) -> tuple[SourceRoot, Document, DocumentVersion]:
    """Insert a workspace-rooted document with one version, flushing along the way."""
    workspace = session.execute(select(Workspace).where(Workspace.name == "default")).scalar_one()
    root = SourceRoot(workspace_id=workspace.id, name="notes", abs_path="/tmp/notes")
    session.add(root)
    session.flush()
    document = Document(workspace_id=workspace.id, source_root_id=root.id, format="md")
    session.add(document)
    session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version_no=1,
        content_hash="hash-1",
        rel_path_at_version="notes/a.md",
        mtime_ns=1_000,
    )
    session.add(version)
    session.flush()
    return root, document, version


def _file(source_root_id: int, rel_path: str, status: str) -> File:
    """Build a minimal file row with the given status."""
    return File(
        source_root_id=source_root_id,
        rel_path=rel_path,
        rel_path_norm=rel_path,
        size=10,
        mtime_ns=1,
        format="md",
        status=status,
    )


def test_upgrade_creates_all_tables_and_default_workspace(db_session: Session) -> None:
    """After the migration the 8 corpus tables exist and workspace 'default' is present."""
    tables = set(
        db_session.execute(
            text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        ).scalars()
    )
    assert tables >= EXPECTED_TABLES
    assert (
        db_session.execute(
            text("SELECT count(*) FROM workspace WHERE name = 'default'")
        ).scalar_one()
        == 1
    )


def test_f_unaccent_strips_accents(db_session: Session) -> None:
    """The immutable unaccent wrapper folds 'Acao' input to unaccented output."""
    assert db_session.execute(text("SELECT f_unaccent('Ação')")).scalar_one() == "Acao"


def test_chunk_ids_are_uuidv7_and_insertion_ordered(db_session: Session) -> None:
    """Chunks without an id get uuidv7 values that sort in insertion order."""
    _, document, version = _add_document_graph(db_session)
    first = Chunk(
        document_version_id=version.id,
        document_id=document.id,
        ordinal=0,
        content_hash="hash-0",
        text="alpha",
        embed_text="alpha",
        token_count=1,
    )
    db_session.add(first)
    db_session.flush()
    second = Chunk(
        document_version_id=version.id,
        document_id=document.id,
        ordinal=1,
        content_hash="hash-1b",
        text="beta",
        embed_text="beta",
        token_count=1,
    )
    db_session.add(second)
    db_session.flush()

    assert isinstance(first.id, uuid.UUID)
    assert isinstance(second.id, uuid.UUID)
    assert first.id != second.id
    assert first.id < second.id


def test_partial_unique_rejects_duplicate_active_files(db_session: Session) -> None:
    """Two non-deleted files with the same path collide; deleted plus active is allowed."""
    root, _, _ = _add_document_graph(db_session)
    db_session.add(_file(root.id, "notes/a.md", "active"))
    db_session.flush()
    with pytest.raises(IntegrityError), db_session.begin_nested():
        db_session.add(_file(root.id, "notes/a.md", "active"))
        db_session.flush()

    db_session.add(_file(root.id, "notes/b.md", "active"))
    db_session.add(_file(root.id, "notes/b.md", "deleted"))
    db_session.flush()


def test_partial_unique_rejects_two_latest_versions(db_session: Session) -> None:
    """A second is_latest version for one document violates the partial unique index."""
    _, document, version = _add_document_graph(db_session)
    assert version.is_latest
    with pytest.raises(IntegrityError), db_session.begin_nested():
        db_session.add(
            DocumentVersion(
                document_id=document.id,
                version_no=2,
                content_hash="hash-2",
                rel_path_at_version="notes/a.md",
                mtime_ns=2_000,
            )
        )
        db_session.flush()

    superseded = DocumentVersion(
        document_id=document.id,
        version_no=2,
        content_hash="hash-2",
        rel_path_at_version="notes/a.md",
        mtime_ns=2_000,
        is_latest=False,
    )
    db_session.add(superseded)
    db_session.flush()


def test_check_constraint_rejects_unknown_file_status(db_session: Session) -> None:
    """A file.status outside the CHECK list is rejected."""
    root, _, _ = _add_document_graph(db_session)
    db_session.add(_file(root.id, "notes/c.md", "weird"))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_session_scope_commits(settings_with_db: object) -> None:
    """session_scope commits on success through the cached engine."""
    with session_scope() as session:
        workspace = session.execute(
            select(Workspace).where(Workspace.name == "default")
        ).scalar_one()
        session.add(
            SourceRoot(workspace_id=workspace.id, name="persisted", abs_path="/tmp/persisted")
        )
    with session_scope() as session:
        names = (
            session.execute(select(SourceRoot.name).where(SourceRoot.name == "persisted"))
            .scalars()
            .all()
        )
    assert names == ["persisted"]


def test_downgrade_base_then_upgrade_head(alembic_cfg: Config) -> None:
    """The migration can be fully reverted and reapplied."""
    command.downgrade(alembic_cfg, "base")
    command.upgrade(alembic_cfg, "head")
