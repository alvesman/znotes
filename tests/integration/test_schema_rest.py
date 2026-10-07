"""Integration tests for migration ``0002_tags_relations_kg_ops`` (Postgres 18)."""

from __future__ import annotations

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from mynotes.db.models import (
    AclGrant,
    DocRelation,
    Document,
    DocumentTag,
    DocumentVersion,
    IndexOutbox,
    Job,
    KgEntity,
    KgRelation,
    SourceRoot,
    Tag,
    Workspace,
)

pytestmark = pytest.mark.integration

EXPECTED_TABLES = {
    "tag",
    "tag_alias",
    "document_tag",
    "doc_relation",
    "kg_entity",
    "kg_entity_alias",
    "kg_mention",
    "kg_relation",
    "kg_relation_evidence",
    "kg_extraction_cache",
    "job",
    "index_outbox",
    "acl_grant",
}


def _add_document_graph(session: Session) -> tuple[Workspace, Document, DocumentVersion]:
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
    return workspace, document, version


def _add_entity_pair(session: Session, workspace_id: int) -> tuple[KgEntity, KgEntity]:
    """Insert two KG entities in the workspace, flushing along the way."""
    src = KgEntity(workspace_id=workspace_id, type="person", canonical_name="Ada", norm_name="ada")
    session.add(src)
    session.flush()
    dst = KgEntity(
        workspace_id=workspace_id, type="project", canonical_name="Alpha", norm_name="alpha"
    )
    session.add(dst)
    session.flush()
    return src, dst


def test_upgrade_creates_all_tables(db_session: Session) -> None:
    """After the migration the 13 new tables exist."""
    tables = set(
        db_session.execute(
            text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        ).scalars()
    )
    assert tables >= EXPECTED_TABLES


def test_job_partial_unique_dedupe_key(db_session: Session) -> None:
    """A second queued job with the same dedupe_key collides; done frees the key."""
    first = Job(type="embed_document", dedupe_key="doc:42")
    db_session.add(first)
    db_session.flush()
    with pytest.raises(IntegrityError), db_session.begin_nested():
        db_session.add(Job(type="embed_document", dedupe_key="doc:42"))
        db_session.flush()

    first.status = "done"
    db_session.flush()
    second = Job(type="embed_document", dedupe_key="doc:42")
    db_session.add(second)
    db_session.flush()
    assert second.id != first.id


def test_job_defaults(db_session: Session) -> None:
    """New jobs queue with priority 100, attempts 0, max_attempts 5 and run_after set."""
    job = Job(type="embed_document")
    db_session.add(job)
    db_session.flush()

    assert job.status == "queued"
    assert job.priority == 100
    assert job.attempts == 0
    assert job.max_attempts == 5
    assert job.run_after is not None


def test_kg_relation_nulls_not_distinct(db_session: Session) -> None:
    """A duplicate (src, dst, type, NULL, NULL) edge is rejected."""
    workspace = db_session.execute(
        select(Workspace).where(Workspace.name == "default")
    ).scalar_one()
    src, dst = _add_entity_pair(db_session, workspace.id)
    db_session.add(
        KgRelation(
            workspace_id=workspace.id, src_entity_id=src.id, dst_entity_id=dst.id, type="works_on"
        )
    )
    db_session.flush()
    with pytest.raises(IntegrityError), db_session.begin_nested():
        db_session.add(
            KgRelation(
                workspace_id=workspace.id,
                src_entity_id=src.id,
                dst_entity_id=dst.id,
                type="works_on",
            )
        )
        db_session.flush()

    db_session.add(
        KgRelation(
            workspace_id=workspace.id,
            src_entity_id=src.id,
            dst_entity_id=dst.id,
            type="works_on",
            attribute="role",
            value="lead",
        )
    )
    db_session.flush()


def test_trigram_index_used_for_similarity(db_session: Session) -> None:
    """The similarity query on kg_entity.norm_name uses the GIN trigram index."""
    db_session.execute(text("SET enable_seqscan = off"))
    plan = (
        db_session.execute(text("EXPLAIN SELECT id FROM kg_entity WHERE norm_name % 'alpha'"))
        .scalars()
        .all()
    )
    assert any("ix_kg_entity_norm_name_trgm" in line for line in plan)


def test_document_delete_cascades_to_tags_and_relations(db_session: Session) -> None:
    """Deleting a document removes its document_tag and doc_relation rows."""
    workspace, document, _ = _add_document_graph(db_session)
    tag = Tag(workspace_id=workspace.id, name="project:alpha", display_name="Project Alpha")
    db_session.add(tag)
    db_session.flush()
    db_session.add(DocumentTag(document_id=document.id, tag_id=tag.id, source="manual"))
    other = Document(workspace_id=workspace.id, source_root_id=document.source_root_id, format="md")
    db_session.add(other)
    db_session.flush()
    db_session.add(
        DocRelation(
            src_document_id=document.id,
            dst_document_id=other.id,
            type="links_to",
            method="manual",
            confidence=1.0,
            status="auto",
        )
    )
    db_session.flush()

    db_session.delete(document)
    db_session.flush()

    assert db_session.execute(text("SELECT count(*) FROM document_tag")).scalar_one() == 0
    assert db_session.execute(text("SELECT count(*) FROM doc_relation")).scalar_one() == 0
    assert db_session.get(Tag, tag.id) is not None


def test_check_constraint_rejects_unknown_document_tag_source(db_session: Session) -> None:
    """A document_tag.source outside the CHECK list is rejected."""
    workspace, document, _ = _add_document_graph(db_session)
    tag = Tag(workspace_id=workspace.id, name="topic:graphs", display_name="Graphs")
    db_session.add(tag)
    db_session.flush()
    db_session.add(DocumentTag(document_id=document.id, tag_id=tag.id, source="magic"))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_kg_entity_defaults(db_session: Session) -> None:
    """New entities start active with a stale summary and zero current mentions."""
    workspace = db_session.execute(
        select(Workspace).where(Workspace.name == "default")
    ).scalar_one()
    entity = KgEntity(workspace_id=workspace.id, type="tool", canonical_name="uv", norm_name="uv")
    db_session.add(entity)
    db_session.flush()

    assert entity.status == "active"
    assert entity.summary_stale is True
    assert entity.mention_count_current == 0


def test_index_outbox_and_acl_grant_inserts(db_session: Session) -> None:
    """The operations tables accept representative rows with their defaults."""
    workspace = db_session.execute(
        select(Workspace).where(Workspace.name == "default")
    ).scalar_one()
    outbox = IndexOutbox(index_name="chunks", doc_kind="chunk", doc_id="0190ab", op="upsert")
    db_session.add(outbox)
    grant = AclGrant(workspace_id=workspace.id, principal="user:manuel", path_prefix="projects/")
    db_session.add(grant)
    db_session.flush()

    assert outbox.attempts == 0
    assert outbox.processed_at is None
    assert grant.permission == "read"


def test_downgrade_0001_corpus_then_upgrade_head(alembic_cfg: Config) -> None:
    """The migration can be reverted to the corpus schema and reapplied."""
    command.downgrade(alembic_cfg, "0001_corpus")
    command.upgrade(alembic_cfg, "head")
