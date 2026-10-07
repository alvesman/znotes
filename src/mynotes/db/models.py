"""SQLAlchemy ORM models mirroring migrations ``0001_corpus``,
``0002_tags_relations_kg_ops`` and ``0003_kg_entity_distinct``.

The Alembic migrations are the source of truth for the DDL; these models match
them exactly and are used for queries. ``Base.metadata`` is handed to Alembic
as ``target_metadata`` for reference only; ``Base.metadata.create_all`` is
never used.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

_NOW = sa.text("now()")
_EMPTY_JSONB = sa.text("'{}'::jsonb")
_EMPTY_TEXT_ARRAY = sa.text("'{}'::text[]")


class Base(DeclarativeBase):
    """Declarative base for all mynotes tables."""


def pack_vector(vector: Sequence[float]) -> bytes:
    """Pack floats as little-endian float16 bytes for ``embedding_cache.vector``.

    ``array.array`` has no float16 typecode, so this uses ``struct`` format
    ``<e`` (IEEE 754 half precision, little-endian); numpy stays out of the
    runtime dependencies.

    Args:
        vector: The embedding components.

    Returns:
        Two bytes per component.

    Raises:
        OverflowError: If a component exceeds the float16 range.
    """
    return struct.pack(f"<{len(vector)}e", *vector)


def unpack_vector(packed: bytes) -> list[float]:
    """Unpack little-endian float16 bytes produced by :func:`pack_vector`.

    Args:
        packed: Two bytes per component.

    Returns:
        The embedding components as floats.

    Raises:
        ValueError: If ``packed`` does not hold a whole number of float16 values.
    """
    return list(struct.unpack(f"<{len(packed) // 2}e", packed))


class Workspace(Base):
    """Top-level container; every corpus row carries a workspace id."""

    __tablename__ = "workspace"
    __table_args__ = (sa.UniqueConstraint("name", name="uq_workspace_name"),)

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(always=True), primary_key=True)
    name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )


class SourceRoot(Base):
    """One watched folder tree inside a workspace."""

    __tablename__ = "source_root"
    __table_args__ = (
        sa.UniqueConstraint("workspace_id", "name", name="uq_source_root_workspace_name"),
    )

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(always=True), primary_key=True)
    workspace_id: Mapped[int] = mapped_column(sa.ForeignKey("workspace.id"), nullable=False)
    name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    abs_path: Mapped[str] = mapped_column(sa.Text, nullable=False)
    enabled: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, server_default=sa.true())
    case_sensitive: Mapped[bool | None] = mapped_column(sa.Boolean, nullable=True)
    config: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=_EMPTY_JSONB
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )


class ScanRun(Base):
    """One scan pass over a source root (incremental, full or targeted)."""

    __tablename__ = "scan_run"
    __table_args__ = (
        sa.CheckConstraint("kind IN ('incremental', 'full', 'targeted')", name="ck_scan_run_kind"),
        sa.CheckConstraint(
            "trigger IN ('schedule', 'manual', 'watch', 'startup')", name="ck_scan_run_trigger"
        ),
        sa.CheckConstraint("status IN ('running', 'done', 'failed')", name="ck_scan_run_status"),
        sa.Index("ix_scan_run_source_root_id", "source_root_id"),
    )

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(always=True), primary_key=True)
    source_root_id: Mapped[int] = mapped_column(sa.ForeignKey("source_root.id"), nullable=False)
    kind: Mapped[str] = mapped_column(sa.Text, nullable=False)
    trigger: Mapped[str] = mapped_column(sa.Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )
    finished_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="running")
    counts: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=_EMPTY_JSONB
    )
    error: Mapped[str | None] = mapped_column(sa.Text, nullable=True)


class Document(Base):
    """A stable logical document identity that survives edits, moves and renames."""

    __tablename__ = "document"
    __table_args__ = (
        sa.CheckConstraint(
            "status IN ('current', 'superseded', 'archived', 'deleted')", name="ck_document_status"
        ),
        sa.Index("ix_document_workspace_status", "workspace_id", "status"),
        sa.Index("ix_document_title_norm", "title_norm"),
        sa.Index("ix_document_source_root_id", "source_root_id"),
        sa.Index("ix_document_superseded_by_document_id", "superseded_by_document_id"),
    )

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(always=True), primary_key=True)
    workspace_id: Mapped[int] = mapped_column(sa.ForeignKey("workspace.id"), nullable=False)
    source_root_id: Mapped[int] = mapped_column(sa.ForeignKey("source_root.id"), nullable=False)
    current_file_id: Mapped[int | None] = mapped_column(
        sa.ForeignKey(
            "file.id",
            ondelete="SET NULL",
            use_alter=True,
            name="fk_document_current_file_id",
        ),
        nullable=True,
    )
    latest_version_id: Mapped[int | None] = mapped_column(
        sa.ForeignKey(
            "document_version.id",
            ondelete="SET NULL",
            use_alter=True,
            name="fk_document_latest_version_id",
        ),
        nullable=True,
    )
    title: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    title_norm: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    format: Mapped[str] = mapped_column(sa.Text, nullable=False)
    language: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    status: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="current")
    superseded_by_document_id: Mapped[int | None] = mapped_column(
        sa.ForeignKey("document.id"), nullable=True
    )
    doc_date: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )

    versions: Mapped[list[DocumentVersion]] = relationship(
        back_populates="document", foreign_keys="DocumentVersion.document_id", passive_deletes=True
    )


class File(Base):
    """A concrete file observed under a source root, keyed by exact path."""

    __tablename__ = "file"
    __table_args__ = (
        sa.CheckConstraint(
            "status IN ('active', 'missing', 'deleted', 'unsupported', 'placeholder')",
            name="ck_file_status",
        ),
        sa.Index(
            "uq_file_root_rel_path_active",
            "source_root_id",
            "rel_path",
            unique=True,
            postgresql_where=sa.text("status <> 'deleted'"),
        ),
        sa.Index("ix_file_source_root_rel_path_norm", "source_root_id", "rel_path_norm"),
        sa.Index("ix_file_content_hash", "content_hash"),
        sa.Index("ix_file_document_id", "document_id"),
    )

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(always=True), primary_key=True)
    source_root_id: Mapped[int] = mapped_column(sa.ForeignKey("source_root.id"), nullable=False)
    rel_path: Mapped[str] = mapped_column(sa.Text, nullable=False)
    rel_path_norm: Mapped[str] = mapped_column(sa.Text, nullable=False)
    size: Mapped[int] = mapped_column(sa.BigInteger, nullable=False)
    mtime_ns: Mapped[int] = mapped_column(sa.BigInteger, nullable=False)
    content_hash: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    format: Mapped[str] = mapped_column(sa.Text, nullable=False)
    status: Mapped[str] = mapped_column(sa.Text, nullable=False)
    missing_since: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )
    first_seen_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )
    deleted_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    document_id: Mapped[int | None] = mapped_column(
        sa.ForeignKey("document.id", use_alter=True, name="fk_file_document_id"), nullable=True
    )


class DocumentVersion(Base):
    """A content snapshot of one document; chunked and embedded independently."""

    __tablename__ = "document_version"
    __table_args__ = (
        sa.UniqueConstraint(
            "document_id", "version_no", name="uq_document_version_document_id_version_no"
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'extracted', 'chunked', 'embedded', 'failed', 'empty')",
            name="ck_document_version_status",
        ),
        sa.Index(
            "uq_document_version_latest",
            "document_id",
            unique=True,
            postgresql_where=sa.text("is_latest"),
        ),
    )

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(always=True), primary_key=True)
    document_id: Mapped[int] = mapped_column(
        sa.ForeignKey("document.id", ondelete="CASCADE"), nullable=False
    )
    version_no: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(sa.Text, nullable=False)
    rel_path_at_version: Mapped[str] = mapped_column(sa.Text, nullable=False)
    mtime_ns: Mapped[int] = mapped_column(sa.BigInteger, nullable=False)
    doc_date: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    frontmatter: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=_EMPTY_JSONB
    )
    ir_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    extractor: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    extractor_version: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    status: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="pending")
    error: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    is_latest: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, server_default=sa.true())
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )

    document: Mapped[Document] = relationship(
        back_populates="versions", foreign_keys="DocumentVersion.document_id"
    )
    chunks: Mapped[list[Chunk]] = relationship(back_populates="version")


class Chunk(Base):
    """One retrieval unit of a document version, with an optional uuidv7 id."""

    __tablename__ = "chunk"
    __table_args__ = (
        sa.Index(
            "uq_chunk_document_version_ordinal", "document_version_id", "ordinal", unique=True
        ),
        sa.Index("ix_chunk_document_section", "document_id", "section"),
        sa.Index("ix_chunk_content_hash", "content_hash"),
        sa.Index("ix_chunk_document_latest", "document_id", postgresql_where=sa.text("is_latest")),
    )

    id: Mapped[UUID] = mapped_column(sa.Uuid, primary_key=True, server_default=sa.text("uuidv7()"))
    document_version_id: Mapped[int] = mapped_column(
        sa.ForeignKey("document_version.id", ondelete="CASCADE"), nullable=False
    )
    document_id: Mapped[int] = mapped_column(
        sa.ForeignKey("document.id", ondelete="CASCADE"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(sa.Text, nullable=False)
    text: Mapped[str] = mapped_column(sa.Text, nullable=False)
    embed_text: Mapped[str] = mapped_column(sa.Text, nullable=False)
    page: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    slide: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    section: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    heading_path: Mapped[list[str]] = mapped_column(
        ARRAY(sa.Text), nullable=False, server_default=_EMPTY_TEXT_ARRAY
    )
    is_table: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, server_default=sa.false())
    token_count: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    language: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    is_latest: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, server_default=sa.true())
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )

    version: Mapped[DocumentVersion] = relationship(back_populates="chunks")


class EmbeddingCache(Base):
    """Cached embedding vectors keyed by model and content hash (float16 storage)."""

    __tablename__ = "embedding_cache"
    __table_args__ = (sa.PrimaryKeyConstraint("model", "content_hash", name="pk_embedding_cache"),)

    model: Mapped[str] = mapped_column(sa.Text, nullable=False, primary_key=True)
    content_hash: Mapped[str] = mapped_column(sa.Text, nullable=False, primary_key=True)
    dim: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    vector: Mapped[bytes] = mapped_column(sa.LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )


class Tag(Base):
    """A normalized tag in a workspace vocabulary (e.g. ``project:alpha``)."""

    __tablename__ = "tag"
    __table_args__ = (
        sa.UniqueConstraint("workspace_id", "name", name="uq_tag_workspace_name"),
        sa.CheckConstraint(
            "kind IN ('project', 'topic', 'status', 'folder', 'other')", name="ck_tag_kind"
        ),
    )

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(always=True), primary_key=True)
    workspace_id: Mapped[int] = mapped_column(sa.ForeignKey("workspace.id"), nullable=False)
    name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    display_name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    kind: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="other")
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )


class TagAlias(Base):
    """A normalized alias that resolves to a canonical tag (vocabulary control)."""

    __tablename__ = "tag_alias"
    __table_args__ = (
        sa.PrimaryKeyConstraint("workspace_id", "alias_norm", name="pk_tag_alias"),
        sa.Index("ix_tag_alias_tag_id", "tag_id"),
    )

    workspace_id: Mapped[int] = mapped_column(sa.ForeignKey("workspace.id"), primary_key=True)
    alias_norm: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    tag_id: Mapped[int] = mapped_column(sa.ForeignKey("tag.id", ondelete="CASCADE"), nullable=False)


class DocumentTag(Base):
    """One tag on one document, with its source and review status."""

    __tablename__ = "document_tag"
    __table_args__ = (
        sa.PrimaryKeyConstraint("document_id", "tag_id", "source", name="pk_document_tag"),
        sa.CheckConstraint(
            "source IN ('folder', 'frontmatter', 'hashtag', 'llm', 'manual')",
            name="ck_document_tag_source",
        ),
        sa.CheckConstraint(
            "status IN ('accepted', 'proposed', 'rejected')", name="ck_document_tag_status"
        ),
        sa.Index("ix_document_tag_tag_status", "tag_id", "status"),
        sa.Index("ix_document_tag_document_version_id", "document_version_id"),
    )

    document_id: Mapped[int] = mapped_column(
        sa.ForeignKey("document.id", ondelete="CASCADE"), primary_key=True
    )
    tag_id: Mapped[int] = mapped_column(
        sa.ForeignKey("tag.id", ondelete="CASCADE"), primary_key=True
    )
    source: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    confidence: Mapped[float] = mapped_column(
        sa.REAL, nullable=False, server_default=sa.text("1.0")
    )
    status: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="accepted")
    document_version_id: Mapped[int | None] = mapped_column(
        sa.ForeignKey("document_version.id", ondelete="SET NULL"), nullable=True
    )
    reason: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )


class DocRelation(Base):
    """A directed edge between two documents in the structural graph."""

    __tablename__ = "doc_relation"
    __table_args__ = (
        sa.UniqueConstraint(
            "src_document_id", "dst_document_id", "type", name="uq_doc_relation_src_dst_type"
        ),
        sa.CheckConstraint(
            "type IN ('supersedes', 'links_to', 'duplicate_of', 'derived_from', 'same_project')",
            name="ck_doc_relation_type",
        ),
        sa.CheckConstraint(
            "method IN ('frontmatter', 'name_content', 'hash', 'manual', 'link_parse')",
            name="ck_doc_relation_method",
        ),
        sa.CheckConstraint(
            "status IN ('auto', 'proposed', 'confirmed', 'rejected')",
            name="ck_doc_relation_status",
        ),
        sa.Index("ix_doc_relation_dst_document_id_type", "dst_document_id", "type"),
        sa.Index("ix_doc_relation_status", "status"),
    )

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(always=True), primary_key=True)
    src_document_id: Mapped[int] = mapped_column(
        sa.ForeignKey("document.id", ondelete="CASCADE"), nullable=False
    )
    dst_document_id: Mapped[int] = mapped_column(
        sa.ForeignKey("document.id", ondelete="CASCADE"), nullable=False
    )
    type: Mapped[str] = mapped_column(sa.Text, nullable=False)
    method: Mapped[str] = mapped_column(sa.Text, nullable=False)
    confidence: Mapped[float] = mapped_column(sa.REAL, nullable=False)
    status: Mapped[str] = mapped_column(sa.Text, nullable=False)
    evidence: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=_EMPTY_JSONB
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )


class KgEntity(Base):
    """A resolved real-world thing (person, project, tool, ...) in a workspace."""

    __tablename__ = "kg_entity"
    __table_args__ = (
        sa.CheckConstraint("status IN ('active', 'merged', 'stale')", name="ck_kg_entity_status"),
        sa.Index("ix_kg_entity_workspace_type_norm_name", "workspace_id", "type", "norm_name"),
        # One active entity per (workspace, type, normalized name): concurrent resolution
        # jobs must link to the first insert instead of creating a duplicate (migration 0004).
        sa.Index(
            "uq_kg_entity_active_workspace_type_norm_name",
            "workspace_id",
            "type",
            "norm_name",
            unique=True,
            postgresql_where=sa.text("status = 'active'"),
        ),
        sa.Index(
            "ix_kg_entity_norm_name_trgm",
            "norm_name",
            postgresql_using="gin",
            postgresql_ops={"norm_name": "gin_trgm_ops"},
        ),
        sa.Index("ix_kg_entity_merged_into_id", "merged_into_id"),
    )

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(always=True), primary_key=True)
    workspace_id: Mapped[int] = mapped_column(sa.ForeignKey("workspace.id"), nullable=False)
    type: Mapped[str] = mapped_column(sa.Text, nullable=False)
    canonical_name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    norm_name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    description: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    summary_stale: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.true()
    )
    mention_count_current: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default=sa.text("0")
    )
    status: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="active")
    merged_into_id: Mapped[int | None] = mapped_column(sa.ForeignKey("kg_entity.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )


class KgEntityAlias(Base):
    """A surface name that resolves to a canonical KG entity."""

    __tablename__ = "kg_entity_alias"
    __table_args__ = (
        sa.PrimaryKeyConstraint("entity_id", "alias_norm", name="pk_kg_entity_alias"),
        sa.Index(
            "ix_kg_entity_alias_alias_norm_trgm",
            "alias_norm",
            postgresql_using="gin",
            postgresql_ops={"alias_norm": "gin_trgm_ops"},
        ),
    )

    entity_id: Mapped[int] = mapped_column(
        sa.ForeignKey("kg_entity.id", ondelete="CASCADE"), primary_key=True
    )
    alias: Mapped[str] = mapped_column(sa.Text, nullable=False)
    alias_norm: Mapped[str] = mapped_column(sa.Text, primary_key=True)


class KgMention(Base):
    """One occurrence of an entity inside a chunk."""

    __tablename__ = "kg_mention"
    __table_args__ = (
        sa.Index("ix_kg_mention_chunk_id", "chunk_id"),
        sa.Index("ix_kg_mention_entity_id", "entity_id"),
        sa.Index("ix_kg_mention_document_version_id", "document_version_id"),
    )

    id: Mapped[UUID] = mapped_column(sa.Uuid, primary_key=True, server_default=sa.text("uuidv7()"))
    entity_id: Mapped[int] = mapped_column(
        sa.ForeignKey("kg_entity.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("chunk.id", ondelete="CASCADE"), nullable=False
    )
    document_version_id: Mapped[int] = mapped_column(
        sa.ForeignKey("document_version.id", ondelete="CASCADE"), nullable=False
    )
    surface_form: Mapped[str] = mapped_column(sa.Text, nullable=False)
    confidence: Mapped[float] = mapped_column(
        sa.REAL, nullable=False, server_default=sa.text("1.0")
    )


class KgRelation(Base):
    """A directed edge between two KG entities, optionally a temporal attribute fact."""

    __tablename__ = "kg_relation"
    __table_args__ = (
        sa.UniqueConstraint(
            "src_entity_id",
            "dst_entity_id",
            "type",
            "attribute",
            "value",
            name="uq_kg_relation_edge",
            postgresql_nulls_not_distinct=True,
        ),
        sa.CheckConstraint(
            "status IN ('active', 'historical', 'stale')", name="ck_kg_relation_status"
        ),
        sa.Index("ix_kg_relation_src_entity_id_status", "src_entity_id", "status"),
        sa.Index("ix_kg_relation_dst_entity_id_status", "dst_entity_id", "status"),
        sa.Index("ix_kg_relation_workspace_id", "workspace_id"),
    )

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(always=True), primary_key=True)
    workspace_id: Mapped[int] = mapped_column(sa.ForeignKey("workspace.id"), nullable=False)
    src_entity_id: Mapped[int] = mapped_column(
        sa.ForeignKey("kg_entity.id", ondelete="CASCADE"), nullable=False
    )
    dst_entity_id: Mapped[int] = mapped_column(
        sa.ForeignKey("kg_entity.id", ondelete="CASCADE"), nullable=False
    )
    type: Mapped[str] = mapped_column(sa.Text, nullable=False)
    raw_label: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    description: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    attribute: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    value: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    weight: Mapped[float] = mapped_column(sa.REAL, nullable=False, server_default=sa.text("1.0"))
    valid_from: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    valid_to: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="active")
    first_seen_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )


class KgRelationEvidence(Base):
    """A chunk that backs one KG relation, with its document date for temporal validity."""

    __tablename__ = "kg_relation_evidence"
    __table_args__ = (
        sa.Index("ix_kg_relation_evidence_relation_id", "relation_id"),
        sa.Index("ix_kg_relation_evidence_chunk_id", "chunk_id"),
        sa.Index("ix_kg_relation_evidence_document_version_id", "document_version_id"),
    )

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(always=True), primary_key=True)
    relation_id: Mapped[int] = mapped_column(
        sa.ForeignKey("kg_relation.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("chunk.id", ondelete="CASCADE"), nullable=False
    )
    document_version_id: Mapped[int] = mapped_column(
        sa.ForeignKey("document_version.id", ondelete="CASCADE"), nullable=False
    )
    doc_date: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    quote: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    confidence: Mapped[float] = mapped_column(
        sa.REAL, nullable=False, server_default=sa.text("1.0")
    )
    is_current: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, server_default=sa.true())


class KgExtractionCache(Base):
    """Cached LLM extraction results keyed by content hash and extractor version."""

    __tablename__ = "kg_extraction_cache"
    __table_args__ = (
        sa.PrimaryKeyConstraint("content_hash", "extractor_version", name="pk_kg_extraction_cache"),
    )

    content_hash: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    extractor_version: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    model: Mapped[str] = mapped_column(sa.Text, nullable=False)
    result: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )


class Job(Base):
    """One unit of background work, claimed by workers via locking columns."""

    __tablename__ = "job"
    __table_args__ = (
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'done', 'failed', 'dead')", name="ck_job_status"
        ),
        sa.Index(
            "uq_job_dedupe_key_active",
            "dedupe_key",
            unique=True,
            postgresql_where=sa.text("status IN ('queued', 'running')"),
        ),
        sa.Index("ix_job_status_priority_run_after", "status", "priority", "run_after"),
    )

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(always=True), primary_key=True)
    type: Mapped[str] = mapped_column(sa.Text, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=_EMPTY_JSONB
    )
    dedupe_key: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    priority: Mapped[int] = mapped_column(sa.Integer, nullable=False, server_default=sa.text("100"))
    status: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="queued")
    attempts: Mapped[int] = mapped_column(sa.Integer, nullable=False, server_default=sa.text("0"))
    max_attempts: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default=sa.text("5")
    )
    run_after: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )
    locked_by: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    locked_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )
    finished_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)


class IndexOutbox(Base):
    """A pending upsert/delete to mirror from Postgres into an OpenSearch index."""

    __tablename__ = "index_outbox"
    __table_args__ = (
        sa.CheckConstraint("doc_kind IN ('chunk', 'entity')", name="ck_index_outbox_doc_kind"),
        sa.CheckConstraint("op IN ('upsert', 'delete')", name="ck_index_outbox_op"),
        sa.Index("ix_index_outbox_pending", "id", postgresql_where=sa.text("processed_at IS NULL")),
    )

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(always=True), primary_key=True)
    index_name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    doc_kind: Mapped[str] = mapped_column(sa.Text, nullable=False)
    doc_id: Mapped[str] = mapped_column(sa.Text, nullable=False)
    op: Mapped[str] = mapped_column(sa.Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )
    processed_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    attempts: Mapped[int] = mapped_column(sa.Integer, nullable=False, server_default=sa.text("0"))
    last_error: Mapped[str | None] = mapped_column(sa.Text, nullable=True)


class AclGrant(Base):
    """A team-mode read grant for one principal under one path prefix."""

    __tablename__ = "acl_grant"
    __table_args__ = (
        sa.UniqueConstraint(
            "workspace_id", "principal", "path_prefix", name="uq_acl_grant_workspace_principal_path"
        ),
        sa.CheckConstraint("permission IN ('read')", name="ck_acl_grant_permission"),
    )

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(always=True), primary_key=True)
    workspace_id: Mapped[int] = mapped_column(sa.ForeignKey("workspace.id"), nullable=False)
    principal: Mapped[str] = mapped_column(sa.Text, nullable=False)
    path_prefix: Mapped[str] = mapped_column(sa.Text, nullable=False)
    permission: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="read")
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )


class KgEntityDistinct(Base):
    """A recorded "these two entities are different" decision (task 41).

    The pair is stored with ``a_id < b_id`` so lookups normalize with
    ``(min(id0, id1), max(id0, id1))``. A recorded pair never links or
    adjudicates again (entity resolution) and is written by entity
    adjudication and by :func:`mynotes.graph.store.unmerge_entity`.
    """

    __tablename__ = "kg_entity_distinct"
    __table_args__ = (
        sa.PrimaryKeyConstraint("a_id", "b_id", name="pk_kg_entity_distinct"),
        sa.CheckConstraint("a_id < b_id", name="ck_kg_entity_distinct_ordered"),
        sa.Index("ix_kg_entity_distinct_b_id", "b_id"),
    )

    a_id: Mapped[int] = mapped_column(
        sa.ForeignKey("kg_entity.id", ondelete="CASCADE"), primary_key=True
    )
    b_id: Mapped[int] = mapped_column(
        sa.ForeignKey("kg_entity.id", ondelete="CASCADE"), primary_key=True
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )


class EndpointStat(Base):
    """A model-endpoint snapshot published by one process role (migration 0005).

    The worker publishes its guard snapshots every 30 seconds (and on stop) so
    the Status page, which runs in the web process, can show the calls the
    worker makes; one row per ``(process_role, name)``.
    """

    __tablename__ = "endpoint_stat"
    __table_args__ = (sa.PrimaryKeyConstraint("process_role", "name", name="pk_endpoint_stat"),)

    process_role: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    name: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    state: Mapped[str] = mapped_column(sa.Text, nullable=False)
    calls: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    error_rate: Mapped[float] = mapped_column(sa.REAL, nullable=False)
    p50_ms: Mapped[float | None] = mapped_column(sa.REAL, nullable=True)
    p95_ms: Mapped[float | None] = mapped_column(sa.REAL, nullable=True)
    last_error: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    host: Mapped[str] = mapped_column(sa.Text, nullable=False)
    pid: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )


class ChatEntry(Base):
    """One chat question and its answer, kept for the History page (migration 0006).

    Written by the streaming ask endpoint when the answer completes; scoped to
    the caller's ``user`` so team members only see their own chats. The turns of
    one chat page share a ``conversation_id`` (migration 0007).
    """

    __tablename__ = "chat_entry"
    __table_args__ = (
        sa.Index("ix_chat_entry_user_created_at", "user_name", "created_at"),
        sa.Index("ix_chat_entry_user_conversation", "user_name", "conversation_id"),
    )

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(always=True), primary_key=True)
    user_name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    conversation_id: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    question: Mapped[str] = mapped_column(sa.Text, nullable=False)
    standalone_question: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    answer: Mapped[str] = mapped_column(sa.Text, nullable=False)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")
    )
    filters: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")
    )
    relevant: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, server_default=sa.true())
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=_NOW
    )
