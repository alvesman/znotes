"""The one service facade behind every interface (PLAN.md §3).

The CLI, the HTTP API, the MCP server and the web UI call these plain
functions; no business logic lives in the interfaces. Every function takes a
:class:`~mynotes.core.principal.Principal` and applies its ``acl_prefixes`` to
every retrieval or document access; ``LOCAL`` (the default) is unrestricted.
Read functions open their own :func:`~mynotes.db.session.session_scope`
transaction, so callers never see a session.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import exists, func, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, aliased

from mynotes.config import get_settings
from mynotes.core.acl import (
    allowed_entity_ids as _allowed_entity_ids,
)
from mynotes.core.acl import filter_subgraph_by_acl as _filter_subgraph_by_acl
from mynotes.core.acl import (
    path_allowed as _path_allowed,
)
from mynotes.core.acl import (
    relation_has_allowed_evidence as _relation_has_allowed_evidence,
)
from mynotes.core.errors import Ambiguous, Forbidden, NotFound
from mynotes.core.principal import LOCAL, Principal
from mynotes.core.setup import collect_status
from mynotes.db.models import (
    Chunk,
    DocRelation,
    Document,
    DocumentTag,
    DocumentVersion,
    File,
    Job,
    KgEntity,
    KgEntityDistinct,
    KgMention,
    KgRelation,
    KgRelationEvidence,
    ScanRun,
    SourceRoot,
    Tag,
    Workspace,
)
from mynotes.db.session import session_scope
from mynotes.extract.ir import IRDocument
from mynotes.graph.expand import Subgraph, fetch_subgraph, score_entities
from mynotes.graph.store import entity_card as build_entity_card
from mynotes.graph.store import find_entities
from mynotes.graph.store import merge_entities as merge_entity_rows
from mynotes.graph.structural import DocNode, find_document
from mynotes.graph.temporal import (
    AttributeFact,
    AttributeValue,
    TimelineItem,
    _attribute_evidence,
    pick_current_attribute,
    project_entity_id,
    tag_timeline,
)
from mynotes.graph.temporal import timeline as entity_timeline
from mynotes.platform.open import open_in_default_app
from mynotes.platform.paths import from_rel_path, is_within, long_path
from mynotes.retrieval.answer import AnswerEvent, AnswerResult, answer, answer_stream
from mynotes.retrieval.pipeline import RetrievalResult, retrieve
from mynotes.retrieval.types import SearchFilters
from mynotes.scan.scanner import ScanSummary, scan_root
from mynotes.tags.autotag import (
    DocumentProposals,
    decide_proposal,
    pending_proposals,
    tag_id_by_name,
)
from mynotes.tags.store import (
    TagDetail,
    TagSummary,
    document_ids_with_tag,
    document_tag_details,
    tag_summaries,
)
from mynotes.tags.store import merge_tags as merge_tag_names
from mynotes.versioning.history import VersionInfo
from mynotes.versioning.history import document_history as document_version_history
from mynotes.versioning.review import RelationView, confirm, list_relations, reject, swap
from mynotes.versioning.review import document_preview as document_preview_text

logger = logging.getLogger(__name__)

TagView = TagSummary
"""The facade's tag view (the store's :class:`TagSummary`)."""

__all__ = [
    "LOCAL",
    "Ambiguous",
    "DocumentView",
    "EntityPair",
    "NotFound",
    "Principal",
    "RelatedDocument",
    "RelationView",
    "ResolvedFile",
    "SectionRef",
    "TagDetail",
    "TagView",
    "TimelineItem",
    "VersionInfo",
    "ask",
    "ask_stream",
    "document_chunk_text",
    "document_history",
    "document_previews",
    "document_tags",
    "entity_attributes",
    "entity_card",
    "entity_merge_candidates",
    "entity_neighbors",
    "filter_graph_by_acl",
    "get_document",
    "list_tags",
    "list_version_relations",
    "mark_entities_distinct",
    "merge_entities",
    "merge_tags",
    "open_original",
    "resolve_file",
    "search",
    "status",
    "tag_action",
    "tag_proposals",
    "timeline",
    "trigger_scan",
    "version_action",
    "version_chain",
]


@dataclass
class RelatedDocument:
    """One end of an active supersedes relation (task 48 document view).

    Attributes:
        document_id: The document id.
        title: The extracted title, or None.
        path: The current file path relative to its source root.
        doc_date: The document date, or None.
        status: The document status (``current``, ``superseded``, ...).
    """

    document_id: int
    title: str | None
    path: str
    doc_date: datetime | None
    status: str


@dataclass(frozen=True)
class EntityPair:
    """Two same-type entities with similar names and no distinct decision.

    Attributes:
        a_id: The lower entity id (the pair's canonical order).
        b_id: The higher entity id.
        a_name: The first entity's canonical name.
        b_name: The second entity's canonical name.
        a_type: The first entity's type.
        b_type: The second entity's type.
        a_created_at: When the first entity was created.
        b_created_at: When the second entity was created.
        similarity: The trigram similarity of the two normalized names.
    """

    a_id: int
    b_id: int
    a_name: str
    b_name: str
    a_type: str
    b_type: str
    a_created_at: datetime
    b_created_at: datetime
    similarity: float


@dataclass
class SectionRef:
    """One section of a document's latest version (its first chunk position).

    Attributes:
        section: The section label, or None.
        page: The PDF page number, or None.
        slide: The PPTX slide number, or None.
        first_ordinal: The ordinal of the section's first chunk.
    """

    section: str | None
    page: int | None
    slide: int | None
    first_ordinal: int


@dataclass
class DocumentView:
    """One document prepared for display (HTTP, MCP, web UI).

    Attributes:
        document_id: The document id.
        title: The extracted title, or None.
        path: The current file path relative to its source root (empty when
            the document has no file row anymore).
        format: The file format (``md``, ``pdf``, ...).
        status: ``current``, ``superseded``, ``archived`` or ``deleted``.
        doc_date: The document date, or None.
        language: The detected language, or None.
        tags: The document's accepted tag names.
        latest_version_no: The latest snapshot number, or None.
        markdown: The extracted text with headings rendered as Markdown.
        sections: The latest version's sections in document order.
        openable: Whether "Open in app" can work here: false for remote roots,
            whose originals live on the owner's machine.
    """

    document_id: int
    title: str | None
    path: str
    format: str
    status: str
    doc_date: datetime | None
    language: str | None
    tags: list[str]
    latest_version_no: int | None
    markdown: str
    sections: list[SectionRef] = field(default_factory=list)
    openable: bool = True


def _apply_acl(filters: SearchFilters | None, principal: Principal) -> SearchFilters:
    """Merge the principal's prefixes into the filters (a shallow copy)."""
    merged = replace(filters) if filters is not None else SearchFilters()
    if principal.acl_prefixes is None:
        return merged
    if merged.acl_prefixes is None:
        merged.acl_prefixes = list(principal.acl_prefixes)
    else:
        allowed = set(principal.acl_prefixes)
        merged.acl_prefixes = [prefix for prefix in merged.acl_prefixes if prefix in allowed]
    return merged


def _ensure_document_allowed(path: str, principal: Principal) -> None:
    """Raise :class:`Forbidden` unless the path is inside the principal's prefixes."""
    if principal.acl_prefixes is None:
        return
    if not _path_allowed(path, principal.acl_prefixes):
        raise Forbidden("the document is outside your allowed folders")


def _require_admin(principal: Principal) -> None:
    """Raise :class:`Forbidden` unless the caller may run mutations (task 52)."""
    if not principal.is_admin:
        raise Forbidden("this action requires an admin principal")


def _candidates(nodes: Sequence[DocNode]) -> list[dict[str, Any]]:
    return [
        {"document_id": node.document_id, "title": node.title, "path": node.path} for node in nodes
    ]


def _resolve_document(session: Session, ref: str | int) -> tuple[int, str]:
    """Resolve an id, a path or a title to ``(document_id, path)``.

    Raises:
        NotFound: When nothing matches.
        Ambiguous: When several documents match.
    """
    if isinstance(ref, int):
        row = session.execute(
            select(Document, File.rel_path)
            .outerjoin(File, File.id == Document.current_file_id)
            .where(Document.id == ref, Document.status != "deleted")
        ).first()
        if row is None:
            raise NotFound(f"no document with id {ref}")
        document, path = row
        return int(document.id), path or ""
    cleaned = ref.strip()
    if not cleaned:
        raise NotFound("empty document reference")
    nodes = find_document(session, cleaned)
    if len(nodes) == 1:
        return int(nodes[0].document_id), nodes[0].path
    if len(nodes) > 1:
        raise Ambiguous(f"{len(nodes)} documents match {cleaned!r}", _candidates(nodes))
    if cleaned.isdigit():
        return _resolve_document(session, int(cleaned))
    raise NotFound(f"no document matches {cleaned!r}")


def _ir_markdown(ir: IRDocument) -> str:
    """Render the IR blocks as Markdown text (headings keep their level)."""
    parts: list[str] = []
    for block in ir.blocks:
        if block.kind == "heading" and block.level:
            parts.append(f"{'#' * block.level} {block.text}")
        else:
            parts.append(block.text)
    return "\n\n".join(parts)


def _document_view(session: Session, document_id: int) -> DocumentView:
    """Build the :class:`DocumentView` for a resolved document."""
    row = session.execute(
        select(Document, File.rel_path, SourceRoot.kind)
        .outerjoin(File, File.id == Document.current_file_id)
        .join(SourceRoot, SourceRoot.id == Document.source_root_id)
        .where(Document.id == document_id)
    ).first()
    if row is None:
        raise NotFound(f"no document with id {document_id}")
    document, path, root_kind = row
    tags = _accepted_tags(session, document_id)
    latest_version_no: int | None = None
    markdown = ""
    sections: list[SectionRef] = []
    version: DocumentVersion | None = (
        session.get(DocumentVersion, document.latest_version_id)
        if document.latest_version_id is not None
        else None
    )
    if version is not None:
        latest_version_no = int(version.version_no)
        if version.ir_json is not None:
            markdown = _ir_markdown(IRDocument.from_json(version.ir_json))
        sections = _version_sections(session, int(version.id))
    return DocumentView(
        document_id=int(document.id),
        title=document.title,
        path=path or "",
        format=document.format,
        status=document.status,
        doc_date=document.doc_date,
        language=document.language,
        tags=tags,
        latest_version_no=latest_version_no,
        markdown=markdown,
        sections=sections,
        openable=root_kind != "remote",
    )


def _accepted_tags(session: Session, document_id: int) -> list[str]:
    from mynotes.tags.store import accepted_tags

    return accepted_tags(session, document_id)


def _version_sections(session: Session, version_id: int) -> list[SectionRef]:
    """Group the version's chunks into sections, in first-appearance order."""
    rows = session.execute(
        select(Chunk.section, Chunk.page, Chunk.slide, func.min(Chunk.ordinal))
        .where(Chunk.document_version_id == version_id)
        .group_by(Chunk.section, Chunk.page, Chunk.slide)
        .order_by(func.min(Chunk.ordinal))
    ).all()
    return [
        SectionRef(
            section=section,
            page=page,
            slide=slide,
            first_ordinal=int(first_ordinal),
        )
        for section, page, slide, first_ordinal in rows
    ]


def search(
    query: str,
    filters: SearchFilters | None = None,
    *,
    principal: Principal = LOCAL,
    rerank: bool = True,
) -> RetrievalResult:
    """Run the retrieval pipeline (query analysis, hybrid search, fuse, rerank).

    Args:
        query: The raw user query, possibly with inline filter tokens.
        filters: Base filters from interface options; inline filters win.
        principal: The caller; its prefixes are applied to the search.
        rerank: When False, skip the reranker and return the fused order.

    Returns:
        The :class:`RetrievalResult` for the query.
    """
    return retrieve(query, _apply_acl(filters, principal), rerank=rerank)


def ask(
    question: str, filters: SearchFilters | None = None, *, principal: Principal = LOCAL
) -> AnswerResult:
    """Generate one grounded, cited answer for a question.

    Args:
        question: The raw user question, possibly with inline filter tokens.
        filters: Base filters from interface options; inline filters win.
        principal: The caller; its prefixes are applied to the retrieval.

    Returns:
        The :class:`AnswerResult` for the question.
    """
    return answer(question, _apply_acl(filters, principal))


def ask_stream(
    question: str,
    filters: SearchFilters | None = None,
    *,
    principal: Principal = LOCAL,
    conversation_id: str | None = None,
) -> Iterator[AnswerEvent]:
    """Stream one answer as ``token``/``sources``/``done`` (or ``error``) events.

    With a ``conversation_id`` whose chat already has turns, the question is a
    possible follow-up: the last ``retrieval.followup_turns`` turns rewrite it
    into a standalone question for retrieval (yielded first as a ``rewritten``
    event when it differs) and are given to the answer as earlier messages.

    Args:
        question: The raw user question, possibly with inline filter tokens.
        filters: Base filters from interface options; inline filters win.
        principal: The caller; its prefixes are applied to the retrieval.
        conversation_id: The chat the question belongs to (History grouping).

    Yields:
        The :class:`AnswerEvent` pieces as produced by the answer layer.
    """
    from mynotes.llm.chat import get_chat
    from mynotes.retrieval.followup import Turn, condense_question, history_messages

    turns = _previous_turns(conversation_id, principal)
    if not turns:
        yield from answer_stream(question, _apply_acl(filters, principal))
        return
    standalone = condense_question(question, turns, get_chat("answer"))
    if standalone != question:
        yield AnswerEvent(type="rewritten", data=standalone)
    yield from answer_stream(
        question,
        _apply_acl(filters, principal),
        search_question=standalone,
        history=history_messages([Turn(t.question, t.answer) for t in turns]),
    )


def _previous_turns(conversation_id: str | None, principal: Principal) -> list[Any]:
    """The last ``retrieval.followup_turns`` stored turns of the caller's chat, oldest first."""
    limit = get_settings().retrieval.followup_turns
    if not conversation_id or limit <= 0:
        return []
    from mynotes.core.chat_history import get_conversation

    try:
        conversation = get_conversation(conversation_id, principal=principal)
    except Exception:
        logger.warning("chat history unavailable; answering without earlier turns", exc_info=True)
        return []
    return list(conversation.turns[-limit:]) if conversation is not None else []


def get_document(ref: str | int, *, principal: Principal = LOCAL) -> DocumentView:
    """Resolve a document by id, path or title and return its view.

    Args:
        ref: The document id, the path relative to a source root, or a title.
        principal: The caller; id-based calls check its prefixes.

    Returns:
        The :class:`DocumentView` of the resolved document.

    Raises:
        NotFound: When nothing matches.
        Ambiguous: When several documents match (with candidates).
        Forbidden: When the document is outside the allowed prefixes.
    """
    with session_scope() as session:
        document_id, path = _resolve_document(session, ref)
        _ensure_document_allowed(path, principal)
        return _document_view(session, document_id)


def document_history(ref: str | int, *, principal: Principal = LOCAL) -> list[VersionInfo]:
    """List a document's retained version snapshots, newest first.

    Args:
        ref: The document id, path or title.
        principal: The caller; the resolved document must be allowed.

    Returns:
        One :class:`VersionInfo` per retained version, newest first.

    Raises:
        NotFound: When nothing matches.
        Ambiguous: When several documents match (with candidates).
        Forbidden: When the document is outside the allowed prefixes.
    """
    with session_scope() as session:
        document_id, path = _resolve_document(session, ref)
        _ensure_document_allowed(path, principal)
        return document_version_history(session, document_id)


def document_tags(document_id: int, *, principal: Principal = LOCAL) -> list[TagDetail]:
    """List a document's tag rows with source and review status (task 48).

    Args:
        document_id: The document id.
        principal: The caller; the resolved document must be allowed.

    Returns:
        The rows ordered by source then tag name.

    Raises:
        NotFound: When the document does not exist.
        Forbidden: When the document is outside the allowed prefixes.
    """
    with session_scope() as session:
        _ensure_document_allowed(_document_path(session, document_id), principal)
        return document_tag_details(session, document_id)


def document_chunk_text(
    document_id: int, ordinal: int, *, principal: Principal = LOCAL
) -> str | None:
    """The text of one chunk of a document's latest version (task 48).

    Args:
        document_id: The document id.
        ordinal: The chunk's 0-based ordinal within the latest version.
        principal: The caller; the resolved document must be allowed.

    Returns:
        The chunk text, or None when the document has no latest version or no
        chunk with that ordinal.

    Raises:
        NotFound: When the document does not exist.
        Forbidden: When the document is outside the allowed prefixes.
    """
    with session_scope() as session:
        _ensure_document_allowed(_document_path(session, document_id), principal)
        return session.execute(
            select(Chunk.text)
            .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
            .where(
                DocumentVersion.document_id == document_id,
                DocumentVersion.is_latest.is_(True),
                Chunk.ordinal == ordinal,
            )
        ).scalar_one_or_none()


def version_chain(
    document_id: int, *, principal: Principal = LOCAL
) -> tuple[list[RelatedDocument], list[RelatedDocument]]:
    """The newer and older documents of a document's supersedes chain (task 48).

    Only relations with status ``auto`` or ``confirmed`` count as active.

    Args:
        document_id: The document id.
        principal: The caller; the resolved document must be allowed.

    Returns:
        ``(newer, older)``: documents that supersede this one, and documents
        it supersedes, each ordered by document id.

    Raises:
        NotFound: When the document does not exist.
        Forbidden: When the document is outside the allowed prefixes.
    """
    with session_scope() as session:
        resolved_id, path = _resolve_document(session, document_id)
        _ensure_document_allowed(path, principal)
        active = DocRelation.status.in_(("auto", "confirmed"))

        def _linked(source_column: Any, target_filter: Any) -> list[RelatedDocument]:
            rows = session.execute(
                select(Document, File.rel_path)
                .select_from(DocRelation)
                .join(Document, Document.id == source_column)
                .outerjoin(File, File.id == Document.current_file_id)
                .where(
                    DocRelation.type == "supersedes",
                    active,
                    target_filter,
                )
                .order_by(Document.id)
            ).all()
            return [
                RelatedDocument(
                    document_id=int(document.id),
                    title=document.title,
                    path=rel_path or "",
                    doc_date=document.doc_date,
                    status=document.status,
                )
                for document, rel_path in rows
            ]

        newer = _linked(DocRelation.src_document_id, DocRelation.dst_document_id == resolved_id)
        older = _linked(DocRelation.dst_document_id, DocRelation.src_document_id == resolved_id)
    return newer, older


def document_previews(
    document_ids: Sequence[int], *, principal: Principal = LOCAL
) -> dict[int, str]:
    """The first 300 characters of several documents' latest-version text.

    Args:
        document_ids: The document ids to preview (duplicates ignored).
        principal: The caller; restricted callers only get previews of
            documents inside their allowed prefixes (task 52).

    Returns:
        A mapping from document id to its preview (empty string when the
        document has no chunked version or does not exist).
    """
    unique_ids = list(dict.fromkeys(int(document_id) for document_id in document_ids))
    with session_scope() as session:
        if principal.acl_prefixes is None:
            return {
                document_id: document_preview_text(session, document_id)
                for document_id in unique_ids
            }
        allowed: dict[int, str] = {}
        for document_id in unique_ids:
            try:
                path = _document_path(session, document_id)
            except NotFound:
                continue
            if _path_allowed(path, principal.acl_prefixes):
                allowed[document_id] = document_preview_text(session, document_id)
        return allowed


def _document_path(session: Session, document_id: int) -> str:
    """A document's current file path (empty when it has no file row anymore)."""
    rel_path = session.execute(
        select(File.rel_path)
        .join(Document, Document.current_file_id == File.id)
        .where(Document.id == document_id)
    ).scalar_one_or_none()
    if rel_path is not None:
        return rel_path
    document = session.get(Document, document_id)
    if document is None:
        raise NotFound(f"no document with id {document_id}")
    if document.latest_version_id is None:
        return ""
    return (
        session.execute(
            select(DocumentVersion.rel_path_at_version).where(
                DocumentVersion.id == document.latest_version_id
            )
        ).scalar_one_or_none()
        or ""
    )


@dataclass
class ResolvedFile:
    """A document's original file prepared for serving or opening (task 45).

    Attributes:
        document_id: The document id.
        path: The resolved absolute path on disk.
        filename: The stored file name (for ``Content-Disposition``).
        media_type: The HTTP ``Content-Type`` (with charset for text).
        inline: Whether the file may be served inline (PDF/text/HTML).
    """

    document_id: int
    path: Path
    filename: str
    media_type: str
    inline: bool


_FILE_MEDIA_TYPES: dict[str, tuple[str, bool]] = {
    "pdf": ("application/pdf", True),
    "html": ("text/html; charset=utf-8", True),
    "md": ("text/plain; charset=utf-8", True),
    "txt": ("text/plain; charset=utf-8", True),
    "docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        False,
    ),
    "pptx": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        False,
    ),
    "xlsx": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        False,
    ),
    "epub": ("application/epub+zip", False),
}
"""Serving media type and inline disposition per file format (task 45)."""


def resolve_file(document_id: int, *, principal: Principal = LOCAL) -> ResolvedFile:
    """Resolve a document's current file for serving or opening (task 45).

    The file is addressed by document id only; the path comes from the
    database and is verified to stay inside its source root (resolved real
    path, so ``..`` and symlink escapes fail as ``NotFound``).

    Args:
        document_id: The document id.
        principal: The caller; restricted callers may only access files under
            their allowed prefixes.

    Returns:
        The :class:`ResolvedFile` with the on-disk path, media type and
        disposition.

    Raises:
        NotFound: When the document has no active file, the stored path
            escapes its source root, or the file no longer exists.
        Forbidden: When the file is outside the caller's allowed prefixes.
    """
    with session_scope() as session:
        row = session.execute(
            select(
                File.rel_path,
                File.format,
                File.content_hash,
                SourceRoot.kind,
                SourceRoot.abs_path,
                SourceRoot.workspace_id,
                SourceRoot.id,
            )
            .join(Document, Document.current_file_id == File.id)
            .join(SourceRoot, SourceRoot.id == File.source_root_id)
            .where(Document.id == document_id, File.status == "active")
        ).first()
    if row is None:
        raise NotFound(f"no accessible file for document {document_id}")
    rel_path, file_format, content_hash, root_kind, root_abs, workspace_id, root_id = row
    _ensure_document_allowed(rel_path, principal)
    if root_kind == "remote":
        # The agent uploaded the bytes: serve the server's copy from the blob store.
        from mynotes.storage.blobs import StorageNotConfigured, get_blob_store

        if content_hash is None:
            raise NotFound(f"no accessible file for document {document_id}")
        try:
            store = get_blob_store()
        except StorageNotConfigured as exc:
            raise NotFound(str(exc)) from exc
        root_path = store.dir
        candidate = store.path_for(int(workspace_id), int(root_id), content_hash, file_format)
    else:
        if root_abs is None:
            raise NotFound(f"no accessible file for document {document_id}")
        root_path = Path(root_abs)
        candidate = from_rel_path(root_path, rel_path)
    if not is_within(root_path, candidate) or not long_path(candidate).exists():
        raise NotFound(f"no accessible file for document {document_id}")
    media_type, inline = _FILE_MEDIA_TYPES.get(file_format, ("application/octet-stream", False))
    path = candidate.resolve()
    return ResolvedFile(
        document_id=document_id,
        path=path,
        filename=Path(rel_path).name,
        media_type=media_type,
        inline=inline,
    )


def open_original(document_id: int, *, principal: Principal = LOCAL) -> None:
    """Open a document's original file in the OS default application (task 45).

    Local mode only: the file is opened on the machine the code runs on, so
    team principals (``can_open_files=False``) are denied before anything is
    resolved.

    Args:
        document_id: The document id.
        principal: The caller; must have ``can_open_files``.

    Raises:
        Forbidden: When ``principal.can_open_files`` is false (team mode), or
            the document belongs to a remote root (its original is on the
            owner's machine, not here).
        NotFound: When the document has no active, in-root file.
    """
    if not principal.can_open_files:
        raise Forbidden("opening files in apps is only allowed in local mode")
    with session_scope() as session:
        kind = session.execute(
            select(SourceRoot.kind)
            .join(Document, Document.source_root_id == SourceRoot.id)
            .where(Document.id == document_id)
        ).scalar_one_or_none()
    if kind == "remote":
        raise Forbidden("this note was uploaded by an agent; open it on the owner's machine")
    open_in_default_app(resolve_file(document_id, principal=principal).path)


def list_tags(prefix: str | None = None, *, principal: Principal = LOCAL) -> list[TagView]:
    """List the tag vocabulary with accepted-usage counts.

    Args:
        prefix: Only tags whose name starts with this prefix.
        principal: The caller; restricted callers only count documents inside
            their allowed prefixes (task 52).

    Returns:
        The summaries ordered by tag name.
    """
    with session_scope() as session:
        workspace_id = _workspace_id(session)
        if workspace_id is None:
            return []
        if principal.acl_prefixes is None:
            return list(tag_summaries(session, workspace_id, prefix=prefix))
        return _visible_tag_summaries(session, workspace_id, principal.acl_prefixes, prefix)


def _visible_tag_summaries(
    session: Session,
    workspace_id: int,
    acl_prefixes: Sequence[str],
    prefix: str | None,
) -> list[TagView]:
    """The tag vocabulary with counts and sources over visible documents only."""
    counts: dict[str, set[int]] = {}
    sources: dict[str, set[str]] = {}
    for name, document_id, source, rel_path in session.execute(
        select(Tag.name, DocumentTag.document_id, DocumentTag.source, File.rel_path)
        .select_from(DocumentTag)
        .join(Tag, Tag.id == DocumentTag.tag_id)
        .join(Document, Document.id == DocumentTag.document_id)
        .outerjoin(File, File.id == Document.current_file_id)
        .where(Tag.workspace_id == workspace_id, DocumentTag.status == "accepted")
        .order_by(Tag.name)
    ).all():
        if rel_path and _path_allowed(str(rel_path), acl_prefixes):
            counts.setdefault(str(name), set()).add(int(document_id))
            sources.setdefault(str(name), set()).add(str(source))
    return [
        TagSummary(
            name=summary.name,
            kind=summary.kind,
            document_count=len(counts.get(summary.name, set())),
            sources=sorted(sources.get(summary.name, set())),
        )
        for summary in tag_summaries(session, workspace_id, prefix=prefix)
    ]


def _workspace_id(session: Session) -> int | None:
    from mynotes.tags.store import workspace_id_by_name

    return workspace_id_by_name(session, get_settings().workspace)


def tag_action(
    document_ref: str | int,
    tag: str,
    action: Literal["accept", "reject"],
    *,
    principal: Principal = LOCAL,
) -> None:
    """Accept or reject one proposed LLM tag of a document.

    Args:
        document_ref: The document id, path or title.
        tag: The proposed tag's name (as shown in the review queues).
        action: ``accept`` or ``reject``.
        principal: The caller; the resolved document must be allowed.

    Raises:
        NotFound: When the document, the tag or a pending proposal is missing.
        Ambiguous: When several documents match (with candidates).
        Forbidden: When the document is outside the allowed prefixes.
    """
    decision = "accepted" if action == "accept" else "rejected"
    with session_scope() as session:
        document_id, path = _resolve_document(session, document_ref)
        _ensure_document_allowed(path, principal)
        workspace_id = _workspace_id(session)
        if workspace_id is None:
            raise NotFound("no workspace; run 'mynotes init' first")
        tag_id = tag_id_by_name(session, workspace_id, tag)
        if tag_id is None:
            raise NotFound(f"tag {tag!r} is not in the vocabulary")
        if not decide_proposal(session, document_id, tag_id, decision):
            raise NotFound(f"no proposed LLM tag {tag!r} for document {document_id}")
        _enqueue_reindex(session, document_id)


def _enqueue_reindex(session: Session, document_id: int) -> None:
    from mynotes.jobs.queue import enqueue
    from mynotes.jobs.registry import Priority

    enqueue(
        session,
        "reindex_metadata",
        {"document_id": document_id},
        dedupe_key=f"reindex_metadata:{document_id}",
        priority=Priority.EMBED,
    )


def merge_tags(from_name: str, into_name: str, *, principal: Principal = LOCAL) -> int:
    """Merge one tag into another and queue a reindex of the affected documents.

    Args:
        from_name: The tag to merge away.
        into_name: The surviving tag (created when new).
        principal: The caller (the vocabulary is workspace-wide).

    Returns:
        The number of ``document_tag`` rows moved.

    Raises:
        NotFound: When the workspace does not exist.
        Forbidden: When the caller is not an admin (task 52).
    """
    _require_admin(principal)
    with session_scope() as session:
        workspace_id = _workspace_id(session)
        if workspace_id is None:
            raise NotFound("no workspace; run 'mynotes init' first")
        affected = document_ids_with_tag(session, workspace_id, from_name)
        moved = merge_tag_names(session, workspace_id, from_name, into_name)
        for document_id in affected:
            _enqueue_reindex(session, document_id)
    return moved


def list_version_relations(
    status: str | None = None, *, principal: Principal = LOCAL
) -> list[RelationView]:
    """List supersedes relations with both documents resolved.

    Args:
        status: Only relations with this status (``proposed``, ``auto``,
            ``confirmed``, ``rejected``); None means all.
        principal: The caller; restricted callers only see relations whose
            both endpoint documents are inside their allowed prefixes (task 52).

    Returns:
        The relations ordered by id.
    """
    with session_scope() as session:
        relations = list_relations(session, [status] if status else None)
    if principal.acl_prefixes is None:
        return relations
    return [
        relation
        for relation in relations
        if _path_allowed(relation.newer.path, principal.acl_prefixes)
        and _path_allowed(relation.older.path, principal.acl_prefixes)
    ]


def version_action(
    relation_id: int,
    action: Literal["confirm", "reject", "swap"],
    *,
    principal: Principal = LOCAL,
) -> None:
    """Confirm, reject or swap one supersedes relation (admin only, task 52).

    Args:
        relation_id: The relation to act on.
        action: ``confirm`` (apply the chain effects), ``reject`` (undo them)
            or ``swap`` (reverse the direction and confirm).
        principal: The caller; must be an admin in team mode.

    Raises:
        NotFound: When the relation does not exist (or the action is invalid).
        Forbidden: When the caller is not an admin.
    """
    _require_admin(principal)
    function = {"confirm": confirm, "reject": reject, "swap": swap}.get(action)
    if function is None:
        raise NotFound(f"unknown version action {action!r}")
    try:
        with session_scope() as session:
            function(session, relation_id)
    except ValueError as exc:
        raise NotFound(str(exc)) from exc


def filter_graph_by_acl(
    session: Session, sub: Subgraph, acl_prefixes: Sequence[str] | None
) -> Subgraph:
    """Restrict a subgraph to entities and relations with allowed evidence.

    Only entities with at least one mention or relation-evidence chunk inside
    the allowed prefixes and only edges with such an evidence chunk survive
    (PLAN.md §11 "Graph ACL"). ``acl_prefixes=None`` (local mode) returns the
    subgraph unchanged; an empty list returns an empty subgraph. The shared
    implementation lives in :mod:`mynotes.core.acl` so the retrieval expanders
    apply exactly the same rule.

    Args:
        session: An open session/transaction.
        sub: The subgraph to filter.
        acl_prefixes: The allowed path prefixes, None for unrestricted.

    Returns:
        The filtered :class:`Subgraph`.
    """
    return _filter_subgraph_by_acl(session, sub, acl_prefixes)


def entity_id_ref(ref: str) -> int | None:
    """The id of a ``#<id>`` entity reference, or None for a name."""
    cleaned = ref.strip()
    if cleaned.startswith("#") and cleaned[1:].isdigit():
        return int(cleaned[1:])
    return None


def _entity_candidates(entities: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {"id": int(entity.id), "name": entity.canonical_name, "type": entity.type}
        for entity in entities
    ]


def _resolve_entity(session: Session, ref: str) -> int:
    """Resolve an entity reference (a name, or ``#<id>``) to one id.

    ``#<id>`` names an entity by its id and is never ambiguous: the web UI's
    candidate links use it, since two candidates can share a name.

    Raises:
        NotFound: When nothing matches.
        Ambiguous: When several entities match (with candidates).
    """
    explicit_id = entity_id_ref(ref)
    if explicit_id is not None:
        if session.get(KgEntity, explicit_id) is None:
            raise NotFound(f"no entity matches {ref!r}")
        return explicit_id
    matches = find_entities(session, ref)
    if len(matches) == 1:
        return int(matches[0].id)
    if not matches:
        cleaned = ref.strip()
        if cleaned.isdigit() and session.get(KgEntity, int(cleaned)) is not None:
            return int(cleaned)
        raise NotFound(f"no entity matches {ref!r}")
    raise Ambiguous(f"{len(matches)} entities match {ref!r}", _entity_candidates(matches))


def entity_card(ref: str, *, principal: Principal = LOCAL) -> dict[str, Any]:
    """The entity card for a name reference, restricted to allowed evidence.

    Args:
        ref: The entity name (exact, normalized or fuzzy).
        principal: The caller; restricted callers only see allowed documents,
            counts and (when nothing allowed supports it) no shared summary.

    Returns:
        The card dict (``id``, ``name``, ``type``, ``status``, ``aliases``,
        ``relations_out``/``relations_in``, ``mention_count_current``,
        ``top_documents``, ``description``).

    Raises:
        NotFound: When nothing matches.
        Ambiguous: When several entities match (with candidates).
    """
    with session_scope() as session:
        entity_id = _resolve_entity(session, ref)
        card = build_entity_card(session, entity_id)
        if principal.acl_prefixes is not None:
            return _restricted_card(session, card, entity_id, principal.acl_prefixes)
        return card


def _restricted_card(
    session: Session, card: dict[str, Any], entity_id: int, acl_prefixes: Sequence[str]
) -> dict[str, Any]:
    """Recompute the card's document-dependent fields over allowed evidence only.

    The shared summary is LLM-built from *all* evidence, so it stays hidden
    whenever the entity has any mention or relation evidence outside the
    allowed prefixes (PLAN.md §11, task 52); relation and document lists are
    recomputed from allowed evidence only.
    """
    mention_rows = session.execute(
        select(File.rel_path, func.count())
        .select_from(KgMention)
        .join(Chunk, Chunk.id == KgMention.chunk_id)
        .join(Document, Document.id == Chunk.document_id)
        .outerjoin(File, File.id == Document.current_file_id)
        .where(KgMention.entity_id == entity_id)
        .group_by(File.rel_path)
    ).all()
    allowed = [
        (rel_path, count)
        for rel_path, count in mention_rows
        if _path_allowed(rel_path, acl_prefixes)
    ]
    allowed_mentions = int(sum(count for _path, count in allowed))
    top_documents = [
        {
            "document_id": int(document_id),
            "title": title,
            "rel_path": rel_path,
            "mentions": int(count),
        }
        for document_id, title, rel_path, count in session.execute(
            select(Document.id, Document.title, File.rel_path, func.count())
            .select_from(KgMention)
            .join(Chunk, Chunk.id == KgMention.chunk_id)
            .join(Document, Document.id == Chunk.document_id)
            .join(File, File.id == Document.current_file_id)
            .where(KgMention.entity_id == entity_id)
            .group_by(Document.id, Document.title, File.rel_path)
            .order_by(func.count().desc(), Document.id)
            .limit(10)
        ).all()
        if _path_allowed(rel_path, acl_prefixes)
    ]
    card["top_documents"] = top_documents
    card["mention_count_current"] = allowed_mentions
    if _entity_has_restricted_evidence(session, entity_id, acl_prefixes, mention_rows):
        card["description"] = ""
    card["relations_out"] = _allowed_relation_count(session, entity_id, "src", acl_prefixes)
    card["relations_in"] = _allowed_relation_count(session, entity_id, "dst", acl_prefixes)
    return card


def _entity_has_restricted_evidence(
    session: Session,
    entity_id: int,
    acl_prefixes: Sequence[str],
    mention_rows: Sequence[tuple[str | None, int]],
) -> bool:
    """Whether the entity has any mention or active-relation evidence outside the prefixes."""
    if any(not _path_allowed(rel_path or "", acl_prefixes) for rel_path, _count in mention_rows):
        return True
    endpoint = (KgRelation.src_entity_id == entity_id) | (KgRelation.dst_entity_id == entity_id)
    relation_ids = session.execute(
        select(KgRelation.id).where(endpoint, KgRelation.status == "active")
    ).scalars()
    return any(
        not _relation_has_allowed_evidence(session, int(relation_id), acl_prefixes)
        for relation_id in relation_ids
    )


def _allowed_relation_count(
    session: Session, entity_id: int, endpoint: Literal["src", "dst"], acl_prefixes: Sequence[str]
) -> int:
    """Active relations on one entity endpoint with allowed evidence."""
    endpoint_column = KgRelation.src_entity_id if endpoint == "src" else KgRelation.dst_entity_id
    relation_ids = session.execute(
        select(KgRelation.id).where(endpoint_column == entity_id, KgRelation.status == "active")
    ).scalars()
    return sum(
        1
        for relation_id in relation_ids
        if _relation_has_allowed_evidence(session, int(relation_id), acl_prefixes)
    )


def entity_neighbors(ref: str, hops: int = 1, *, principal: Principal = LOCAL) -> dict[str, Any]:
    """The scored neighbours of one entity, restricted to allowed evidence.

    Args:
        ref: The entity name (exact, normalized or fuzzy).
        hops: Maximum relation steps (>= 1).
        principal: The caller; restricted callers see only the allowed
            subgraph.

    Returns:
        A dict with the ``entity`` (``id``, ``name``, ``type``), the requested
        ``hops``, the ``neighbors`` list (``id``, ``name``, ``type``, ``score``,
        ``types``), at most 20 entries, and the ``edges`` list (``src``, ``dst``,
        ``relation_id``, ``type``, ``attribute``, ``value``, ``status``,
        ``evidence`` documents and ``evidence_count``) among the returned
        nodes only, for the graph explorer (task 48): an edge to an entity
        that is not listed can't be drawn.

    Raises:
        NotFound: When nothing matches.
        Ambiguous: When several entities match (with candidates).
    """
    with session_scope() as session:
        entity_id = _resolve_entity(session, ref)
        sub = fetch_subgraph(session, [entity_id], max(1, hops))
        if principal.acl_prefixes is not None:
            sub = filter_graph_by_acl(session, sub, principal.acl_prefixes)
        scores = score_entities(sub, {entity_id: 1.0}, method="one_hop")
        relation_types = _subgraph_relation_types(session, sub)
        entity_types = _entity_types(session, set(sub.nodes))
        neighbors = [
            {
                "id": neighbour_id,
                "name": sub.nodes.get(neighbour_id, f"entity {neighbour_id}"),
                "type": entity_types.get(neighbour_id, ""),
                "score": round(score, 4),
                "types": sorted(relation_types.get(neighbour_id, [])),
            }
            for neighbour_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            if neighbour_id != entity_id
        ][:20]
        shown = {entity_id, *(neighbour["id"] for neighbour in neighbors)}
        return {
            "entity": {
                "id": entity_id,
                "name": sub.nodes.get(entity_id, ref),
                "type": entity_types.get(entity_id, ""),
            },
            "hops": max(1, hops),
            "neighbors": neighbors,
            "edges": _subgraph_edge_views(
                session, edges_among(sub.edges, shown), principal.acl_prefixes
            ),
        }


def _entity_types(session: Session, entity_ids: set[int]) -> dict[int, str]:
    """Entity types for the given ids."""
    if not entity_ids:
        return {}
    return dict(
        session.execute(select(KgEntity.id, KgEntity.type).where(KgEntity.id.in_(entity_ids))).all()
    )


_MAX_EDGE_EVIDENCE = 5
"""Evidence documents listed per edge in the graph explorer."""


def edges_among(
    edges: Sequence[tuple[int, int, int, float]], node_ids: set[int]
) -> list[tuple[int, int, int, float]]:
    """The subgraph edges whose both endpoints are in ``node_ids``, in order."""
    return [edge for edge in edges if edge[0] in node_ids and edge[1] in node_ids]


def _subgraph_edge_views(
    session: Session,
    edges: Sequence[tuple[int, int, int, float]],
    acl_prefixes: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """Subgraph edges (``(src, dst, relation_id, weight)``) with relation metadata and evidence."""
    if not edges:
        return []
    relations = {
        int(relation.id): relation
        for relation in session.execute(
            select(KgRelation).where(KgRelation.id.in_([edge[2] for edge in edges]))
        ).scalars()
    }
    evidence: dict[int, list[dict[str, Any]]] = {}
    seen_documents: dict[int, set[int]] = {}
    for relation_id, document_id, title, rel_path in session.execute(
        select(
            KgRelationEvidence.relation_id,
            Document.id,
            Document.title,
            File.rel_path,
        )
        .select_from(KgRelationEvidence)
        .join(Chunk, Chunk.id == KgRelationEvidence.chunk_id)
        .join(Document, Document.id == Chunk.document_id)
        .outerjoin(File, File.id == Document.current_file_id)
        .where(KgRelationEvidence.relation_id.in_([edge[2] for edge in edges]))
        .order_by(KgRelationEvidence.id)
    ).all():
        if acl_prefixes is not None and not (rel_path and _path_allowed(rel_path, acl_prefixes)):
            continue
        known = seen_documents.setdefault(int(relation_id), set())
        if document_id in known:
            continue
        known.add(document_id)
        evidence.setdefault(int(relation_id), []).append(
            {"document_id": int(document_id), "title": title, "path": rel_path or ""}
        )
    views: list[dict[str, Any]] = []
    for src, dst, relation_id, _weight in edges:
        relation = relations.get(relation_id)
        if relation is None:
            continue
        documents = evidence.get(relation_id, [])
        views.append(
            {
                "src": int(src),
                "dst": int(dst),
                "relation_id": int(relation_id),
                "type": relation.type,
                "attribute": relation.attribute,
                "value": relation.value,
                "status": relation.status,
                "evidence": documents[:_MAX_EDGE_EVIDENCE],
                "evidence_count": len(documents),
            }
        )
    return views


def entity_attributes(entity_id: int, *, principal: Principal = LOCAL) -> list[AttributeValue]:
    """The current attribute values of one entity, read-only (task 48).

    Unlike :func:`mynotes.graph.temporal.current_attribute_values` this never
    demotes superseded relations; it only reports what that function would
    pick as the current value per attribute.

    Args:
        entity_id: The source entity's id.
        principal: The caller; restricted callers only see values backed by
            allowed evidence.

    Returns:
        One :class:`AttributeValue` per attribute that has a current value,
        ordered by attribute name.

    Raises:
        NotFound: When the entity does not exist.
    """
    with session_scope() as session:
        if session.get(KgEntity, entity_id) is None:
            raise NotFound(f"no kg entity with id {entity_id}")
        acl_prefixes = principal.acl_prefixes
        facts_by_attribute: dict[str, list[AttributeFact]] = {}
        relations = session.execute(
            select(KgRelation)
            .where(
                KgRelation.src_entity_id == entity_id,
                KgRelation.attribute.is_not(None),
                KgRelation.status.in_(("active", "historical", "stale")),
            )
            .order_by(KgRelation.id)
        ).scalars()
        for relation in relations:
            if acl_prefixes is not None and not _relation_has_allowed_evidence(
                session, int(relation.id), acl_prefixes
            ):
                continue
            doc_date, confidence, _evidence_date = _attribute_evidence(session, int(relation.id))
            facts_by_attribute.setdefault(str(relation.attribute), []).append(
                AttributeFact(
                    relation_id=int(relation.id),
                    value=str(relation.value or ""),
                    doc_date=doc_date,
                    confidence=confidence,
                )
            )
        values: list[AttributeValue] = []
        for attribute in sorted(facts_by_attribute):
            current = pick_current_attribute(facts_by_attribute[attribute])
            if current is None:
                continue
            values.append(
                AttributeValue(
                    attribute=attribute,
                    value=current.value,
                    doc_date=current.doc_date,
                    relation_id=current.relation_id,
                )
            )
        return values


_MERGE_CANDIDATE_WINDOW_DAYS = 7
_MERGE_CANDIDATE_SIMILARITY = 0.5


def entity_merge_candidates(limit: int = 50, *, principal: Principal = LOCAL) -> list[EntityPair]:
    """Recent entity pairs that look like duplicates (task 48 review queue).

    Pairs of active same-type entities whose normalized names have trigram
    similarity >= 0.5, with no recorded ``kg_entity_distinct`` decision, where
    at least one side was created in the last 7 days.

    Args:
        limit: Maximum number of pairs.
        principal: The caller; restricted callers only see pairs whose both
            entities have allowed evidence (task 52).

    Returns:
        The pairs, newest creation date first, then highest similarity.
    """
    cutoff = datetime.now(UTC) - timedelta(days=_MERGE_CANDIDATE_WINDOW_DAYS)
    low = aliased(KgEntity)
    high = aliased(KgEntity)
    not_distinct = ~exists(
        select(KgEntityDistinct.a_id).where(
            KgEntityDistinct.a_id == low.id, KgEntityDistinct.b_id == high.id
        )
    )
    with session_scope() as session:
        rows = session.execute(
            select(low, high, func.similarity(low.norm_name, high.norm_name))
            .where(
                low.workspace_id == high.workspace_id,
                low.id < high.id,
                low.type == high.type,
                low.status == "active",
                high.status == "active",
                func.similarity(low.norm_name, high.norm_name) >= _MERGE_CANDIDATE_SIMILARITY,
                or_(low.created_at >= cutoff, high.created_at >= cutoff),
                not_distinct,
            )
            .order_by(func.greatest(low.created_at, high.created_at).desc(), low.id, high.id)
            .limit(limit)
        ).all()
        pairs = [
            EntityPair(
                a_id=int(low.id),
                b_id=int(high.id),
                a_name=low.canonical_name,
                b_name=high.canonical_name,
                a_type=low.type,
                b_type=high.type,
                a_created_at=low.created_at,
                b_created_at=high.created_at,
                similarity=float(similarity),
            )
            for low, high, similarity in rows
        ]
        if principal.acl_prefixes is None:
            return pairs
        allowed = _allowed_entity_ids(
            session,
            [pair_id for pair in pairs for pair_id in (pair.a_id, pair.b_id)],
            principal.acl_prefixes,
        )
        return [pair for pair in pairs if pair.a_id in allowed and pair.b_id in allowed]


def mark_entities_distinct(a_id: int, b_id: int, *, principal: Principal = LOCAL) -> None:
    """Record that two entities are different things (task 48 review queue).

    The pair is written to ``kg_entity_distinct`` in canonical ``(a < b)``
    order; entity resolution never merges or adjudicates it again.

    Args:
        a_id: One entity id.
        b_id: The other entity id.
        principal: The caller; must be an admin in team mode (task 52).

    Raises:
        NotFound: When either entity does not exist.
        Forbidden: When the caller is not an admin.
    """
    _require_admin(principal)
    if a_id == b_id:
        raise NotFound("the two entity ids are the same")
    with session_scope() as session:
        found = session.execute(
            select(func.count()).select_from(KgEntity).where(KgEntity.id.in_([a_id, b_id]))
        ).scalar_one()
        if int(found) != 2:
            raise NotFound(f"entity pair ({a_id}, {b_id}) not found")
        session.execute(
            pg_insert(KgEntityDistinct)
            .values(a_id=min(a_id, b_id), b_id=max(a_id, b_id))
            .on_conflict_do_nothing(index_elements=["a_id", "b_id"])
        )


def merge_entities(keep_id: int, drop_id: int, *, principal: Principal = LOCAL) -> None:
    """Merge one entity into another (reversible via the store's unmerge).

    Args:
        keep_id: The surviving entity (conventionally the older one).
        drop_id: The entity folded into ``keep_id``.
        principal: The caller; must be an admin in team mode (task 52).

    Raises:
        NotFound: When either entity does not exist.
        Forbidden: When the caller is not an admin.
    """
    _require_admin(principal)
    with session_scope() as session:
        if session.get(KgEntity, keep_id) is None or session.get(KgEntity, drop_id) is None:
            raise NotFound(f"entity pair ({keep_id}, {drop_id}) not found")
        merge_entity_rows(session, keep_id, drop_id)


def tag_proposals(*, principal: Principal = LOCAL) -> list[DocumentProposals]:
    """Documents with pending LLM tag proposals (task 48 review queue).

    Args:
        principal: The caller; restricted callers only see documents inside
            their allowed prefixes (task 52).

    Returns:
        One entry per document with ``source='llm'`` rows still ``proposed``,
        oldest proposal first.
    """
    with session_scope() as session:
        proposals = pending_proposals(session)
    if principal.acl_prefixes is None:
        return proposals
    return [entry for entry in proposals if _path_allowed(entry.rel_path, principal.acl_prefixes)]


def _subgraph_relation_types(session: Session, sub: Subgraph) -> dict[int, set[str]]:
    """Relation types of the subgraph's edges, keyed by every endpoint."""
    if not sub.edges:
        return {}
    relations = {
        int(relation.id): relation
        for relation in session.execute(
            select(KgRelation).where(KgRelation.id.in_([edge[2] for edge in sub.edges]))
        ).scalars()
    }
    types: dict[int, set[str]] = {}
    for src, dst, relation_id, _weight in sub.edges:
        relation = relations.get(relation_id)
        if relation is None:
            continue
        types.setdefault(src, set()).add(relation.type)
        types.setdefault(dst, set()).add(relation.type)
    return types


def timeline(ref: str, *, principal: Principal = LOCAL) -> list[TimelineItem]:
    """The temporal timeline of one entity or ``project:NAME`` tag.

    Args:
        ref: An entity name, or ``project:NAME`` for the project tag's
            documents.
        principal: The caller; items outside the allowed prefixes are dropped.

    Returns:
        The timeline items sorted by date (undated last).

    Raises:
        NotFound: When nothing matches.
        Ambiguous: When several entities match (with candidates).
    """
    with session_scope() as session:
        workspace_id = _workspace_id(session)
        cleaned = ref.strip()
        if cleaned.lower().startswith("project:"):
            name = cleaned[len("project:") :].strip()
            entity_id = (
                project_entity_id(session, workspace_id, name)
                if workspace_id is not None and name
                else None
            )
            items = (
                entity_timeline(session, entity_id)
                if entity_id is not None
                else tag_timeline(session, workspace_id or 0, cleaned)
            )
        else:
            entity_id = _resolve_entity(session, cleaned)
            items = entity_timeline(session, entity_id)
        if principal.acl_prefixes is None:
            return items
        return [
            item
            for item in items
            if item.document_path and _path_allowed(item.document_path, principal.acl_prefixes)
        ]


def trigger_scan(
    root: str | None = None, full: bool = False, *, principal: Principal = LOCAL
) -> list[ScanSummary]:
    """Scan the enabled source roots (or one named enabled root) and persist changes.

    Args:
        root: Only this root name; None scans every enabled root.
        full: Re-hash every file (full verification) instead of an
            incremental scan.
        principal: The caller; must be an admin (the LOCAL principal is).

    Returns:
        One :class:`ScanSummary` per scanned root; a failing root reports its
        error in ``errors`` without stopping the others.

    Raises:
        Forbidden: When the caller is not an admin.
        NotFound: When no root matches.
    """
    _require_admin(principal)
    settings = get_settings()
    statement = (
        select(SourceRoot)
        .join(Workspace, Workspace.id == SourceRoot.workspace_id)
        .where(Workspace.name == settings.workspace)
    )
    statement = statement.where(SourceRoot.enabled.is_(True), SourceRoot.kind == "local")
    if root is not None:
        statement = statement.where(SourceRoot.name == root)
    with session_scope() as session:
        roots = [
            (int(source_root.id), source_root.name)
            for source_root in session.execute(statement).scalars()
        ]
    if not roots:
        raise NotFound(
            f"no enabled local source root named {root!r}; remote roots are fed by their "
            "agent; check mynotes.yaml and run 'mynotes init'"
            if root is not None
            else "no enabled local source roots; run 'mynotes init' first"
        )
    kind: Literal["incremental", "full"] = "full" if full else "incremental"
    summaries: list[ScanSummary] = []
    for source_root_id, source_root_name in roots:
        try:
            summaries.append(scan_root(source_root_id, kind=kind, trigger="manual"))
        except Exception as exc:
            logger.exception("scan of root %r failed", source_root_name)
            summaries.append(
                ScanSummary(
                    scan_run_id=0,
                    counts={"failed": 1},
                    errors=[f"{type(exc).__name__}: {exc}"],
                )
            )
    return summaries


def status(*, principal: Principal = LOCAL) -> dict[str, Any]:
    """The current state of config, database, indices and endpoints.

    Args:
        principal: The caller; admins only (the LOCAL principal is one).

    Returns:
        The JSON-serializable status dict (rendered by ``mynotes status`` and
        the web status dashboard). Beyond :func:`collect_status`'s sections it
        adds the watcher states (``watchers``), the last scan per root
        (``last_scans``), queued jobs by type (``queue_by_type``), the KG
        backlog with an ETA (``kg_backlog``) and the last drift report
        (``drift``), so the dashboard needs no extra queries (task 48).

    Raises:
        Forbidden: When the caller is not an admin.
    """
    if not principal.is_admin:
        raise Forbidden("the status endpoint is admin-only")
    data = collect_status()
    data.update(_status_extras())
    return data


def _status_extras() -> dict[str, Any]:
    """The dashboard-only status sections; each part fails soft."""
    extras: dict[str, Any] = {}
    try:
        from mynotes.scan.watch import watcher_status_rows

        extras["watchers"] = watcher_status_rows()
    except Exception:
        logger.warning("status: watcher state unavailable", exc_info=True)
        extras["watchers"] = []
    try:
        extras["watchers"].extend(_agent_rows())
    except Exception:
        logger.warning("status: agent state unavailable", exc_info=True)
    try:
        extras["last_scans"] = _last_scan_rows()
    except Exception:
        logger.warning("status: last scans unavailable", exc_info=True)
        extras["last_scans"] = []
    try:
        extras["queue_by_type"] = _queue_by_type()
    except Exception:
        logger.warning("status: queue by type unavailable", exc_info=True)
        extras["queue_by_type"] = {}
    try:
        extras["kg_backlog"] = _kg_backlog()
    except Exception:
        logger.warning("status: kg backlog unavailable", exc_info=True)
        extras["kg_backlog"] = {"error": "unavailable"}
    try:
        extras["drift"] = _drift_summary()
    except Exception:
        logger.warning("status: drift report unavailable", exc_info=True)
        extras["drift"] = None
    return extras


AGENT_STALE_AFTER_S = 120
"""An agent that has not sent a heartbeat for this long is shown as stale."""


def _agent_rows() -> list[dict[str, Any]]:
    """Remote roots as watcher-like rows: the agent's last heartbeat decides the state."""
    now = datetime.now(UTC)
    with session_scope() as session:
        rows = session.execute(
            select(SourceRoot.name, SourceRoot.agent_last_seen_at, SourceRoot.agent_info)
            .join(Workspace, Workspace.id == SourceRoot.workspace_id)
            .where(
                Workspace.name == get_settings().workspace,
                SourceRoot.kind == "remote",
                SourceRoot.enabled.is_(True),
            )
            .order_by(SourceRoot.name)
        ).all()
    result: list[dict[str, Any]] = []
    for name, seen_at, info in rows:
        stale = seen_at is None or (now - seen_at).total_seconds() > AGENT_STALE_AFTER_S
        host = (info or {}).get("hostname") or ""
        result.append(
            {
                "root": name,
                "state": "agent" if not stale else "down",
                "backend": f"agent on {host}" if host else "agent",
                "since": seen_at.isoformat() if seen_at else "(never)",
                "stale": stale,
            }
        )
    return result


def _last_scan_rows() -> list[dict[str, Any]]:
    """The most recent scan run per source root, newest first."""
    latest = (
        select(ScanRun.source_root_id, func.max(ScanRun.id).label("run_id"))
        .group_by(ScanRun.source_root_id)
        .subquery()
    )
    with session_scope() as session:
        rows = session.execute(
            select(
                SourceRoot.name,
                ScanRun.kind,
                ScanRun.status,
                ScanRun.started_at,
                ScanRun.finished_at,
                ScanRun.counts,
                ScanRun.error,
            )
            .join(latest, latest.c.run_id == ScanRun.id)
            .join(SourceRoot, SourceRoot.id == ScanRun.source_root_id)
            .order_by(ScanRun.started_at.desc())
        ).all()
    return [
        {
            "root": name,
            "kind": kind,
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "counts": dict(counts or {}),
            "error": error,
        }
        for name, kind, status, started_at, finished_at, counts, error in rows
    ]


def _queue_by_type() -> dict[str, int]:
    """Queued job counts by job type."""
    with session_scope() as session:
        rows = session.execute(
            select(Job.type, func.count()).where(Job.status == "queued").group_by(Job.type)
        ).all()
    return {str(job_type): int(count) for job_type, count in rows}


def _kg_backlog() -> dict[str, Any]:
    """Queued KG extraction jobs, the pending-group estimate and an ETA."""
    from mynotes.graph.extract import groups_overview
    from mynotes.llm.limits import all_snapshots

    with session_scope() as session:
        queued = int(
            session.scalar(
                select(func.count())
                .select_from(Job)
                .where(Job.type == "kg_extract", Job.status == "queued")
            )
            or 0
        )
        groups = groups_overview(session)
    p50_ms = next((snapshot.p50_ms for snapshot in all_snapshots() if snapshot.name == "llm"), None)
    concurrency = get_settings().models.llm.limits.background_concurrency
    eta_seconds = (
        groups["pending"] * (p50_ms / 1000.0) / concurrency if p50_ms is not None else None
    )
    return {
        "queued_kg_extract_jobs": queued,
        "groups_total": groups["total"],
        "groups_cached": groups["cached"],
        "groups_pending": groups["pending"],
        "eta_seconds": eta_seconds,
    }


def _drift_summary() -> dict[str, Any] | None:
    """The last reconcile report, reduced to what the dashboard shows."""
    from mynotes.index.reconcile import load_last_report

    report = load_last_report()
    if report is None:
        return None
    return {
        "saved_at": report.get("saved_at"),
        "expected": report.get("expected"),
        "indexed": report.get("indexed"),
        "missing": len(report.get("missing") or []),
        "extra": len(report.get("extra") or []),
        "stale": len(report.get("stale") or []),
    }
