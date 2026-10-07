"""Shared types for the retrieval pipeline (PLAN.md §9).

:class:`SearchFilters` is the filter set every retrieval entry point accepts
(the CLI, the HTTP API and later the fusion in task 28), and
:class:`SearchHit` is one ranked chunk result. ``sources`` and
``rerank_score`` stay unfilled here: they are set by the client-side fusion
and the reranker in task 28.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SearchFilters:
    """Filters applied inside every hybrid-search leg (PLAN.md §6.1, §9 step 2).

    Attributes:
        tags: Normalized tag names; all must match.
        folders: User folder prefixes, matched folded.
        formats: File formats (``md``, ``pdf``, ...).
        since: Only documents dated on or after this instant.
        until: Only documents dated on or before this instant.
        include_superseded: History intent: also match superseded documents.
        include_old_versions: History intent plus ``versions.index_old_versions``:
            also match chunks whose version is not the latest.
        acl_prefixes: Exact path prefixes the caller may access; ``None`` means
            no ACL restriction (local mode), an empty list means no access.
        workspace_id: Restrict to one workspace, or None for the default.
        document_ids: Restrict the search to these documents (version
            expansion, task 35); empty means no restriction.
    """

    tags: list[str] = field(default_factory=list)
    folders: list[str] = field(default_factory=list)
    formats: list[str] = field(default_factory=list)
    since: datetime | None = None
    until: datetime | None = None
    include_superseded: bool = False
    include_old_versions: bool = False
    acl_prefixes: list[str] | None = None
    workspace_id: int | None = None
    document_ids: list[int] = field(default_factory=list)


@dataclass
class SearchHit:
    """One ranked chunk result of the hybrid search.

    Attributes:
        chunk_id: The chunk id as stored in the index.
        document_id: The owning document id.
        version_id: The owning document version id.
        rank: 1-based position in the result list.
        score: The RRF score from OpenSearch (rank-based, tiny).
        text: The chunk text.
        title: The document title, or None.
        section: The chunk's section label, or None.
        heading_path: The chunk's heading path joined with ``" > "``.
        path: The file path relative to its source root.
        page: The PDF page number, or None.
        slide: The PPTX slide number, or None.
        doc_date: The document date, or None.
        doc_status: The document status (``current``, ``archived``, ...).
        is_latest: Whether the chunk belongs to the latest version.
        is_table: Whether the chunk is a table chunk.
        format: The document format (``md``, ``pdf``, ...).
        language: The detected language, or None.
        tags: The document's index tags.
        sources: Retrieval source name to rank; filled by the fusion (task 28).
        rerank_score: The reranker score; filled in task 28.
    """

    chunk_id: str
    document_id: int
    version_id: int
    rank: int
    score: float
    text: str
    title: str | None
    section: str | None
    heading_path: str
    path: str
    page: int | None
    slide: int | None
    doc_date: datetime | None
    doc_status: str
    is_latest: bool
    is_table: bool
    format: str
    language: str | None
    tags: list[str]
    sources: dict[str, int] = field(default_factory=dict)
    rerank_score: float | None = None
    query_hint: str | None = None
    """For second-hop evidence: the entity that led to this hit (e.g. the tool named by
    the top document). The reranker also scores such hits against ``query + hint``."""
    query_hint_type: str | None = None
    """Entity type of ``query_hint``; promotion prefers the type the question asks for."""
