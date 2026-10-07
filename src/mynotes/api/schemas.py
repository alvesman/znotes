"""Pydantic request/response models for the HTTP API (PLAN.md §10.2)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mynotes.config import get_settings
from mynotes.retrieval.types import SearchFilters


class FiltersModel(BaseModel):
    """Base filters for ask and search; inline query tokens win over these."""

    tags: list[str] = Field(default_factory=list)
    folders: list[str] = Field(default_factory=list)
    formats: list[str] = Field(default_factory=list)
    since: datetime | None = None
    until: datetime | None = None
    history: bool = False

    def to_search_filters(self) -> SearchFilters:
        """Convert to the pipeline's :class:`SearchFilters`.

        ``history`` selects superseded documents and, when
        ``versions.index_old_versions`` is on, old versions too (the CLI's
        ``--history`` behavior).
        """
        history = self.history
        return SearchFilters(
            tags=list(self.tags),
            folders=list(self.folders),
            formats=list(self.formats),
            since=self.since,
            until=self.until,
            include_superseded=history,
            include_old_versions=history and get_settings().versions.index_old_versions,
        )


class AskRequest(BaseModel):
    """One question with optional base filters."""

    question: str
    filters: FiltersModel | None = None
    save_history: bool = True
    """Streamed answers only: keep the question and answer in the chat history."""
    conversation_id: str | None = Field(default=None, max_length=64)
    """Groups the questions of one chat in the history (sent by the chat page)."""


class CitationModel(BaseModel):
    """One answer citation (as serialized by the answer layer)."""

    n: int
    title: str | None
    path: str
    page: int | None
    slide: int | None
    section: str | None
    doc_date: str | None
    is_current: bool
    document_id: int
    format: str | None = None


class AskResponse(BaseModel):
    """The complete answer for one question."""

    answer: str
    citations: list[CitationModel]
    relevant: bool
    timings_ms: dict[str, int]


class SearchRequest(BaseModel):
    """One search query with optional base filters."""

    query: str
    filters: FiltersModel | None = None
    rerank: bool = True
    limit: int = Field(default=10, ge=1)


class HitModel(BaseModel):
    """One ranked search hit."""

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
    sources: dict[str, int]
    rerank_score: float | None


class AnalysisModel(BaseModel):
    """The query analysis of one search (``filters`` as applied)."""

    intent: str
    text: str
    filters: FiltersModel


class SearchResponse(BaseModel):
    """The ranked hits and the query analysis of one search."""

    hits: list[HitModel]
    analysis: AnalysisModel


class SectionModel(BaseModel):
    """One section of a document's latest version."""

    section: str | None
    page: int | None
    slide: int | None
    first_ordinal: int


class DocumentModel(BaseModel):
    """One document with its extracted content and section map."""

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
    sections: list[SectionModel]


class VersionInfoModel(BaseModel):
    """One retained snapshot in a document's version timeline."""

    version_no: int
    created_at: datetime
    doc_date: datetime | None
    rel_path: str
    status: str
    chunk_count: int
    is_latest: bool
    changed_sections: list[str]


class TagModel(BaseModel):
    """One row of the tag vocabulary listing."""

    name: str
    kind: str
    document_count: int
    sources: list[str]


class DocRefModel(BaseModel):
    """One endpoint of a version relation."""

    document_id: int
    title: str | None
    path: str
    doc_date: datetime | None
    status: str


class RelationModel(BaseModel):
    """One supersedes relation (task 35 review)."""

    id: int
    status: str
    confidence: float
    method: str
    newer: DocRefModel
    older: DocRefModel
    evidence: dict[str, Any]


class MergeTagsRequest(BaseModel):
    """Merge one tag into another (JSON keys ``from``/``into``)."""

    model_config = ConfigDict(populate_by_name=True)

    from_name: str = Field(alias="from")
    into: str


class ScanRequest(BaseModel):
    """Scan the enabled roots (or one named root)."""

    root: str | None = None
    full: bool = False


class TimelineItemModel(BaseModel):
    """One row of an entity or project timeline."""

    date: datetime | None
    kind: str
    text: str
    status: str
    document_path: str | None


class ErrorModel(BaseModel):
    """The error body of failed responses."""

    error: str
    detail: str
    candidates: list[dict[str, Any]] | None = None
