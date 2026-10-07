"""MCP server exposing mynotes as tools and resources over stdio (PLAN.md §10.3).

A thin layer over the core facade: every tool is a sync function that calls
:mod:`mynotes.core.service` with ``principal=LOCAL`` (stdio is local) and
returns JSON-serializable data. Facade failures come back as
``{"error": <kind>, "message": ...}`` dicts (plus ``candidates`` for ambiguous
references) so the client model can react instead of the call raising.
Run with ``mynotes mcp``; stdout is the protocol channel, so logging goes to
stderr and nothing here may print to stdout.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any
from urllib.parse import unquote

from mcp.server.fastmcp import FastMCP
from sqlalchemy import func, select

from mynotes.core import service
from mynotes.core.errors import Ambiguous, Forbidden, NotFound
from mynotes.core.principal import LOCAL
from mynotes.retrieval.answer import citation_to_dict
from mynotes.retrieval.assemble import Citation
from mynotes.retrieval.types import SearchFilters, SearchHit

logger = logging.getLogger(__name__)

mcp = FastMCP("mynotes")

_SNIPPET_CHARS = 500
"""Maximum snippet length in ``search_notes`` results."""

_MARKDOWN_CHARS = 20_000
"""Maximum markdown length in ``get_document`` results."""

_TRUNCATION_NOTE = (
    "\n\n[truncated: showing the first {shown} of {total} characters; "
    "use search_notes to locate specific passages]"
)

_ERROR_KINDS: dict[type[Exception], str] = {
    NotFound: "not_found",
    Ambiguous: "ambiguous",
    Forbidden: "forbidden",
}
"""Tool error kinds, aligned with the HTTP API's error bodies."""


def _tool_error(exc: Exception) -> dict[str, Any]:
    """Render one exception as the tool error dict."""
    payload: dict[str, Any] = {
        "error": _ERROR_KINDS.get(type(exc), type(exc).__name__.lower()),
        "message": str(exc),
    }
    if isinstance(exc, Ambiguous):
        payload["candidates"] = list(exc.candidates)
    return payload


def _safe(tool: str, function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Run one tool implementation and turn failures into error dicts."""
    try:
        return function(*args, **kwargs)
    except (NotFound, Ambiguous, Forbidden) as exc:
        logger.info("%s: %s", tool, exc)
        return _tool_error(exc)
    except Exception as exc:
        logger.exception("%s failed", tool)
        return {
            "error": type(exc).__name__.lower(),
            "message": f"{tool} failed: {type(exc).__name__}; details in the server log on stderr",
        }


def _filters(tags: Sequence[str], folder: str | None, history: bool) -> SearchFilters:
    """Build the pipeline filters from tool arguments (``history`` matches the CLI flag)."""
    from mynotes.config import get_settings

    return SearchFilters(
        tags=list(tags),
        folders=[folder] if folder else [],
        include_superseded=history,
        include_old_versions=history and get_settings().versions.index_old_versions,
    )


def _iso(value: Any) -> str | None:
    """Format a datetime as ISO 8601, or None."""
    return value.isoformat() if value is not None else None


def _hit_view(hit: SearchHit) -> dict[str, Any]:
    """One search hit as a compact tool result row."""
    return {
        "rank": hit.rank,
        "title": hit.title,
        "path": hit.path,
        "page": hit.page,
        "slide": hit.slide,
        "section": hit.section,
        "snippet": hit.text[:_SNIPPET_CHARS],
        "doc_date": _iso(hit.doc_date),
        "is_current": hit.doc_status == "current" and hit.is_latest,
        "document_id": hit.document_id,
    }


def _citation_view(citation: Citation) -> dict[str, Any]:
    """One answer citation, projected to the compact tool fields."""
    full = citation_to_dict(citation)
    return {
        key: full[key] for key in ("n", "title", "path", "page", "slide", "doc_date", "is_current")
    }


def _run_search_notes(
    query: str, tags: list[str], folder: str | None, history: bool, limit: int
) -> list[dict[str, Any]]:
    result = service.search(query, _filters(tags, folder, history), principal=LOCAL)
    return [_hit_view(hit) for hit in result.hits[: max(1, limit)]]


def _search_notes_impl(
    query: str,
    tags: list[str] | None = None,
    folder: str | None = None,
    history: bool = False,
    limit: int = 8,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Search implementation behind :func:`search_notes`."""
    return _safe("search_notes", _run_search_notes, query, list(tags or []), folder, history, limit)


def _run_ask_notes(
    question: str, tags: list[str], folder: str | None, history: bool
) -> dict[str, Any]:
    result = service.ask(question, _filters(tags, folder, history), principal=LOCAL)
    return {
        "answer": result.answer,
        "citations": [_citation_view(citation) for citation in result.citations],
        "relevant": result.relevant,
    }


def _ask_notes_impl(
    question: str,
    tags: list[str] | None = None,
    folder: str | None = None,
    history: bool = False,
) -> dict[str, Any]:
    """Ask implementation behind :func:`ask_notes`."""
    return _safe("ask_notes", _run_ask_notes, question, list(tags or []), folder, history)


def _run_get_document(ref: str) -> dict[str, Any]:
    view = service.get_document(ref, principal=LOCAL)
    markdown = view.markdown
    if len(markdown) > _MARKDOWN_CHARS:
        markdown = markdown[:_MARKDOWN_CHARS] + _TRUNCATION_NOTE.format(
            shown=_MARKDOWN_CHARS, total=len(view.markdown)
        )
    return {
        "document_id": view.document_id,
        "title": view.title,
        "path": view.path,
        "format": view.format,
        "status": view.status,
        "doc_date": _iso(view.doc_date),
        "tags": list(view.tags),
        "markdown": markdown,
    }


def _get_document_impl(ref: str) -> dict[str, Any]:
    """Get-document implementation behind :func:`get_document`."""
    return _safe("get_document", _run_get_document, ref)


def _run_document_history(ref: str) -> list[dict[str, Any]]:
    infos = service.document_history(ref, principal=LOCAL)
    return [
        {
            "version_no": info.version_no,
            "created_at": _iso(info.created_at),
            "doc_date": _iso(info.doc_date),
            "changed_sections": list(info.changed_sections),
            "is_latest": info.is_latest,
        }
        for info in infos
    ]


def _document_history_impl(ref: str) -> list[dict[str, Any]] | dict[str, Any]:
    """History implementation behind :func:`document_history`."""
    return _safe("document_history", _run_document_history, ref)


def _run_list_tags(prefix: str | None) -> list[dict[str, Any]]:
    return [
        {"tag": tag.name, "kind": tag.kind, "documents": tag.document_count}
        for tag in service.list_tags(prefix, principal=LOCAL)
    ]


def _list_tags_impl(prefix: str | None = None) -> list[dict[str, Any]] | dict[str, Any]:
    """Tags implementation behind :func:`list_tags`."""
    return _safe("list_tags", _run_list_tags, prefix)


def _run_entity_neighbors(name: str, hops: int) -> dict[str, Any]:
    return service.entity_neighbors(name, max(1, hops), principal=LOCAL)


def _entity_neighbors_impl(name: str, hops: int = 1) -> dict[str, Any]:
    """Entity-neighbours implementation behind :func:`entity_neighbors`."""
    return _safe("entity_neighbors", _run_entity_neighbors, name, hops)


def _run_project_timeline(name: str) -> list[dict[str, Any]]:
    items = service.timeline(f"project:{name.strip()}", principal=LOCAL)
    return [
        {
            "date": item.date.date().isoformat() if item.date else None,
            "kind": item.kind,
            "text": item.text,
            "status": item.status,
            "document_path": item.document_path,
        }
        for item in items
    ]


def _project_timeline_impl(name: str) -> list[dict[str, Any]] | dict[str, Any]:
    """Timeline implementation behind :func:`project_timeline`."""
    return _safe("project_timeline", _run_project_timeline, name)


def _last_scans() -> list[dict[str, Any]]:
    """The most recent scan run per source root, newest first."""
    from mynotes.db.models import ScanRun, SourceRoot
    from mynotes.db.session import session_scope

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
            "started_at": _iso(started_at),
            "finished_at": _iso(finished_at),
            "counts": dict(counts or {}),
            "error": error,
        }
        for name, kind, status, started_at, finished_at, counts, error in rows
    ]


def _kg_backlog() -> dict[str, Any]:
    """Queued KG extraction jobs and the pending-group estimate."""
    from mynotes.db.models import Job
    from mynotes.db.session import session_scope
    from mynotes.graph.extract import groups_overview

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
    return {
        "queued_kg_extract_jobs": queued,
        "groups_total": groups["total"],
        "groups_cached": groups["cached"],
        "groups_pending": groups["pending"],
    }


def _run_scan_status() -> dict[str, Any]:
    status = service.status(principal=LOCAL)
    postgres = status.get("postgres") or {}
    jobs = postgres.get("jobs") or {}
    queue = (
        {
            "queued": int(jobs.get("queued", 0)),
            "running": int(jobs.get("running", 0)),
            "failed": int(jobs.get("failed", 0)),
            "outbox_unprocessed": int(postgres.get("outbox_unprocessed", 0)),
        }
        if "error" not in postgres
        else {"error": str(postgres["error"])}
    )
    endpoints = [
        {
            "name": snapshot.get("name"),
            "state": snapshot.get("state"),
            "calls": snapshot.get("calls"),
            "error_rate": snapshot.get("error_rate"),
            "last_error": snapshot.get("last_error"),
        }
        for snapshot in status.get("endpoints") or []
        if "error" not in snapshot
    ]
    return {
        "last_scans": _last_scans(),
        "queue": queue,
        "kg_backlog": _kg_backlog(),
        "endpoints": endpoints,
    }


def _scan_status_impl() -> dict[str, Any]:
    """Scan-status implementation behind :func:`scan_status`."""
    return _safe("scan_status", _run_scan_status)


@mcp.tool()
def search_notes(
    query: str,
    tags: list[str] | None = None,
    folder: str | None = None,
    history: bool = False,
    limit: int = 8,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Search the notes corpus with hybrid keyword + semantic retrieval (reranked).

    Best for finding *where* something is written: returns the top matching
    passages with the document each came from. Use get_document to read a full
    document, or ask_notes for a synthesized answer with citations. When the
    result is empty, nothing in the corpus matched; try broader terms.

    Args:
        query: Free-text query. May include inline filter tokens such as
            ``tag:projeto-x``, ``folder:work`` or ``since:2026-01``.
        tags: Only documents carrying ALL of these accepted tags.
        folder: Only documents whose path starts with this prefix (relative to
            a source root), e.g. ``work/reports``.
        history: Set true to include superseded documents and old versions
            (history intent); default searches only current content.
        limit: Maximum number of hits to return (default 8).
    """
    return _search_notes_impl(query, tags, folder, history, limit)


@mcp.tool()
def ask_notes(
    question: str,
    tags: list[str] | None = None,
    folder: str | None = None,
    history: bool = False,
) -> dict[str, Any]:
    """Answer a question from the notes with a grounded, cited response.

    Runs retrieval over the corpus and generates an answer whose claims carry
    ``[n]`` citation markers resolved in ``citations``. Use this when the user
    asks a question whose answer should be synthesized from the notes; use
    search_notes to just locate passages. When ``relevant`` is false the
    corpus had nothing relevant to the question.

    Args:
        question: The user's question in natural language (inline filter
            tokens like ``tag:x`` are honored).
        tags: Only retrieve from documents carrying ALL of these accepted tags.
        folder: Only retrieve from documents under this path prefix (relative
            to a source root).
        history: Set true to answer about past states (superseded documents
            and old versions) instead of only current content.
    """
    return _ask_notes_impl(question, tags, folder, history)


@mcp.tool()
def get_document(ref: str) -> dict[str, Any]:
    """Read one document's metadata and its full extracted Markdown text.

    Resolves ``ref`` as a document id, a path relative to a source root, or a
    title. Use it to read the whole content of a document found via
    search_notes or cited in ask_notes. The ``markdown`` field is capped at
    20,000 characters and marked when truncated. If the reference matches
    several documents the result is ``{"error": "ambiguous", "candidates":
    [...]}``; retry with one candidate's id or exact path.

    Args:
        ref: Document id, path relative to a source root, or (partial) title.
    """
    return _get_document_impl(ref)


@mcp.tool()
def document_history(ref: str) -> list[dict[str, Any]] | dict[str, Any]:
    """List a document's retained version snapshots, newest first.

    Each entry reports when the snapshot was recorded, the document date at
    that version, which sections changed against the previous snapshot and
    whether it is the document's latest version. Use it for questions about
    how a document evolved over time. ``ref`` resolves like get_document's.

    Args:
        ref: Document id, path relative to a source root, or (partial) title.
    """
    return _document_history_impl(ref)


@mcp.tool()
def list_tags(prefix: str | None = None) -> list[dict[str, Any]] | dict[str, Any]:
    """List the tag vocabulary with accepted-usage counts.

    Use it to discover which tags exist before filtering search_notes or
    ask_notes with ``tags=[...]``.

    Args:
        prefix: Only tags whose name starts with this prefix.
    """
    return _list_tags_impl(prefix)


@mcp.tool()
def entity_neighbors(name: str, hops: int = 1) -> dict[str, Any]:
    """Show the knowledge-graph neighbours of one entity.

    Explores what the notes connect: people, projects, products and their
    relations, scored by evidence strength and limited to 20 neighbours.
    ``name`` resolves exactly, normalized or fuzzily; the result is
    ``{"error": "ambiguous", "candidates": [...]}`` when several entities
    match.

    Args:
        name: The entity's name (a person, project, product, ...).
        hops: Maximum relation steps to walk (default 1, at least 1).
    """
    return _entity_neighbors_impl(name, hops)


@mcp.tool()
def project_timeline(name: str) -> list[dict[str, Any]] | dict[str, Any]:
    """Show the chronological timeline of one project tag's documents.

    Each entry is one dated event (a document or a dated fact) with its
    status, sorted by date; undated entries come last. Use it for "what
    happened in project X over time" questions.

    Args:
        name: The project tag's name without the ``project:`` prefix, e.g.
            ``beta-launch`` for the tag ``project:beta-launch``.
    """
    return _project_timeline_impl(name)


@mcp.tool()
def scan_status() -> dict[str, Any]:
    """Report the ingestion pipeline's health in compact form.

    Returns the last scan per source root, the job-queue depth (queued,
    running, failed, unprocessed index-outbox entries), the knowledge-graph
    extraction backlog and the model-endpoint states (breaker state and error
    rates). Use it when answers look stale: an empty index, a large queue or
    a large KG backlog means results may be incomplete. The worker process
    runs separately and must be up for the index to stay fresh.
    """
    return _scan_status_impl()


@mcp.resource("note://{path}", mime_type="text/markdown")
def note(path: str) -> str:
    """Read the extracted Markdown of one document.

    Build the URI from the document's path (relative to its source root, as
    returned by search_notes/get_document), percent-encoded so it contains no
    slashes: ``note://work%2Freports%2Fplan.md`` for ``work/reports/plan.md``.
    """
    view = _get_document_impl(unquote(path))
    if "error" in view:
        raise ValueError(str(view.get("message") or view["error"]))
    return str(view.get("markdown") or "")
