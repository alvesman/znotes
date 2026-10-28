"""Server-rendered web UI routes (PLAN.md §10.4).

HTML pages under ``/ui`` share the FastAPI app with the JSON API. Every
response carries a strict CSP (no inline scripts or styles), so all JS lives
in ``static/`` and rendering never touches model output as HTML.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Form, Query, Request
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse, Response

from mynotes import PROJECT_URL, __version__
from mynotes.api import schemas
from mynotes.api.auth import TOKEN_COOKIE, PrincipalDep
from mynotes.config import get_settings
from mynotes.core import service
from mynotes.core.errors import Ambiguous, NotFound
from mynotes.core.principal import Principal
from mynotes.retrieval.types import SearchHit
from mynotes.web.render import heading_slug, render_document_markdown, render_help_markdown

router = APIRouter(prefix="/ui")

STATIC_DIR = Path(__file__).parent / "static"
"""Directory served at ``/static`` by the app factory."""

DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"
"""The repository's ``docs/`` folder (``/app/docs`` in the Docker image)."""

HELP_DOCS: dict[str, tuple[str, str]] = {
    "cli": ("Setup and command line", "user-manual.md"),
    "web": ("Web interface", "web-interface-manual.md"),
    "mcp": ("Claude and MCP", "mcp.md"),
}
"""Help tabs: key -> (label, file under DOCS_DIR). The first is the default."""

SOURCE_BASE_URL = f"{PROJECT_URL}/blob/main"
"""Where relative manual links to files not shown in the UI point."""

FORMATS = ["md", "txt", "pdf", "docx", "pptx", "xlsx", "epub", "html"]
"""The checkbox options of the search form, in display order."""

_MAX_TAGS = 500
_COOKIE_MAX_AGE_S = 30 * 24 * 3600
_SNIPPET_CHARS = 300

_UI_CSP = (
    "default-src 'self'; img-src 'self' data:; style-src 'self'; "
    "script-src 'self'; frame-ancestors 'none'"
)


def _sec_anchor(section: str | None) -> str:
    """The fragment for a section anchor, or an empty string."""
    return f"#sec-{heading_slug(section)}" if section else ""


def _fmt_day(value: Any) -> str:
    """A date as ``YYYY-MM-DD`` (ISO strings pass through, empty for None)."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        return f"{value:%Y-%m-%d}"
    text = str(value)
    return text[:10] if len(text) >= 10 and text[4] == "-" and text[7] == "-" else text


def _fmt_duration(seconds: Any) -> str:
    """Seconds as a compact human string (``45s``, ``3 min``, ``2 h 05 min``)."""
    if seconds is None:
        return "unknown"
    total = int(seconds)
    if total < 90:
        return f"{total}s"
    minutes = round(total / 60)
    if minutes < 90:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"


templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
templates.env.filters["slug"] = heading_slug
templates.env.filters["day"] = _fmt_day
templates.env.filters["duration"] = _fmt_duration


def _cytoscape_src() -> str:
    """The ``/static`` URL of the vendored cytoscape release."""
    releases = sorted(STATIC_DIR.glob("cytoscape-*.min.js"))
    return f"/static/{releases[-1].name}" if releases else ""


templates.env.globals["CYTOSCAPE_SRC"] = _cytoscape_src()


def _asset(name: str) -> str:
    """``/static/<name>`` with the file's mtime as ``?v=``: a changed script or stylesheet is
    never served from the browser cache (the static mount sends no cache headers)."""
    try:
        stamp = int((STATIC_DIR / name).stat().st_mtime)
    except OSError:
        return f"/static/{name}"
    return f"/static/{name}?v={stamp}"


templates.env.globals["asset"] = _asset
templates.env.globals["PROJECT_URL"] = PROJECT_URL
templates.env.globals["VERSION"] = __version__


def _render(
    request: Request, name: str, context: dict[str, Any], status_code: int = 200
) -> Response:
    """Render one UI template with the UI security headers."""
    context.setdefault("signed_in_user", _signed_in_user(request))
    context.setdefault("is_admin", _is_admin(request))
    return templates.TemplateResponse(
        request,
        name,
        context,
        status_code=status_code,
        headers={"Content-Security-Policy": _UI_CSP},
    )


def _is_admin(request: Request) -> bool:
    """Whether the menu offers the admin pages (Status): the caller's principal decides."""
    principal = getattr(request.state, "principal", None)
    if principal is None:  # the sign-in page: nobody yet, unless in local mode
        return get_settings().auth.mode == "none"
    return bool(principal.is_admin)


def _signed_in_user(request: Request) -> str | None:
    """The name shown in the top bar with a sign-out link; None in local mode."""
    if get_settings().auth.mode == "none":
        return None
    principal = getattr(request.state, "principal", None)
    return principal.user if principal is not None else None


def _safe_next(value: str | None) -> str:
    """Where to go after signing in: a UI path from the same site, else the chat page."""
    if value and value.startswith("/ui/") and not value.startswith("//") and "\n" not in value:
        return value
    return "/ui/chat"


@router.get("/login")
def login_page(request: Request, next: str | None = None) -> Response:
    """The sign-in page (server mode): paste a personal token."""
    return _render(
        request,
        "login.html",
        {"nav": "login", "next": _safe_next(next), "error": None, "signed_in_user": None},
    )


@router.post("/login")
def login_submit(
    request: Request,
    token: Annotated[str, Form()] = "",
    next: Annotated[str, Form()] = "",
) -> Response:
    """Check the token and set the ``mynotes_token`` cookie."""
    from mynotes.core import tokens

    info = tokens.verify_token(token.strip()) if token.strip() else None
    error = None
    if info is None:
        error = "That token is not valid (or was revoked). Check it and try again."
    elif not tokens.can_read(info):
        error = "That token is for an agent and cannot sign in. Ask for a web token."
    if error is not None:
        return _render(
            request,
            "login.html",
            {"nav": "login", "next": _safe_next(next), "error": error, "signed_in_user": None},
        )
    response = RedirectResponse(_safe_next(next), status_code=303)
    response.set_cookie(
        TOKEN_COOKIE,
        token.strip(),
        httponly=True,
        samesite="lax",
        secure=_is_https(request),
        max_age=_COOKIE_MAX_AGE_S,
        path="/",
    )
    return response


@router.post("/logout")
def logout(request: Request) -> Response:
    """Clear the cookie; the login page follows in server mode."""
    target = "/ui/login" if get_settings().auth.mode != "none" else "/ui/chat"
    response = RedirectResponse(target, status_code=303)
    response.delete_cookie(TOKEN_COOKIE, path="/")
    return response


def _is_https(request: Request) -> bool:
    forwarded = request.headers.get("x-forwarded-proto", "")
    return request.url.scheme == "https" or forwarded.split(",")[0].strip() == "https"


def _parse_since(value: str) -> datetime | None:
    """Parse a ``<input type="month">`` value; invalid input is ignored."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m").replace(tzinfo=UTC)
    except ValueError:
        return None


def _remote_ids(principal: Principal, items: Iterable[dict[str, Any]]) -> set[int]:
    """The cited documents uploaded by an agent, which get no "Open in app" button."""
    if not principal.can_open_files:
        return set()  # the button is not shown at all
    return service.remote_document_ids(
        item["document_id"]
        for item in items
        if item.get("format") in service.OPEN_IN_APP_FORMATS and item.get("document_id") is not None
    )


def _hit_item(hit: SearchHit) -> dict[str, Any]:
    """One search hit as template context, shaped like a citation."""
    return {
        "n": None,
        "title": hit.title,
        "path": hit.path,
        "page": hit.page,
        "slide": hit.slide,
        "section": hit.section,
        "doc_date": f"{hit.doc_date:%Y-%m-%d}" if hit.doc_date is not None else None,
        "is_current": hit.doc_status == "current" and hit.is_latest,
        "document_id": hit.document_id,
        "format": hit.format,
        "snippet": hit.text[:_SNIPPET_CHARS],
        "tags": list(hit.tags),
    }


@router.get("/chat")
def chat_page(
    request: Request,
    principal: PrincipalDep,
    conversation: Annotated[str | None, Query(max_length=64)] = None,
) -> Response:
    """The chat page with the streaming answer form.

    With ``?conversation=<key>`` (History's "Continue this chat") the chat's earlier
    questions and answers are shown and new questions join the same conversation; an
    unknown key (or someone else's) opens an empty chat.
    """
    context: dict[str, Any] = {
        "nav": "chat",
        "can_open_files": principal.can_open_files,
        "conversation_id": None,
        "turns": [],
    }
    if conversation:
        from mynotes.core.chat_history import get_conversation

        found = get_conversation(conversation, principal=principal)
        if found is not None:
            context["conversation_id"] = found.key
            context["turns"] = [_turn_item(turn) for turn in found.turns]
    context["remote_ids"] = _remote_ids(
        principal, (item for turn in context["turns"] for item in turn["citations"])
    )
    return _render(request, "chat.html", context)


def _tag_options(principal: Principal) -> list[Any]:
    """The search page's tag choices: tags in use, most used first."""
    used = [tag for tag in service.list_tags(principal=principal) if tag.document_count > 0]
    used.sort(key=lambda tag: (-tag.document_count, tag.name))
    return used[:_MAX_TAGS]


@router.get("/search")
def search_page(request: Request, principal: PrincipalDep) -> Response:
    """The search page with the facet form and tag options."""
    return _render(
        request,
        "search.html",
        {
            "nav": "search",
            "can_open_files": principal.can_open_files,
            "tags": _tag_options(principal),
            "formats": FORMATS,
        },
    )


@router.post("/partials/search")
def partial_search(
    request: Request,
    principal: PrincipalDep,
    query: Annotated[str, Form()] = "",
    tags: Annotated[list[str] | None, Form()] = None,
    folder: Annotated[str, Form()] = "",
    formats: Annotated[list[str] | None, Form()] = None,
    since: Annotated[str, Form()] = "",
    history: Annotated[bool, Form()] = False,
    rerank: Annotated[bool, Form()] = False,
) -> Response:
    """The HTMX result partial for the search form."""
    context: dict[str, Any] = {
        "can_open_files": principal.can_open_files,
        "hits": [],
        "remote_ids": set(),
    }
    if not query.strip():
        return _render(request, "partials/search_results.html", context)
    filters = schemas.FiltersModel(
        tags=[tag for tag in (tags or []) if tag],
        folders=[folder] if folder else [],
        formats=[fmt for fmt in (formats or []) if fmt in FORMATS],
        since=_parse_since(since),
        history=history,
    )
    result = service.search(query, filters.to_search_filters(), principal=principal, rerank=rerank)
    strong, weak = split_weak_hits(result.hits)
    context["hits"] = [_hit_item(hit) for hit in strong]
    context["weak_hits"] = [_hit_item(hit) for hit in weak]
    context["remote_ids"] = _remote_ids(principal, [*context["hits"], *context["weak_hits"]])
    return _render(request, "partials/search_results.html", context)


SEARCH_WEAK_SHARE = 0.05
"""Search results whose rerank score is below this share of the best one are collapsed."""


def split_weak_hits(
    hits: Sequence[SearchHit], share: float = SEARCH_WEAK_SHARE
) -> tuple[list[SearchHit], list[SearchHit]]:
    """Split reranked search hits into the relevant ones and the far weaker rest.

    The semantic half of the hybrid search always returns the nearest passages, related
    or not, so a query that matches one note ("portainer") fills the page with Docker
    and server notes the reranker scores near zero (0.002 against 0.84). Only the search
    page applies this cut: answers keep ``retrieval.min_rerank_score`` 0, because their
    model declines by itself. Without rerank scores nothing is split.
    """
    scores = [hit.rerank_score for hit in hits if hit.rerank_score is not None]
    if not scores or max(scores) <= 0:
        return list(hits), []
    floor = share * max(scores)
    strong = [hit for hit in hits if hit.rerank_score is None or hit.rerank_score >= floor]
    weak = [hit for hit in hits if hit not in strong]
    return strong, weak


def _when(value: datetime) -> str:
    """A stored timestamp in local time, to the minute."""
    return f"{value.astimezone():%Y-%m-%d %H:%M}"


def _turn_item(turn: Any) -> dict[str, Any]:
    """One stored question and answer as template context (History and resumed chats)."""
    return {
        "id": turn.id,
        "question": turn.question,
        "standalone": getattr(turn, "standalone_question", None),
        "answer": turn.answer,
        "relevant": turn.relevant,
        "when": _when(turn.created_at),
        "citations": [
            {**citation, "doc_date": _fmt_day(citation.get("doc_date")) or None}
            for citation in turn.citations
        ],
    }


def _history_context(principal: Principal, message: str | None = None) -> dict[str, Any]:
    """The History page's list: the caller's conversations, most recently active first."""
    from mynotes.core.chat_history import list_conversations

    conversations, total = list_conversations(principal=principal)

    items = [
        {
            "key": conversation.key,
            "title": conversation.turns[0].question,
            "count": len(conversation.turns),
            "started": _when(conversation.started_at),
            "last": _when(conversation.last_at),
            "turns": [_turn_item(turn) for turn in conversation.turns],
        }
        for conversation in conversations
    ]
    return {
        "nav": "history",
        "can_open_files": principal.can_open_files,
        "remote_ids": _remote_ids(
            principal,
            (
                citation
                for conversation in items
                for turn in conversation["turns"]
                for citation in turn["citations"]
            ),
        ),
        "conversations": items,
        "total": total,
        "message": message,
    }


@router.get("/history")
def history_page(request: Request, principal: PrincipalDep) -> Response:
    """The History page: past chat questions and answers, with a cleanup form."""
    return _render(request, "history.html", _history_context(principal))


@router.post("/partials/history/delete")
def history_delete(
    request: Request,
    principal: PrincipalDep,
    scope: Annotated[str, Form()] = "older",
    days: Annotated[int | None, Form()] = None,
) -> Response:
    """Delete the caller's chats idle for ``days`` days (``scope=older``) or all of them."""
    from mynotes.core.chat_history import delete_chats

    if scope == "all":
        deleted = delete_chats(principal=principal)
        message = f"Deleted all chats ({deleted})."
    elif days is None or days < 1:
        message = "Enter a number of days (1 or more)."
        return _render(request, "partials/history_list.html", _history_context(principal, message))
    else:
        deleted = delete_chats(older_than_days=days, principal=principal)
        chats = f"{deleted} chat{'s' if deleted != 1 else ''}"
        message = f"Deleted {chats} not used for {days} day{'s' if days != 1 else ''}."
    return _render(request, "partials/history_list.html", _history_context(principal, message))


def _document_context(document_id: int, principal: Principal, chunk: int | None) -> dict[str, Any]:
    """The template context of the document page (task 48)."""
    view = service.get_document(document_id, principal=principal)
    newer, older = service.version_chain(document_id, principal=principal)
    chunk_text = None
    if chunk is not None and chunk >= 0:
        chunk_text = service.document_chunk_text(document_id, chunk, principal=principal)
    return {
        "view": view,
        "tags": service.document_tags(document_id, principal=principal),
        "newer": newer,
        "older": older,
        "history": service.document_history(document_id, principal=principal),
        "content_html": render_document_markdown(view.markdown, chunk_text),
    }


@router.get("/documents/{document_id}")
def document_page(
    request: Request,
    principal: PrincipalDep,
    document_id: int,
    chunk: Annotated[int | None, Query()] = None,
) -> Response:
    """The document view: content, tags and version chain (task 48)."""
    context = _document_context(document_id, principal, chunk)
    context.update({"nav": "documents", "can_open_files": principal.can_open_files})
    return _render(request, "document.html", context)


@router.get("/partials/documents/{document_id}/tags")
def partial_document_tags(request: Request, principal: PrincipalDep, document_id: int) -> Response:
    """The tag chips fragment of a document (re-rendered after Accept/Reject)."""
    context: dict[str, Any] = {
        "view": service.get_document(document_id, principal=principal),
        "tags": service.document_tags(document_id, principal=principal),
    }
    return _render(request, "partials/document_tags.html", context)


def _review_version_rows(principal: Principal) -> list[dict[str, Any]]:
    """Proposed relations with a 300-character preview per endpoint document."""
    relations = service.list_version_relations("proposed", principal=principal)
    previews = service.document_previews(
        [
            document_id
            for relation in relations
            for document_id in (relation.newer.document_id, relation.older.document_id)
        ],
        principal=principal,
    )
    return [
        {
            "relation": relation,
            "newer_preview": previews.get(relation.newer.document_id, ""),
            "older_preview": previews.get(relation.older.document_id, ""),
        }
        for relation in relations
    ]


def _render_review_versions(request: Request, principal: Principal) -> Response:
    return _render(
        request,
        "partials/review_versions.html",
        {"rows": _review_version_rows(principal)},
    )


def _render_review_tags(request: Request, principal: Principal) -> Response:
    return _render(
        request,
        "partials/review_tags.html",
        {"proposals": service.tag_proposals(principal=principal)},
    )


def _render_review_entities(request: Request, principal: Principal) -> Response:
    pairs = service.entity_merge_candidates(principal=principal)
    rows = [
        {
            "pair": pair,
            "keep_id": pair.a_id if pair.a_created_at <= pair.b_created_at else pair.b_id,
            "drop_id": pair.b_id if pair.a_created_at <= pair.b_created_at else pair.a_id,
        }
        for pair in pairs
    ]
    return _render(request, "partials/review_entities.html", {"rows": rows})


@router.get("/reviews")
def reviews_page(request: Request, principal: PrincipalDep) -> Response:
    """The review queues: proposed versions, tags and entity merges (task 48)."""
    context: dict[str, Any] = {
        "nav": "reviews",
        "can_open_files": principal.can_open_files,
        "rows": _review_version_rows(principal),
    }
    return _render(request, "reviews.html", context)


@router.get("/partials/reviews/versions")
def partial_review_versions(request: Request, principal: PrincipalDep) -> Response:
    """The proposed version relations tab."""
    return _render_review_versions(request, principal)


@router.get("/partials/reviews/tags")
def partial_review_tags(request: Request, principal: PrincipalDep) -> Response:
    """The proposed LLM tags tab."""
    return _render_review_tags(request, principal)


@router.get("/partials/reviews/entities")
def partial_review_entities(request: Request, principal: PrincipalDep) -> Response:
    """The entity merge candidates tab."""
    return _render_review_entities(request, principal)


@router.post("/reviews/entities/{keep_id}/{drop_id}/merge")
def merge_entity_pair(
    request: Request, principal: PrincipalDep, keep_id: int, drop_id: int
) -> Response:
    """Merge two candidate entities (the older survives) and re-render the tab."""
    service.merge_entities(keep_id, drop_id, principal=principal)
    return _render_review_entities(request, principal)


@router.post("/reviews/entities/{a_id}/{b_id}/distinct")
def mark_entity_pair_distinct(
    request: Request, principal: PrincipalDep, a_id: int, b_id: int
) -> Response:
    """Record two candidate entities as distinct and re-render the tab."""
    service.mark_entities_distinct(a_id, b_id, principal=principal)
    return _render_review_entities(request, principal)


def _entity_panel_context(ref: str, principal: Principal) -> dict[str, Any]:
    """The entity card, attribute values and timeline of the graph explorer."""
    context: dict[str, Any] = {
        "error": None,
        "candidates": [],
        "card": None,
        "attributes": [],
        "timeline": [],
    }
    try:
        card = service.entity_card(ref, principal=principal)
    except Ambiguous as exc:
        context["error"] = str(exc)
        context["candidates"] = list(exc.candidates)
        return context
    except NotFound as exc:
        context["error"] = str(exc)
        return context
    context["card"] = card
    context["attributes"] = service.entity_attributes(int(card["id"]), principal=principal)
    context["timeline"] = service.timeline(str(card["id"]), principal=principal)
    return context


def _render_graph_page(request: Request, principal: Principal, ref: str | None) -> Response:
    context: dict[str, Any] = {
        "nav": "graph",
        "can_open_files": principal.can_open_files,
        "q": ref or "",
        "panel": _entity_panel_context(ref, principal) if ref else None,
    }
    return _render(request, "graph.html", context)


@router.get("/graph")
def graph_page(request: Request, principal: PrincipalDep, entity: str | None = None) -> Response:
    """The graph explorer for one entity (task 48)."""
    return _render_graph_page(request, principal, entity)


@router.get("/partials/entity-panel")
def partial_entity_panel(request: Request, principal: PrincipalDep, q: str = "") -> Response:
    """The entity card + graph + timeline panel of the graph explorer.

    Only the panel's contents: HTMX swaps them into ``#entity-panel``.
    """
    context: dict[str, Any] = {
        "panel": _entity_panel_context(q, principal) if q.strip() else None,
    }
    return _render(request, "partials/entity_panel_body.html", context)


def _help_context(doc: str) -> dict[str, Any]:
    """The template context of one Help tab (task: help page)."""
    if doc not in HELP_DOCS:
        raise NotFound(f"no help topic {doc!r}")
    label, filename = HELP_DOCS[doc]
    path = DOCS_DIR / filename
    if path.is_file():
        markdown = path.read_text(encoding="utf-8")
    else:
        markdown = (
            f"# {label}\n\nThis manual (`docs/{filename}`) is not available in this "
            "installation. The documentation lives in the repository's `docs/` folder."
        )
    doc_links = {file: f"/ui/help/{key}" for key, (_label, file) in HELP_DOCS.items()}
    return {
        "nav": "help",
        "title": label,
        "current": doc,
        "tabs": [(key, label_) for key, (label_, _file) in HELP_DOCS.items()],
        "content_html": render_help_markdown(
            markdown, doc_links=doc_links, source_base_url=SOURCE_BASE_URL
        ),
    }


@router.get("/help")
def help_page(request: Request, principal: PrincipalDep) -> Response:
    """The Help page: the first tab (setup and command line)."""
    context = _help_context(next(iter(HELP_DOCS)))
    context["can_open_files"] = principal.can_open_files
    return _render(request, "help.html", context)


@router.get("/help/{doc}")
def help_topic(request: Request, principal: PrincipalDep, doc: str) -> Response:
    """One Help tab: ``web``, ``cli`` or ``mcp``."""
    context = _help_context(doc)
    context["can_open_files"] = principal.can_open_files
    return _render(request, "help.html", context)


@router.get("/status")
def status_page(request: Request, principal: PrincipalDep) -> Response:
    """The status dashboard (task 48); admins only, others get a short explanation."""
    context: dict[str, Any] = {"nav": "status", "can_open_files": principal.can_open_files}
    if not principal.is_admin:
        return _render(request, "status.html", {**context, "data": None}, status_code=403)
    return _render(request, "status.html", {**context, "data": service.status(principal=principal)})


@router.get("/partials/status")
def partial_status(request: Request, principal: PrincipalDep) -> Response:
    """The auto-refreshing status panel (``hx-trigger="every 30s"``)."""
    return _render(
        request,
        "partials/status_panel.html",
        {"data": service.status(principal=principal)},
    )
