"""The FastAPI application factory (PLAN.md §10.2).

``create_app()`` wires the routers under ``/v1``, the web UI under ``/ui``
(with ``/static`` for its assets, ``/favicon.ico``, and ``/`` redirecting to the chat page) plus
``/healthz``, and maps the facade's errors to HTTP responses. There is no CORS
by default: the web UI is served from the same origin (task 47). Bind
``mynotes serve`` to localhost unless an auth mode other than ``none`` is
configured.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from mynotes import __version__
from mynotes.api.routes import admin, ask, documents, files, graph, search, tags, versions
from mynotes.core.errors import Ambiguous, Forbidden, NotFound
from mynotes.llm.errors import EndpointUnavailable
from mynotes.web import routes as web_routes

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Build the FastAPI app with all routers and error handlers.

    Returns:
        The configured :class:`FastAPI` application.
    """
    from mynotes.jobs.endpoint_stats import set_process_role

    set_process_role("web")
    app = FastAPI(title="mynotes", version=__version__)
    app.include_router(ask.router)
    app.include_router(search.router)
    app.include_router(documents.router)
    app.include_router(files.router)
    app.include_router(tags.router)
    app.include_router(versions.router)
    app.include_router(graph.router)
    app.include_router(admin.router)
    app.include_router(web_routes.router)
    app.mount("/static", StaticFiles(directory=web_routes.STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        """Send browsers to the chat page."""
        return RedirectResponse("/ui/chat")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> FileResponse:
        """The site icon at the address browsers request by default."""
        return FileResponse(web_routes.STATIC_DIR / "favicon.ico", media_type="image/x-icon")

    _register_error_handlers(app)
    return app


def _error_body(error: str, detail: str, candidates: list[dict[str, Any]] | None = None) -> dict:
    """The JSON error body (never includes tracebacks or internals)."""
    body: dict[str, Any] = {"error": error, "detail": detail}
    if candidates is not None:
        body["candidates"] = candidates
    return body


def _register_error_handlers(app: FastAPI) -> None:
    """Map facade errors to status codes; unexpected errors stay generic."""

    @app.exception_handler(NotFound)
    async def _not_found(request: Request, exc: NotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content=_error_body("not_found", str(exc)))

    @app.exception_handler(Ambiguous)
    async def _ambiguous(request: Request, exc: Ambiguous) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content=_error_body("ambiguous", str(exc), list(exc.candidates)),
        )

    @app.exception_handler(Forbidden)
    async def _forbidden(request: Request, exc: Forbidden) -> JSONResponse:
        return JSONResponse(status_code=403, content=_error_body("forbidden", str(exc)))

    @app.exception_handler(EndpointUnavailable)
    async def _endpoint_unavailable(request: Request, exc: EndpointUnavailable) -> JSONResponse:
        logger.warning("endpoint unavailable for %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=503,
            content=_error_body("endpoint_unavailable", "a model endpoint is unavailable"),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error for %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content=_error_body("internal_error", "an unexpected error occurred"),
        )
