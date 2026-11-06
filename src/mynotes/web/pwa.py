"""The web app manifest and the service worker: the web UI as an installable app.

Both routes sit at the origin root and need no sign-in, like ``/favicon.ico``,
so the app can be installed before signing in. The service worker caches only
``/static/`` assets and answers a failed page navigation with the offline
notice; pages (per user, behind the sign-in cookie), the API and the htmx
partials are never cached. Its cache name carries the release version and a
hash of the precached URLs (which embed the assets' ``?v=`` stamps), so any
changed asset means a different ``/sw.js``, a new cache and the old one gone.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from mynotes import __version__
from mynotes.web.routes import STATIC_DIR, asset

router = APIRouter()

MANIFEST: dict[str, Any] = {
    "id": "/ui/chat",
    "name": "mynotes",
    "short_name": "mynotes",
    "description": "Ask questions about your notes.",
    "start_url": "/ui/chat",
    "scope": "/",
    "display": "standalone",
    "background_color": "#fbfbfa",
    "theme_color": "#0b5cb8",
    "icons": [
        {"src": "/static/icon-192.png", "sizes": "192x192", "type": "image/png"},
        {"src": "/static/icon-512.png", "sizes": "512x512", "type": "image/png"},
        {
            "src": "/static/icon-maskable-512.png",
            "sizes": "512x512",
            "type": "image/png",
            "purpose": "maskable",
        },
        {"src": "/static/icon.svg", "sizes": "any", "type": "image/svg+xml"},
    ],
}
"""The web app manifest (``/manifest.webmanifest``)."""

OFFLINE_URLS = ("/static/offline.html", "/static/offline.css")
"""The offline notice and its stylesheet, precached under the bare URLs the page links."""

STAMPED_ASSETS = ("app.css", "theme.js", "ui.js", "favicon.ico", "icon-192.png")
"""Shell assets precached under their ``asset()`` URL (with the ``?v=`` stamp)."""

HTMX_SRC = "/static/htmx-2.0.11.min.js"
"""htmx is referenced by its fixed path in ``base.html``."""

BYPASS_PREFIXES = ("/v1/", "/ui/partials/")
"""Paths the worker never touches: the API (the chat stream included) and the htmx partials."""


def precache_urls() -> list[str]:
    """The URLs the service worker stores at install, the offline notice first."""
    return [*OFFLINE_URLS, *(asset(name) for name in STAMPED_ASSETS), HTMX_SRC]


def cache_name(urls: list[str]) -> str:
    """``mynotes-<version>-<hash of the precache list>``: new on any changed asset."""
    digest = hashlib.sha1(json.dumps(urls).encode("utf-8")).hexdigest()[:8]
    return f"mynotes-{__version__}-{digest}"


@router.get("/manifest.webmanifest", include_in_schema=False)
def manifest() -> JSONResponse:
    """The web app manifest: name, start page, colours and icons."""
    return JSONResponse(MANIFEST, media_type="application/manifest+json")


@router.get("/sw.js", include_in_schema=False)
def service_worker() -> Response:
    """The service worker, with the cache name and the precache list filled in."""
    urls = precache_urls()
    source = (STATIC_DIR / "sw.js").read_text(encoding="utf-8")
    body = source.replace("__CACHE_NAME__", cache_name(urls)).replace(
        "__PRECACHE__", json.dumps(urls)
    )
    return Response(
        body,
        media_type="text/javascript",
        headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"},
    )
