"""The web UI as an installable app: manifest, icons, service worker, offline notice."""

from __future__ import annotations

import json
import re
import struct

from fastapi.testclient import TestClient

from mynotes import __version__
from mynotes.api.app import create_app
from mynotes.web import pwa

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _client() -> TestClient:
    return TestClient(create_app())


def _png_size(data: bytes) -> tuple[int, int]:
    assert data.startswith(PNG_MAGIC)
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def test_manifest_describes_an_installable_app() -> None:
    client = _client()
    response = client.get("/manifest.webmanifest")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/manifest+json")
    manifest = response.json()
    assert manifest["name"] == "mynotes" and manifest["short_name"] == "mynotes"
    assert manifest["start_url"] == "/ui/chat" and manifest["id"] == "/ui/chat"
    assert manifest["scope"] == "/" and manifest["display"] == "standalone"
    assert manifest["background_color"] == "#fbfbfa" and manifest["theme_color"] == "#0b5cb8"
    maskable = [icon for icon in manifest["icons"] if icon.get("purpose") == "maskable"]
    assert len(maskable) == 1
    for icon in manifest["icons"]:
        served = client.get(icon["src"])
        assert served.status_code == 200, icon["src"]
        if icon["type"] == "image/png":
            size = int(icon["sizes"].split("x")[0])
            assert _png_size(served.content) == (size, size), icon["src"]
        else:
            assert icon["type"] == "image/svg+xml"
            assert served.headers["content-type"].startswith("image/svg+xml")
            assert b"<svg" in served.content


def test_apple_touch_icon_is_an_opaque_180px_png() -> None:
    data = _client().get("/static/apple-touch-icon.png").content
    assert _png_size(data) == (180, 180)
    assert data[25] == 2  # colour type 2: RGB without alpha (iOS paints black behind alpha)


def test_service_worker_is_served_with_its_cache_name_and_precache_list() -> None:
    client = _client()
    response = client.get("/sw.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert response.headers["service-worker-allowed"] == "/"
    assert "no-cache" in response.headers["cache-control"]
    source = response.text
    assert "__CACHE_NAME__" not in source and "__PRECACHE__" not in source
    assert f'var CACHE = "mynotes-{__version__}-' in source
    match = re.search(r"var PRECACHE = (\[.*?\]);", source)
    assert match is not None
    urls = json.loads(match.group(1))
    assert urls[0] == "/static/offline.html" and "/static/offline.css" in urls
    assert any(url.startswith("/static/app.css?v=") for url in urls)
    assert pwa.HTMX_SRC in urls
    for url in urls:
        assert client.get(url).status_code == 200, url
    assert all(url.startswith("/static/") for url in urls)  # never a page, never the API


def test_service_worker_cache_name_follows_the_assets() -> None:
    urls = pwa.precache_urls()
    assert pwa.cache_name(urls) == pwa.cache_name(list(urls))
    assert pwa.cache_name(urls) != pwa.cache_name([*urls, "/static/other.js?v=1"])


def test_service_worker_source_states_the_rules() -> None:
    source = _client().get("/sw.js").text
    assert 'var BYPASS = ["/v1/", "/ui/partials/"];' in source
    assert 'request.method !== "GET"' in source
    assert 'request.mode === "navigate"' in source
    assert "skipWaiting" in source and "clients.claim" in source
    assert "/ui/chat" not in source and "/ui/login" not in source  # pages are never stored


def test_offline_notice_is_plain_static_html() -> None:
    client = _client()
    page = client.get("/static/offline.html")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    assert "You are offline" in page.text and 'href="/ui/chat"' in page.text
    assert "<style" not in page.text and "<script" not in page.text and ' style="' not in page.text
    assert 'href="/static/offline.css"' in page.text
    css = client.get("/static/offline.css")
    assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")
    assert "prefers-color-scheme: dark" in css.text


def test_pages_link_the_manifest_icons_and_theme_colours() -> None:
    html = _client().get("/ui/help").text
    head = html[: html.index("</head>")]
    assert '<link rel="manifest" href="/manifest.webmanifest">' in head
    assert '<link rel="apple-touch-icon" href="/static/apple-touch-icon.png">' in head
    assert (
        '<meta name="theme-color" media="(prefers-color-scheme: light)" content="#fbfbfa">' in head
    )
    assert (
        '<meta name="theme-color" media="(prefers-color-scheme: dark)" content="#1b1b1a">' in head
    )
    assert '<link rel="icon" href="/static/favicon.ico"' in head  # the favicon stays


def test_ui_script_registers_the_worker_in_secure_contexts_only() -> None:
    script = _client().get("/static/ui.js").text
    assert 'navigator.serviceWorker.register("/sw.js")' in script
    assert "isSecureContext" in script
