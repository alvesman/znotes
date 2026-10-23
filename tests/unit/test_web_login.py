"""The built-in login of server mode (auth.mode token): page, cookie, redirects, sign-out."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from mynotes.api import auth
from mynotes.api.app import create_app
from mynotes.config import Settings
from mynotes.core import tokens
from mynotes.web import routes as web_routes

GOOD = "mn_good-secret"


def _info() -> tokens.TokenInfo:
    return tokens.TokenInfo(
        id=1,
        principal="alice",
        name="laptop",
        token_prefix="mn_good-",
        scopes=("web",),
        created_at=datetime.now(UTC),
        last_used_at=None,
        revoked_at=None,
    )


@pytest.fixture
def token_mode(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    settings = Settings(auth={"mode": "token"})
    monkeypatch.setattr(auth, "get_settings", lambda: settings)
    monkeypatch.setattr(web_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(tokens, "verify_token", lambda secret: _info() if secret == GOOD else None)
    monkeypatch.setattr(tokens, "resolve_prefixes", lambda user, groups: ["alice-notes"])
    return TestClient(create_app(), follow_redirects=False)


def test_ui_pages_redirect_to_login_without_a_token(token_mode: TestClient) -> None:
    response = token_mode.get("/ui/chat")
    assert response.status_code == 303
    assert response.headers["location"] == "/ui/login?next=%2Fui%2Fchat"
    api = token_mode.post("/v1/search", json={"query": "x"})
    assert api.status_code == 401  # the API answers JSON, not a redirect
    assert api.json()["detail"] == "authentication required"


def test_login_page_and_bad_token(token_mode: TestClient) -> None:
    page = token_mode.get("/ui/login?next=/ui/search")
    assert page.status_code == 200
    assert '<input type="hidden" name="next" value="/ui/search">' in page.text
    assert 'id="token"' in page.text and "Sign out" not in page.text
    bad = token_mode.post("/ui/login", data={"token": "mn_wrong", "next": "/ui/search"})
    assert bad.status_code == 200
    assert "not valid" in bad.text
    assert "set-cookie" not in bad.headers


def test_login_sets_cookie_and_signs_the_user_in(token_mode: TestClient) -> None:
    ok = token_mode.post("/ui/login", data={"token": GOOD, "next": "/ui/search"})
    assert ok.status_code == 303 and ok.headers["location"] == "/ui/search"
    cookie = ok.headers["set-cookie"]
    assert cookie.startswith("mynotes_token=mn_good-secret")
    assert "HttpOnly" in cookie and "SameSite=lax" in cookie.lower().replace(
        "samesite=lax", "SameSite=lax"
    )
    assert "Secure" not in cookie  # plain http in the test client

    token_mode.cookies.set("mynotes_token", GOOD)
    page = token_mode.get("/ui/chat")
    assert page.status_code == 200
    assert "alice" in page.text and 'action="/ui/logout"' in page.text


def test_next_outside_the_ui_is_ignored(token_mode: TestClient) -> None:
    ok = token_mode.post("/ui/login", data={"token": GOOD, "next": "https://evil.example/"})
    assert ok.headers["location"] == "/ui/chat"
    ok = token_mode.post("/ui/login", data={"token": GOOD, "next": "//evil.example"})
    assert ok.headers["location"] == "/ui/chat"


def test_logout_clears_the_cookie(token_mode: TestClient) -> None:
    token_mode.cookies.set("mynotes_token", GOOD)
    out = token_mode.post("/ui/logout")
    assert out.status_code == 303 and out.headers["location"] == "/ui/login"
    assert (
        'mynotes_token=""' in out.headers["set-cookie"] or "Max-Age=0" in out.headers["set-cookie"]
    )


def test_https_behind_a_proxy_marks_the_cookie_secure(token_mode: TestClient) -> None:
    ok = token_mode.post(
        "/ui/login",
        data={"token": GOOD, "next": "/ui/chat"},
        headers={"X-Forwarded-Proto": "https"},
    )
    assert "Secure" in ok.headers["set-cookie"]


def test_local_mode_has_no_sign_out() -> None:
    page = TestClient(create_app()).get("/ui/chat")
    assert page.status_code == 200 and 'action="/ui/logout"' not in page.text
