"""Unit tests for personal tokens: secrets, hashing and the principal they act as."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from mynotes.api import auth
from mynotes.config import Settings
from mynotes.core import tokens
from mynotes.core.principal import LOCAL


def _request(
    headers: dict[str, str] | None = None, cookies: dict[str, str] | None = None
) -> Request:
    raw_headers = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    if cookies:
        raw_headers.append((b"cookie", "; ".join(f"{k}={v}" for k, v in cookies.items()).encode()))
    return Request({"type": "http", "method": "GET", "path": "/", "headers": raw_headers})


def _info(scopes: tuple[str, ...], principal: str = "alice") -> tokens.TokenInfo:
    return tokens.TokenInfo(
        id=1,
        principal=principal,
        name="laptop",
        token_prefix="mn_abc",
        scopes=scopes,
        created_at=datetime.now(UTC),
        last_used_at=None,
        revoked_at=None,
    )


def test_secret_shape_and_hash() -> None:
    secret = tokens.generate_secret()
    assert secret.startswith("mn_") and len(secret) >= 40
    assert tokens.looks_like_token(secret) and not tokens.looks_like_token("eyJhbGciOi")
    assert tokens.hash_secret(secret) == tokens.hash_secret(secret)
    assert tokens.hash_secret(secret) != tokens.hash_secret(tokens.generate_secret())


def test_principal_for_token_scopes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tokens, "resolve_prefixes", lambda user, groups: ["alice-notes"])
    agent = tokens.principal_for(_info(("ingest:alice-notes",)))
    assert agent.user == "alice" and agent.acl_prefixes == []  # an agent token reads nothing
    assert agent.ingest_roots == frozenset({"alice-notes"})
    assert agent.can_open_files is False and agent.is_admin is False
    web = tokens.principal_for(_info(("ingest:alice-notes", "web")))
    assert web.acl_prefixes == ["alice-notes"] and web.ingest_roots == frozenset({"alice-notes"})
    admin = tokens.principal_for(_info(("admin",)))
    assert admin.is_admin is True and admin.acl_prefixes == ["alice-notes"]
    assert admin.ingest_roots == frozenset()
    assert not tokens.can_read(_info(("ingest:alice-notes",)))
    assert tokens.can_read(_info(("web",))) and tokens.can_read(_info(("admin",)))


@pytest.mark.parametrize("mode", ["none", "token", "oidc"])
def test_valid_token_is_accepted_in_every_mode(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    auth_settings = {"mode": mode}
    if mode == "oidc":
        auth_settings["oidc"] = {"issuer": "https://issuer.example", "audience": "mynotes"}
    monkeypatch.setattr(auth, "get_settings", lambda: Settings(auth=auth_settings))
    monkeypatch.setattr(tokens, "verify_token", lambda secret: _info(("web",)))
    monkeypatch.setattr(tokens, "resolve_prefixes", lambda user, groups: [])
    principal = auth.get_principal(_request({"Authorization": "Bearer mn_secret"}))
    assert principal.user == "alice" and principal.acl_prefixes == []
    principal = auth.get_principal(_request(cookies={"mynotes_token": "mn_secret"}))
    assert principal.user == "alice"


def test_unknown_or_revoked_token_is_rejected_even_in_none_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth, "get_settings", lambda: Settings())
    monkeypatch.setattr(tokens, "verify_token", lambda secret: None)
    with pytest.raises(HTTPException) as exc:
        auth.get_principal(_request({"Authorization": "Bearer mn_bad"}))
    assert exc.value.status_code == 401
    assert auth.get_principal(_request()) is LOCAL  # no credentials: local mode as before


def test_token_mode_requires_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth, "get_settings", lambda: Settings(auth={"mode": "token"}))
    with pytest.raises(HTTPException) as missing:
        auth.get_principal(_request())
    assert missing.value.status_code == 401
    with pytest.raises(HTTPException) as jwt_like:
        auth.get_principal(_request({"Authorization": "Bearer eyJhbGciOi.payload.sig"}))
    assert jwt_like.value.status_code == 401
