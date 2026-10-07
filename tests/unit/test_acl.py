"""Unit tests for team-mode auth and path ACL (task 52).

``resolve_prefixes`` runs against a fake ``acl_grant`` table (the SQL union
itself is covered by the integration leakage tests), JWT validation uses a
locally generated RSA key with a stubbed JWKS client, and the ACL filter
mirror is checked in Python only. No network, Docker or model endpoint.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from jwt.algorithms import RSAAlgorithm
from starlette.requests import Request

from mynotes.api.auth import get_principal, principal_from_claims
from mynotes.config import Settings
from mynotes.core.acl import clear_cache, normalize_prefix, path_allowed, resolve_prefixes
from mynotes.retrieval.hybrid import build_filter_clauses, hit_matches_filters
from mynotes.retrieval.types import SearchFilters, SearchHit

ISSUER = "https://keycloak.example/realms/test"
AUDIENCE = "mynotes"


def auth_settings(**overrides: Any) -> Settings:
    """Settings in oidc mode with the test issuer/audience."""
    oidc: dict[str, Any] = {"issuer": ISSUER, "audience": AUDIENCE, "jwks_cache_s": 3600}
    oidc.update(overrides.pop("oidc", {}))
    auth: dict[str, Any] = {"mode": "oidc", "oidc": oidc}
    auth.update(overrides)
    return Settings(auth=auth)


# ---------------------------------------------------------------------------
# resolve_prefixes (fake acl_grant table)
# ---------------------------------------------------------------------------


class FakeGrantTable:
    """Stands in for the ``acl_grant`` rows keyed by principal."""

    def __init__(self, rows: dict[str, list[str]]) -> None:
        self.rows = rows
        self.queries: list[list[str]] = []

    def __call__(self, principals: Sequence[str]) -> list[str]:
        self.queries.append(list(principals))
        found: set[str] = set()
        for principal in principals:
            found.update(self.rows.get(principal, []))
        return sorted(found)


@pytest.fixture(autouse=True)
def _clean_cache() -> Any:
    clear_cache()
    yield
    clear_cache()


def test_resolve_prefixes_unions_user_and_group_grants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = FakeGrantTable({"alice": ["teamA"], "group:dev": ["teamB", "teamA"]})
    monkeypatch.setattr("mynotes.core.acl.grant_prefixes", table)

    prefixes = resolve_prefixes("alice", ["dev"])

    assert prefixes == ["teamA", "teamB"]
    assert table.queries == [["alice", "group:dev"]]


def test_resolve_prefixes_empty_prefix_means_whole_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = FakeGrantTable({"root": [""], "group:all": [""]})
    monkeypatch.setattr("mynotes.core.acl.grant_prefixes", table)

    assert resolve_prefixes("root", []) == [""]
    assert resolve_prefixes("nobody", ["all"]) == [""]
    assert path_allowed("any/where.md", [""])


def test_resolve_prefixes_without_grants_means_no_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = FakeGrantTable({})
    monkeypatch.setattr("mynotes.core.acl.grant_prefixes", table)

    assert resolve_prefixes("alice", ["dev"]) == []


def test_resolve_prefixes_cache_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    rows: dict[str, list[str]] = {"alice": ["teamA"]}
    table = FakeGrantTable(rows)
    monkeypatch.setattr("mynotes.core.acl.grant_prefixes", table)
    now = 100.0
    monkeypatch.setattr("mynotes.core.acl._now", lambda: now)

    assert resolve_prefixes("alice", []) == ["teamA"]
    rows["alice"] = ["teamB"]
    assert resolve_prefixes("alice", []) == ["teamA"], "cached value must survive"

    now += 61.0
    assert resolve_prefixes("alice", []) == ["teamB"], "the cache must expire after 60 s"
    assert len(table.queries) == 2


def test_resolve_prefixes_is_cached_per_user_and_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = FakeGrantTable({"alice": ["teamA"], "group:dev": ["teamB"]})
    monkeypatch.setattr("mynotes.core.acl.grant_prefixes", table)
    monkeypatch.setattr("mynotes.core.acl._now", lambda: 0.0)

    assert resolve_prefixes("alice", ["dev"]) == ["teamA", "teamB"]
    assert resolve_prefixes("alice", ["dev"]) == ["teamA", "teamB"]
    assert len(table.queries) == 1, "the 60 s cache must avoid re-querying"


def test_normalize_prefix_strips_trailing_slashes_and_rejects_escapes() -> None:
    assert normalize_prefix(" teamA/ ") == "teamA"
    assert normalize_prefix("") == ""
    assert normalize_prefix("a//b/") == "a/b"
    with pytest.raises(ValueError):
        normalize_prefix("/absolute")
    with pytest.raises(ValueError):
        normalize_prefix("../escape")
    with pytest.raises(ValueError):
        normalize_prefix("a/../b")


# ---------------------------------------------------------------------------
# JWT validation (locally generated RSA key, stubbed JWKS client)
# ---------------------------------------------------------------------------


def _keypair() -> dict[str, Any]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk: dict[str, Any] = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    jwk["kid"] = "test-key"
    jwk["alg"] = "RS256"
    jwk["use"] = "sig"
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return {"pem": pem, "jwk": jwk}


class StubJwksClient:
    """Duck-typed :class:`jwt.PyJWKClient` serving one static key."""

    def __init__(self, jwk: dict[str, Any]) -> None:
        self.jwk = jwt.PyJWK(jwk)

    def get_signing_key_from_jwt(self, token: str) -> Any:
        return SimpleNamespace(key=self.jwk)


def _token(
    key: dict[str, Any],
    claims: dict[str, Any],
    *,
    kid: str = "test-key",
) -> str:
    return jwt.encode(claims, key["pem"], algorithm="RS256", headers={"kid": kid})


def _claims(**overrides: Any) -> dict[str, Any]:
    now = datetime.now(UTC)
    base: dict[str, Any] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "alice",
        "preferred_username": "alice",
        "groups": ["teamA-members"],
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=1)).timestamp()),
    }
    base.update(overrides)
    return base


def _request(
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
) -> Request:
    raw_headers = [(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()]
    if cookies:
        raw_headers.append(
            (b"cookie", "; ".join(f"{name}={value}" for name, value in cookies.items()).encode())
        )
    return Request({"type": "http", "method": "GET", "path": "/", "headers": raw_headers})


@pytest.fixture
def rsa_key(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    key = _keypair()
    monkeypatch.setattr(
        "mynotes.api.auth._jwks_client",
        lambda issuer, cache_s: StubJwksClient(key["jwk"]),
    )
    return key


@pytest.fixture
def _patched_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    settings = auth_settings(admin_principals=["admin", "group:sec"])
    monkeypatch.setattr("mynotes.api.auth.get_settings", lambda: settings)
    monkeypatch.setattr(
        "mynotes.api.auth.resolve_prefixes",
        lambda user, groups: ["teamA"] if "teamA-members" in groups else [],
    )
    return settings


def test_valid_token_becomes_a_restricted_principal(
    rsa_key: dict[str, Any], _patched_settings: Settings
) -> None:
    principal = get_principal(_request({"Authorization": f"Bearer {_token(rsa_key, _claims())}"}))

    assert principal.user == "alice"
    assert principal.acl_prefixes == ["teamA"]
    assert principal.can_open_files is False
    assert principal.is_admin is False


def test_expired_token_is_rejected(rsa_key: dict[str, Any], _patched_settings: Settings) -> None:
    expired = _claims(exp=int((datetime.now(UTC) - timedelta(seconds=120)).timestamp()))
    with pytest.raises(HTTPException) as excinfo:
        get_principal(_request({"Authorization": f"Bearer {_token(rsa_key, expired)}"}))
    assert excinfo.value.status_code == 401


def test_wrong_audience_is_rejected(rsa_key: dict[str, Any], _patched_settings: Settings) -> None:
    with pytest.raises(HTTPException) as excinfo:
        get_principal(
            _request(
                {
                    "Authorization": f"Bearer {_token(rsa_key, _claims(aud='another-client'))}",
                }
            )
        )
    assert excinfo.value.status_code == 401


def test_future_nbf_is_rejected(rsa_key: dict[str, Any], _patched_settings: Settings) -> None:
    future = int((datetime.now(UTC) + timedelta(seconds=300)).timestamp())
    with pytest.raises(HTTPException) as excinfo:
        get_principal(_request({"Authorization": f"Bearer {_token(rsa_key, _claims(nbf=future))}"}))
    assert excinfo.value.status_code == 401


def test_wrong_issuer_is_rejected(rsa_key: dict[str, Any], _patched_settings: Settings) -> None:
    with pytest.raises(HTTPException) as excinfo:
        get_principal(
            _request(
                {"Authorization": f"Bearer {_token(rsa_key, _claims(iss='https://evil.example'))}"}
            )
        )
    assert excinfo.value.status_code == 401


def test_missing_header_is_rejected(rsa_key: dict[str, Any], _patched_settings: Settings) -> None:
    with pytest.raises(HTTPException) as excinfo:
        get_principal(_request())
    assert excinfo.value.status_code == 401
    assert excinfo.value.headers is not None and "WWW-Authenticate" in excinfo.value.headers


def test_cookie_token_is_accepted_for_the_web_ui(
    rsa_key: dict[str, Any], _patched_settings: Settings
) -> None:
    principal = get_principal(_request(cookies={"mynotes_token": _token(rsa_key, _claims())}))

    assert principal.user == "alice"


def test_proxy_header_needs_trust_proxy_header(
    rsa_key: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("mynotes.api.auth.get_settings", lambda: auth_settings())
    monkeypatch.setattr("mynotes.api.auth.resolve_prefixes", lambda user, groups: [])
    token = _token(rsa_key, _claims())
    with pytest.raises(HTTPException) as excinfo:
        get_principal(_request({"X-Forwarded-Access-Token": token}))
    assert excinfo.value.status_code == 401

    trusted = auth_settings(trust_proxy_header=True)
    monkeypatch.setattr("mynotes.api.auth.get_settings", lambda: trusted)
    principal = get_principal(_request({"X-Forwarded-Access-Token": token}))
    assert principal.user == "alice"


def test_bearer_header_wins_over_cookie(
    rsa_key: dict[str, Any], _patched_settings: Settings
) -> None:
    other = _token(rsa_key, _claims(preferred_username="bob"))
    principal = get_principal(
        _request(
            headers={"Authorization": f"Bearer {_token(rsa_key, _claims())}"},
            cookies={"mynotes_token": other},
        )
    )
    assert principal.user == "alice"


def test_admin_principals_cover_users_and_groups(
    rsa_key: dict[str, Any], _patched_settings: Settings
) -> None:
    admin = principal_from_claims(
        _claims(preferred_username="admin", groups=[]), _patched_settings.auth
    )
    member = principal_from_claims(
        _claims(preferred_username="bob", groups=["sec"]), _patched_settings.auth
    )
    other = principal_from_claims(
        _claims(preferred_username="carol", groups=["dev"]), _patched_settings.auth
    )

    assert admin.is_admin is True
    assert member.is_admin is True
    assert other.is_admin is False


def test_missing_user_claim_is_rejected(
    rsa_key: dict[str, Any], _patched_settings: Settings
) -> None:
    with pytest.raises(HTTPException) as excinfo:
        principal_from_claims(_claims(preferred_username=None), _patched_settings.auth)
    assert excinfo.value.status_code == 401


# ---------------------------------------------------------------------------
# ACL filter mirror (exact, case-sensitive, empty-prefix semantics)
# ---------------------------------------------------------------------------


def _hit(path: str, **overrides: Any) -> SearchHit:
    defaults: dict[str, Any] = {
        "chunk_id": "00000000-0000-0000-0000-000000000000",
        "document_id": 1,
        "version_id": 1,
        "rank": 1,
        "score": 0.01,
        "text": "text",
        "title": "t",
        "section": None,
        "heading_path": "",
        "path": path,
        "page": None,
        "slide": None,
        "doc_date": None,
        "doc_status": "current",
        "is_latest": True,
        "is_table": False,
        "format": "md",
        "language": None,
        "tags": [],
    }
    defaults.update(overrides)
    return SearchHit(**defaults)


def test_hit_matches_filters_prefix_is_exact_and_case_sensitive() -> None:
    with_team_a = SearchFilters(acl_prefixes=["teamA"])

    assert hit_matches_filters(_hit("teamA/x.md"), with_team_a)
    assert hit_matches_filters(_hit("teamA/sub/x.md"), with_team_a)
    assert not hit_matches_filters(_hit("teamB/x.md"), with_team_a), "other folder"
    assert not hit_matches_filters(_hit("teama/x.md"), with_team_a), "case-sensitive"
    assert not hit_matches_filters(_hit("teamA2/x.md"), with_team_a), "prefix must not extend"
    assert not hit_matches_filters(_hit("x.md"), with_team_a), "root files are outside folders"

    assert not hit_matches_filters(_hit("teamA/x.md"), SearchFilters(acl_prefixes=[]))
    assert hit_matches_filters(_hit("teamB/x.md"), SearchFilters(acl_prefixes=None))
    assert hit_matches_filters(_hit("teamB/x.md"), SearchFilters(acl_prefixes=[""]))


def test_acl_clause_full_workspace_grant_matches_everything() -> None:
    clauses = build_filter_clauses(SearchFilters(acl_prefixes=[""]))
    assert {"match_all": {}} in clauses
    empty = build_filter_clauses(SearchFilters(acl_prefixes=[]))
    assert {"bool": {"must_not": {"match_all": {}}}} in empty


def test_path_allowed_rules() -> None:
    assert path_allowed("teamA/x.md", ["teamA"])
    assert not path_allowed("", ["teamA"])
    assert not path_allowed("teamB/x.md", ["teamA"])
    assert not path_allowed("teamB/x.md", []), "no grants means no access"
