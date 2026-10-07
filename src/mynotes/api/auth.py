"""Pluggable authentication for the HTTP API and web UI (PLAN.md §10.2, task 52).

``auth.mode = "none"`` (the default, local mode) maps every request to
:class:`LOCAL <mynotes.core.principal.Principal>`. ``auth.mode = "oidc"``
(team mode) validates the ``Authorization: Bearer <jwt>`` against the issuer's
JWKS (OIDC discovery document, cached for ``jwks_cache_s``; the key set is
refreshed once on an unknown ``kid``), checks ``iss``/``aud``/``exp``/``nbf``
with 30 s leeway and resolves the caller's ACL prefixes from the
``acl_grant`` table - no per-request call to an external auth API.

The web UI accepts the same token from the ``mynotes_token`` cookie (set by a
reverse proxy or a login flow outside this task); with
``auth.trust_proxy_header`` the ``X-Forwarded-Access-Token`` header is also
accepted, so the service can run behind an auth proxy such as oauth2-proxy. A
proper browser login flow (authorization code + PKCE) is future work. MCP
stays stdio/local only (no auth); the streamable HTTP transport is out of
scope.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Annotated, Any

import httpx
import jwt
from fastapi import Depends, HTTPException, Request

from mynotes.config import AuthConfig, get_settings
from mynotes.core.acl import GROUP_PREFIX, resolve_prefixes
from mynotes.core.principal import LOCAL, Principal

logger = logging.getLogger(__name__)

_LEEWAY_S = 30
"""Clock leeway for ``exp``/``nbf`` validation."""

_DISCOVERY_TIMEOUT_S = 5.0
_ALGORITHMS = ("RS256", "RS384", "RS512", "ES256", "ES384", "ES512", "PS256", "PS384", "PS512")

_TOKEN_COOKIE = "mynotes_token"
_PROXY_TOKEN_HEADER = "x-forwarded-access-token"

_MISSING_CREDENTIALS = HTTPException(
    status_code=401,
    detail="authentication required",
    headers={"WWW-Authenticate": "Bearer"},
)
_INVALID_TOKEN = HTTPException(
    status_code=401,
    detail="invalid token",
    headers={"WWW-Authenticate": "Bearer"},
)

_discovery_cache: dict[str, tuple[float, str]] = {}
"""``issuer`` to ``(fetched_at, jwks_uri)``; refreshed after ``jwks_cache_s``."""

_jwks_clients: dict[str, tuple[float, jwt.PyJWKClient]] = {}
"""``issuer`` to the cached :class:`jwt.PyJWKClient` for the current JWKS URI."""


def _discover_jwks_uri(issuer: str, cache_s: int) -> str:
    """The issuer's ``jwks_uri`` from ``{issuer}/.well-known/openid-configuration``.

    Args:
        issuer: The OIDC issuer base URL.
        cache_s: How long a successful discovery response stays cached.

    Returns:
        The JWKS URL from the discovery document.

    Raises:
        HTTPException: 401 when discovery fails (the caller's token cannot be
            validated without it; internals stay in the log).
    """
    entry = _discovery_cache.get(issuer)
    now = time.monotonic()
    if entry is not None and now - entry[0] < cache_s:
        return entry[1]
    url = issuer.rstrip("/") + "/.well-known/openid-configuration"
    try:
        response = httpx.get(url, timeout=_DISCOVERY_TIMEOUT_S)
        response.raise_for_status()
        jwks_uri = str(response.json()["jwks_uri"])
    except Exception as exc:
        logger.warning("OIDC discovery for %s failed: %s", issuer, type(exc).__name__)
        raise _INVALID_TOKEN from exc
    _discovery_cache[issuer] = (now, jwks_uri)
    return jwks_uri


def _jwks_client(issuer: str, cache_s: int) -> jwt.PyJWKClient:
    """The JWKS client for one issuer (seam for tests).

    The client is rebuilt after ``cache_s`` so a rotated ``jwks_uri`` is
    picked up; PyJWKClient itself caches the key set for ``cache_s`` and
    refreshes it once when a token carries an unknown ``kid``.

    Args:
        issuer: The OIDC issuer base URL.
        cache_s: Cache lifetime for the discovery document and key set.

    Returns:
        A :class:`jwt.PyJWKClient` bound to the issuer's JWKS URL.
    """
    entry = _jwks_clients.get(issuer)
    now = time.monotonic()
    if entry is None or now - entry[0] >= cache_s:
        entry = (now, jwt.PyJWKClient(_discover_jwks_uri(issuer, cache_s), lifespan=cache_s))
        _jwks_clients[issuer] = entry
    return entry[1]


def _decode_claims(token: str, auth: AuthConfig) -> dict[str, Any]:
    """Validate the token signature and time/audience/issuer claims.

    Args:
        token: The raw bearer token.
        auth: The auth settings (issuer, audience, JWKS cache time).

    Returns:
        The validated claim set.

    Raises:
        HTTPException: 401 when the signature, a required claim or ``iss``/
            ``aud``/``exp``/``nbf`` (30 s leeway) fails.
    """
    try:
        key = (
            _jwks_client(auth.oidc.issuer or "", auth.oidc.jwks_cache_s)
            .get_signing_key_from_jwt(token)
            .key
        )
    except jwt.PyJWKClientError as exc:
        logger.warning("JWKS lookup failed: %s", exc)
        raise _INVALID_TOKEN from exc
    try:
        return jwt.decode(
            token,
            key=key,
            algorithms=list(_ALGORITHMS),
            audience=auth.oidc.audience,
            issuer=auth.oidc.issuer,
            leeway=_LEEWAY_S,
            options={"require": ["exp"]},
        )
    except jwt.PyJWTError as exc:
        logger.info("JWT validation failed: %s", type(exc).__name__)
        raise _INVALID_TOKEN from exc


def _is_admin(user: str, groups: Sequence[str], admin_principals: Sequence[str]) -> bool:
    """Whether the principal is in ``auth.admin_principals`` (user or group entry)."""
    return user in admin_principals or any(
        f"{GROUP_PREFIX}{group}" in admin_principals for group in groups
    )


def principal_from_claims(claims: dict[str, Any], auth: AuthConfig) -> Principal:
    """Build the team-mode principal from validated OIDC claims.

    Args:
        claims: The validated JWT claim set.
        auth: The auth settings (claim names, admin principals).

    Returns:
        The principal with its resolved ACL prefixes; file access is never
        allowed in team mode.

    Raises:
        HTTPException: 401 when the user claim is missing or not a string.
    """
    user = claims.get(auth.oidc.user_claim)
    if not isinstance(user, str) or not user:
        logger.warning("token is missing the %r user claim", auth.oidc.user_claim)
        raise _INVALID_TOKEN
    raw_groups = claims.get(auth.oidc.groups_claim, [])
    groups = [str(group) for group in raw_groups] if isinstance(raw_groups, list) else []
    return Principal(
        user=user,
        acl_prefixes=resolve_prefixes(user, groups),
        can_open_files=False,
        is_admin=_is_admin(user, groups, auth.admin_principals),
    )


def _extract_token(request: Request, auth: AuthConfig) -> str | None:
    """The bearer token from the header, the auth-proxy header or the UI cookie."""
    header = request.headers.get("Authorization")
    if header and header.lower().startswith("bearer "):
        token = header[7:].strip()
        return token or None
    if auth.trust_proxy_header:
        forwarded = request.headers.get(_PROXY_TOKEN_HEADER)
        if forwarded and forwarded.strip():
            return forwarded.strip()
    cookie = request.cookies.get(_TOKEN_COOKIE)
    return cookie if cookie and cookie.strip() else None


def get_principal(request: Request) -> Principal:
    """Resolve the caller's principal from the request.

    Args:
        request: The incoming request (bearer header, optional proxy header
            and ``mynotes_token`` cookie).

    Returns:
        ``LOCAL`` in ``none`` mode; the OIDC principal in ``oidc`` mode.

    Raises:
        HTTPException: 401 when no credentials or an invalid token is
            presented in ``oidc`` mode.
    """
    auth = get_settings().auth
    if auth.mode == "none":
        return LOCAL
    token = _extract_token(request, auth)
    if not token:
        raise _MISSING_CREDENTIALS
    return principal_from_claims(_decode_claims(token, auth), auth)


PrincipalDep = Annotated[Principal, Depends(get_principal)]
"""FastAPI dependency alias so every route signature stays short."""
