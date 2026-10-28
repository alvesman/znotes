"""Personal API tokens for agents and web users (server mode).

A token is ``mn_`` plus 43 URL-safe characters, shown once at creation; only
its sha256 is stored (``api_token.token_hash``). Scopes decide what it may do:
``ingest:<root>`` feeds one remote root (an agent; the root is created for
the token's user when it does not exist yet), ``web`` gives a person read and
review access to what their ACL grants allow, ``admin`` the admin actions.
The principal built from a token can never open files in OS apps.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update

from mynotes.config import get_settings
from mynotes.core.acl import resolve_prefixes
from mynotes.core.errors import NotFound
from mynotes.core.principal import Principal
from mynotes.db.models import ApiToken, SourceRoot, Workspace
from mynotes.db.session import session_scope
from mynotes.tags.store import workspace_id_by_name

logger = logging.getLogger(__name__)

TOKEN_PREFIX = "mn_"
INGEST_SCOPE = "ingest:"
WEB_SCOPE = "web"
ADMIN_SCOPE = "admin"
_LAST_USED_GRANULARITY = timedelta(seconds=60)


@dataclass(frozen=True)
class TokenInfo:
    """One token as listed; never carries the secret."""

    id: int
    principal: str
    name: str
    token_prefix: str
    scopes: tuple[str, ...]
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None
    created_roots: tuple[str, ...] = ()
    """Remote roots ``create_token`` created for the ``ingest:`` scopes (not stored)."""


def looks_like_token(value: str) -> bool:
    """Whether a bearer value is one of ours (as opposed to an OIDC JWT)."""
    return value.startswith(TOKEN_PREFIX)


def hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def generate_secret() -> str:
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


def _workspace_id() -> int:
    with session_scope() as session:
        workspace_id = workspace_id_by_name(session, get_settings().workspace)
    if workspace_id is None:
        raise NotFound(f"workspace {get_settings().workspace!r} does not exist; run 'mynotes init'")
    return int(workspace_id)


def create_token(principal: str, name: str, scopes: Sequence[str]) -> tuple[str, TokenInfo]:
    """Create a token; returns ``(secret, info)``. The secret is never stored.

    Raises:
        ValueError: On an empty principal or name, no scopes, an unknown
            scope, or an ``ingest:`` scope naming a root that is not a remote
            root of the workspace.
        NotFound: When the workspace does not exist.
    """
    principal = principal.strip()
    name = name.strip()
    if not principal or not name:
        raise ValueError("a user and a token name are required")
    cleaned = list(dict.fromkeys(scope.strip() for scope in scopes if scope.strip()))
    if not cleaned:
        raise ValueError("at least one scope is required (ingest:<root>, web or admin)")
    workspace_id = _workspace_id()
    created_roots: list[str] = []
    for scope in cleaned:
        if scope in (WEB_SCOPE, ADMIN_SCOPE):
            continue
        if not scope.startswith(INGEST_SCOPE):
            raise ValueError(f"unknown scope {scope!r}")
        if _ensure_remote_root(workspace_id, scope[len(INGEST_SCOPE) :], principal):
            created_roots.append(scope[len(INGEST_SCOPE) :])
    with session_scope() as session:
        secret = generate_secret()
        row = ApiToken(
            workspace_id=workspace_id,
            principal=principal,
            name=name,
            token_prefix=secret[: len(TOKEN_PREFIX) + 6],
            token_hash=hash_secret(secret),
            scopes=cleaned,
        )
        session.add(row)
        session.flush()
        info = _info(row)
    logger.info("created token %s (%s) for %s with scopes %s", info.id, name, principal, cleaned)
    return secret, replace(info, created_roots=tuple(created_roots))


def _ensure_remote_root(workspace_id: int, root_name: str, owner: str) -> bool:
    """A remote root for an ``ingest:`` scope, created for ``owner`` when it does not exist.

    Returns:
        True when the root was created now.

    Raises:
        ValueError: When the name belongs to a local root, or to a remote root
            owned by someone else.
    """
    from mynotes.core import roots

    with session_scope() as session:
        row = session.execute(
            select(SourceRoot.kind, SourceRoot.owner).where(
                SourceRoot.workspace_id == workspace_id, SourceRoot.name == root_name
            )
        ).first()
    if row is None:
        roots.add_remote_root(root_name, owner)
        return True
    kind, root_owner = row
    if kind != "remote":
        raise ValueError(f"{root_name!r} is a local folder of this server, not a root agents feed")
    if root_owner and root_owner != owner:
        raise ValueError(
            f"root {root_name!r} belongs to {root_owner}; a token for {owner} cannot feed it"
        )
    return False


def verify_token(secret: str) -> TokenInfo | None:
    """The token record for a secret, or None when unknown, revoked or of another workspace."""
    if not looks_like_token(secret):
        return None
    digest = hash_secret(secret)
    with session_scope() as session:
        row = session.execute(
            select(ApiToken)
            .join(Workspace, Workspace.id == ApiToken.workspace_id)
            .where(
                ApiToken.token_hash == digest,
                ApiToken.revoked_at.is_(None),
                Workspace.name == get_settings().workspace,
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        now = datetime.now(UTC)
        if row.last_used_at is None or now - row.last_used_at >= _LAST_USED_GRANULARITY:
            session.execute(update(ApiToken).where(ApiToken.id == row.id).values(last_used_at=now))
        return _info(row)


def can_read(info: TokenInfo) -> bool:
    """Whether the token may read notes and sign in (the ``web`` or ``admin`` scope)."""
    return WEB_SCOPE in info.scopes or ADMIN_SCOPE in info.scopes


def principal_for(info: TokenInfo) -> Principal:
    """The principal a verified token acts as.

    Its user's ACL grants apply only with a read scope (:func:`can_read`); a
    token with only ``ingest:`` scopes feeds its roots and reads nothing.
    """
    return Principal(
        user=info.principal,
        acl_prefixes=resolve_prefixes(info.principal, []) if can_read(info) else [],
        can_open_files=False,
        is_admin=ADMIN_SCOPE in info.scopes,
        ingest_roots=frozenset(
            scope[len(INGEST_SCOPE) :] for scope in info.scopes if scope.startswith(INGEST_SCOPE)
        ),
    )


def list_tokens() -> list[TokenInfo]:
    workspace_id = _workspace_id()
    with session_scope() as session:
        rows = session.execute(
            select(ApiToken).where(ApiToken.workspace_id == workspace_id).order_by(ApiToken.id)
        ).scalars()
        return [_info(row) for row in rows]


def revoke_token(token_id: int) -> None:
    """Revoke a token (idempotent).

    Raises:
        NotFound: When no token of the workspace has that id.
    """
    workspace_id = _workspace_id()
    with session_scope() as session:
        row = session.execute(
            select(ApiToken).where(ApiToken.id == token_id, ApiToken.workspace_id == workspace_id)
        ).scalar_one_or_none()
        if row is None:
            raise NotFound(f"no token with id {token_id}")
        if row.revoked_at is None:
            row.revoked_at = datetime.now(UTC)


def _info(row: ApiToken) -> TokenInfo:
    return TokenInfo(
        id=int(row.id),
        principal=row.principal,
        name=row.name,
        token_prefix=row.token_prefix,
        scopes=tuple(row.scopes),
        created_at=row.created_at,
        last_used_at=row.last_used_at,
        revoked_at=row.revoked_at,
    )
