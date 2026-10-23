"""Tokens against the database: create, verify, revoke, scopes bound to remote roots."""

from __future__ import annotations

from uuid import uuid4

import pytest

from mynotes.config import Settings
from mynotes.core import roots, tokens
from mynotes.core.errors import NotFound

pytestmark = pytest.mark.integration


def test_token_lifecycle(settings_with_db: Settings) -> None:
    del settings_with_db
    user = f"alice-{uuid4().hex[:6]}"
    root = f"{user}-notes"
    root_id = roots.add_remote_root(root, user)
    try:
        with pytest.raises(ValueError):
            tokens.create_token(user, "laptop", ["ingest:nope"])  # not a remote root
        with pytest.raises(ValueError):
            tokens.create_token(user, "laptop", ["bogus"])

        secret, info = tokens.create_token(user, "laptop", [f"ingest:{root}", "web"])
        assert secret.startswith("mn_") and info.scopes == (f"ingest:{root}", "web")
        verified = tokens.verify_token(secret)
        assert verified is not None and verified.id == info.id
        principal = tokens.principal_for(verified)
        assert principal.ingest_roots == {root}
        assert principal.acl_prefixes == [root]  # from the root's grant
        assert [t.id for t in tokens.list_tokens() if t.principal == user] == [info.id]
        assert tokens.verify_token("mn_not-a-real-secret") is None

        tokens.revoke_token(info.id)
        assert tokens.verify_token(secret) is None
        tokens.revoke_token(info.id)  # idempotent
        with pytest.raises(NotFound):
            tokens.revoke_token(999_999)
    finally:
        _remove_root(root_id, user)


def _remove_root(root_id: int, user: str) -> None:
    from sqlalchemy import delete

    from mynotes.db.models import AclGrant, ApiToken, SourceRoot
    from mynotes.db.session import session_scope

    with session_scope() as session:
        session.execute(delete(ApiToken).where(ApiToken.principal == user))
        session.execute(delete(AclGrant).where(AclGrant.principal == user))
        session.execute(delete(SourceRoot).where(SourceRoot.id == root_id))
