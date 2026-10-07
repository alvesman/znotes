"""Team-mode ACL: path-prefix grants, resolution and evidence checks (PLAN.md §11, task 52).

Grants live in the ``acl_grant`` table, one row per (workspace, principal,
path prefix); a principal is a username or a ``group:<name>``.
:func:`resolve_prefixes` is the union of a caller's user and group grants,
cached for 60 s per ``(user, groups)`` so OIDC-authenticated requests never
make a per-request auth call (an anti-pattern). A grant with the empty
prefix means the whole workspace. Prefixes are exact rel-path folder prefixes:
case-sensitive, never folded (PLAN.md §6.1), no trailing slash.

The evidence primitives (``path_allowed``, ``relation_has_allowed_evidence``,
``allowed_entity_ids``, ``filter_subgraph_by_acl``) are the shared rules behind
the facade's :func:`~mynotes.core.service.filter_graph_by_acl`, the retrieval
expanders and the entity cards: a chunk, edge or entity is visible only when
at least one of its paths lies inside the principal's prefixes.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from mynotes.config import get_settings
from mynotes.core.errors import NotFound
from mynotes.db.models import (
    AclGrant,
    Chunk,
    Document,
    File,
    KgMention,
    KgRelation,
    KgRelationEvidence,
)
from mynotes.db.session import session_scope
from mynotes.platform.paths import path_ancestors

if TYPE_CHECKING:
    from mynotes.graph.expand import Subgraph

logger = logging.getLogger(__name__)

GROUP_PREFIX = "group:"
"""Marker for group principals in ``acl_grant`` and ``admin_principals``."""

_CACHE_TTL_S = 60.0
"""How long a ``(user, groups)`` prefix resolution stays cached."""

_cache: dict[tuple[str, tuple[str, ...]], tuple[float, tuple[str, ...]]] = {}
"""Resolved prefixes per (user, sorted groups) with their expiry timestamps."""


def _now() -> float:
    """Monotonic clock (seam for cache-expiry tests)."""
    return time.monotonic()


def normalize_prefix(prefix: str) -> str:
    """Normalize an ACL prefix: trimmed, no trailing slash, path-safe.

    Args:
        prefix: The prefix as typed (``""`` means the whole workspace).

    Returns:
        The normalized prefix (``""`` for the whole workspace).

    Raises:
        ValueError: When the prefix is absolute, contains a ``..`` segment or
            is only slashes.
    """
    cleaned = prefix.strip().replace("\\", "/")
    cleaned = cleaned.rstrip("/")
    if not cleaned:
        return ""
    segments = [segment for segment in cleaned.split("/") if segment]
    if cleaned.startswith("/") or ".." in segments or not segments:
        raise ValueError(f"invalid ACL prefix {prefix!r}")
    normalized = "/".join(segments)
    if normalized != prefix:
        logger.debug("normalized ACL prefix %r to %r", prefix, normalized)
    return normalized


def normalize_principal(principal: str) -> str:
    """Trim a principal name and reject empties (``user`` or ``group:<name>``)."""
    cleaned = principal.strip()
    if not cleaned or cleaned == GROUP_PREFIX:
        raise ValueError("principal must be a username or 'group:<name>'")
    return cleaned


def grant_prefixes(principals: Sequence[str]) -> list[str]:
    """The distinct ``acl_grant.path_prefix`` rows of the workspace for ``principals``.

    Args:
        principals: Principal names to look up (user and ``group:`` entries).

    Returns:
        The normalized prefixes, sorted and deduplicated; empty when the
        workspace does not exist.
    """
    from mynotes.tags.store import workspace_id_by_name

    with session_scope() as session:
        workspace_id = workspace_id_by_name(session, get_settings().workspace)
        if workspace_id is None:
            return []
        rows = session.execute(
            select(AclGrant.path_prefix).where(
                AclGrant.workspace_id == workspace_id,
                AclGrant.principal.in_(list(principals)),
            )
        ).scalars()
        return sorted({normalize_prefix(str(prefix)) for prefix in rows})


def resolve_prefixes(user: str, groups: Sequence[str]) -> list[str]:
    """The path prefixes a principal may access: its user and group grants.

    The union of ``acl_grant.path_prefix`` for the user and every
    ``group:<name>`` principal, in the configured workspace, cached for 60 s
    per ``(user, groups)``. A grant of ``""`` (whole workspace) is returned as
    ``[""]``, which every ACL check treats as allow-all.

    Args:
        user: The authenticated username.
        groups: The group names from the identity claim.

    Returns:
        The allowed prefixes (possibly empty: no grant means no access).
    """
    key = (user, tuple(sorted(groups)))
    now = _now()
    cached = _cache.get(key)
    if cached is not None and cached[0] > now:
        return list(cached[1])
    principals = [user, *(f"{GROUP_PREFIX}{group}" for group in groups)]
    prefixes = tuple(grant_prefixes(principals))
    _cache[key] = (now + _CACHE_TTL_S, prefixes)
    return list(prefixes)


def clear_cache() -> None:
    """Drop every cached prefix resolution (grant changes and tests)."""
    _cache.clear()


def grant(principal: str, path_prefix: str) -> None:
    """Add one read grant, ignoring an identical existing row.

    Args:
        principal: A username or ``group:<name>``.
        path_prefix: Exact folder prefix (``""`` for the whole workspace).

    Raises:
        NotFound: When the workspace does not exist.
        ValueError: When the principal or prefix is invalid.
    """
    who = normalize_principal(principal)
    prefix = normalize_prefix(path_prefix)
    with session_scope() as session:
        workspace_id = _require_workspace_id(session)
        session.execute(
            pg_insert(AclGrant)
            .values(
                workspace_id=workspace_id,
                principal=who,
                path_prefix=prefix,
                permission="read",
            )
            .on_conflict_do_nothing(index_elements=["workspace_id", "principal", "path_prefix"])
        )
    clear_cache()


def revoke(principal: str, path_prefix: str) -> None:
    """Remove one grant.

    Args:
        principal: A username or ``group:<name>``.
        path_prefix: The granted prefix (``""`` for the whole workspace).

    Raises:
        NotFound: When the workspace or the grant row does not exist.
        ValueError: When the principal or prefix is invalid.
    """
    who = normalize_principal(principal)
    prefix = normalize_prefix(path_prefix)
    with session_scope() as session:
        workspace_id = _require_workspace_id(session)
        deleted = session.execute(
            delete(AclGrant).where(
                AclGrant.workspace_id == workspace_id,
                AclGrant.principal == who,
                AclGrant.path_prefix == prefix,
            )
        ).rowcount
    if not deleted:
        raise NotFound(f"no grant for {who!r} on prefix {prefix!r}")
    clear_cache()


@dataclass(frozen=True)
class GrantView:
    """One ``acl_grant`` row for the ``mynotes acl list`` table.

    Attributes:
        principal: The username or ``group:<name>``.
        path_prefix: The allowed prefix (``""`` for the whole workspace).
        created_at: When the grant was created.
    """

    principal: str
    path_prefix: str
    created_at: datetime


def list_grants() -> list[GrantView]:
    """Every grant of the configured workspace, ordered by principal and prefix.

    Raises:
        NotFound: When the workspace does not exist.
    """
    with session_scope() as session:
        workspace_id = _require_workspace_id(session)
        rows = session.execute(
            select(AclGrant.principal, AclGrant.path_prefix, AclGrant.created_at)
            .where(AclGrant.workspace_id == workspace_id)
            .order_by(AclGrant.principal, AclGrant.path_prefix)
        ).all()
    return [
        GrantView(principal=str(who), path_prefix=str(prefix), created_at=created_at)
        for who, prefix, created_at in rows
    ]


def _require_workspace_id(session: Session) -> int:
    """The configured workspace's id, or ``NotFound`` when missing."""
    from mynotes.tags.store import workspace_id_by_name

    workspace_id = workspace_id_by_name(session, get_settings().workspace)
    if workspace_id is None:
        raise NotFound("no workspace; run 'mynotes init' first")
    return int(workspace_id)


def path_allowed(path: str, prefixes: Sequence[str]) -> bool:
    """Whether a relative path is inside one of the prefixes (exact, case-sensitive).

    The same rule as the hybrid-search ACL clause: the path's exact
    ``path_ancestors`` contain a prefix, or the path itself starts with
    ``prefix + "/"``. The empty prefix means the whole workspace; an empty
    path is never allowed (nothing is known about where it lives).

    Args:
        path: The path relative to its source root.
        prefixes: The allowed prefixes.

    Returns:
        True when the path is allowed.
    """
    if not path:
        return False
    if "" in prefixes:
        return True
    ancestors = path_ancestors(path)
    return any(prefix in ancestors or path.startswith(prefix + "/") for prefix in prefixes)


def relation_has_allowed_evidence(
    session: Session, relation_id: int, acl_prefixes: Sequence[str]
) -> bool:
    """Whether one relation has at least one evidence chunk inside the prefixes."""
    paths = session.execute(
        select(File.rel_path)
        .select_from(KgRelationEvidence)
        .join(Chunk, Chunk.id == KgRelationEvidence.chunk_id)
        .join(Document, Document.id == Chunk.document_id)
        .outerjoin(File, File.id == Document.current_file_id)
        .where(KgRelationEvidence.relation_id == relation_id)
    ).scalars()
    return any(rel_path and path_allowed(rel_path, acl_prefixes) for rel_path in paths)


def allowed_entity_ids(
    session: Session, entity_ids: Sequence[int], acl_prefixes: Sequence[str]
) -> set[int]:
    """Entity ids that have at least one mention or evidence chunk in the prefixes."""
    if not entity_ids:
        return set()
    mention_paths = session.execute(
        select(KgMention.entity_id, File.rel_path)
        .join(Chunk, Chunk.id == KgMention.chunk_id)
        .join(Document, Document.id == Chunk.document_id)
        .outerjoin(File, File.id == Document.current_file_id)
        .where(KgMention.entity_id.in_(list(entity_ids)))
    ).all()
    evidence_paths = session.execute(
        select(KgRelation.src_entity_id, KgRelation.dst_entity_id, File.rel_path)
        .select_from(KgRelationEvidence)
        .join(KgRelation, KgRelation.id == KgRelationEvidence.relation_id)
        .join(Chunk, Chunk.id == KgRelationEvidence.chunk_id)
        .join(Document, Document.id == Chunk.document_id)
        .outerjoin(File, File.id == Document.current_file_id)
        .where(
            (KgRelation.src_entity_id.in_(list(entity_ids)))
            | (KgRelation.dst_entity_id.in_(list(entity_ids)))
        )
    ).all()
    allowed: set[int] = set()
    for entity_id, rel_path in mention_paths:
        if rel_path and path_allowed(rel_path, acl_prefixes):
            allowed.add(int(entity_id))
    for src, dst, rel_path in evidence_paths:
        if rel_path and path_allowed(rel_path, acl_prefixes):
            allowed.add(int(src))
            allowed.add(int(dst))
    return allowed & set(entity_ids)


def filter_subgraph_by_acl(
    session: Session, sub: Subgraph, acl_prefixes: Sequence[str] | None
) -> Subgraph:
    """Restrict a subgraph to entities and relations with allowed evidence.

    Only entities with at least one mention or relation-evidence chunk inside
    the allowed prefixes and only edges with such an evidence chunk survive
    (PLAN.md §11 "Graph ACL"). ``acl_prefixes=None`` (local mode) returns the
    subgraph unchanged; an empty list returns an empty subgraph.

    Args:
        session: An open session/transaction.
        sub: The subgraph to filter.
        acl_prefixes: The allowed path prefixes, None for unrestricted.

    Returns:
        The filtered :class:`~mynotes.graph.expand.Subgraph`.
    """
    if acl_prefixes is None:
        return sub
    if not sub.nodes:
        return type(sub)()
    allowed_entities = allowed_entity_ids(session, list(sub.nodes), acl_prefixes)
    kept_edges = [
        edge
        for edge in sub.edges
        if edge[0] in allowed_entities
        and edge[1] in allowed_entities
        and relation_has_allowed_evidence(session, edge[2], acl_prefixes)
    ]
    kept_nodes = {node for edge in kept_edges for node in edge[:2]} & set(sub.nodes)
    return type(sub)(
        nodes={node_id: name for node_id, name in sub.nodes.items() if node_id in kept_nodes},
        edges=kept_edges,
    )
