"""Remote roots: folders fed by an agent on the owner's machine (server mode).

A remote root has no path here. ``add_remote_root`` creates the row and
grants the owner read access under the root's name, which is the prefix every
uploaded path gets (``scan.remote``), so the owner sees their own notes and
nobody else's.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select

from mynotes.config import get_settings
from mynotes.core import acl
from mynotes.core.errors import NotFound
from mynotes.db.models import File, SourceRoot, Workspace
from mynotes.db.session import session_scope

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RootInfo:
    """One source root as listed by ``mynotes root list``."""

    id: int
    name: str
    kind: str
    enabled: bool
    owner: str | None
    abs_path: str | None
    agent_last_seen_at: datetime | None


def valid_root_name(name: str) -> str:
    """A root name is one path segment: it prefixes every uploaded path."""
    cleaned = name.strip()
    if not cleaned or "/" in cleaned or "\\" in cleaned or cleaned in (".", ".."):
        raise ValueError("a root name must be a single path segment")
    return cleaned


def add_remote_root(name: str, owner: str) -> int:
    """Create a remote root owned by ``owner`` and grant them read access under it.

    Raises:
        ValueError: When the name is taken, or clashes with a top-level folder
            of a local root (paths of both would share the prefix).
        NotFound: When the workspace does not exist (run ``mynotes init``).
    """
    name = valid_root_name(name)
    owner = owner.strip()
    if not owner:
        raise ValueError("an owner is required")
    workspace_name = get_settings().workspace
    with session_scope() as session:
        workspace = session.execute(
            select(Workspace).where(Workspace.name == workspace_name)
        ).scalar_one_or_none()
        if workspace is None:
            raise NotFound(f"workspace {workspace_name!r} does not exist; run 'mynotes init'")
        taken = session.execute(
            select(SourceRoot.id).where(
                SourceRoot.workspace_id == workspace.id, SourceRoot.name == name
            )
        ).scalar_one_or_none()
        if taken is not None:
            raise ValueError(f"a root named {name!r} already exists")
        clash = session.execute(
            select(File.id)
            .join(SourceRoot, SourceRoot.id == File.source_root_id)
            .where(
                SourceRoot.workspace_id == workspace.id,
                SourceRoot.kind == "local",
                File.rel_path.startswith(name + "/", autoescape=True),
            )
            .limit(1)
        ).scalar_one_or_none()
        if clash is not None:
            raise ValueError(
                f"a local root has a top-level folder named {name!r}; its paths would share "
                "the prefix with this root, choose another name"
            )
        row = SourceRoot(workspace_id=workspace.id, name=name, kind="remote", owner=owner)
        session.add(row)
        session.flush()
        root_id = int(row.id)
    acl.grant(owner, name)
    logger.info("created remote root %r (id %d) for %s", name, root_id, owner)
    return root_id


def list_roots() -> list[RootInfo]:
    """Every root of the workspace, local and remote, by id."""
    workspace_name = get_settings().workspace
    with session_scope() as session:
        rows = session.execute(
            select(SourceRoot)
            .join(Workspace, Workspace.id == SourceRoot.workspace_id)
            .where(Workspace.name == workspace_name)
            .order_by(SourceRoot.id)
        ).scalars()
        return [
            RootInfo(
                id=int(row.id),
                name=row.name,
                kind=row.kind,
                enabled=bool(row.enabled),
                owner=row.owner,
                abs_path=row.abs_path,
                agent_last_seen_at=row.agent_last_seen_at,
            )
            for row in rows
        ]


def disable_remote_root(name: str) -> int:
    """Disable a remote root: its chunks leave the search index; rows and blobs stay.

    Returns:
        The number of chunks queued for removal from the index.
    """
    from mynotes.core.setup import _set_root_enabled

    settings = get_settings()
    with session_scope() as session:
        row = session.execute(
            select(SourceRoot)
            .join(Workspace, Workspace.id == SourceRoot.workspace_id)
            .where(Workspace.name == settings.workspace, SourceRoot.name == name)
        ).scalar_one_or_none()
        if row is None or row.kind != "remote":
            raise NotFound(f"no remote root named {name!r}")
        if not row.enabled:
            return 0
        return int(_set_root_enabled(session, row, False, settings))
