"""The caller identity passed through the facade (PLAN.md §11).

The facade and every interface (CLI, HTTP, MCP, web UI) pass a
:class:`Principal` on each call. ``LOCAL`` is the single-user mode: no ACL
restriction, file access allowed and every mutation permitted. Team mode
(task 52) builds principals from OIDC claims with ``acl_prefixes`` set to the
allowed path prefixes, ``can_open_files=False`` and ``is_admin`` from the
configured ``admin_principals``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Principal:
    """One authenticated caller.

    Attributes:
        user: A display name (``"local"`` in local mode).
        acl_prefixes: Exact path prefixes (relative to a source root, POSIX
            separators) the caller may access; ``None`` means unrestricted
            (local mode) and an empty list means no access at all.
        can_open_files: Whether the caller may open original files in OS apps
            (True only in local mode, task 45).
        is_admin: Whether the caller may run mutations (tag merges, version
            review, entity merges) and see the status endpoint; always True in
            local mode, from ``auth.admin_principals`` in team mode (task 52).
    """

    user: str
    acl_prefixes: list[str] | None = field(default=None)
    can_open_files: bool = False
    is_admin: bool = False


LOCAL = Principal(user="local", acl_prefixes=None, can_open_files=True, is_admin=True)
