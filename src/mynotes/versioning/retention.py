"""Version retention rules: snapshot coalescing and history trimming (PLAN.md §5.1).

:func:`should_coalesce` decides whether a MODIFIED scan overwrites the latest
snapshot instead of adding a new version, so a frequently saved note does not
explode the history; :func:`versions_to_purge` picks the snapshots that the
configurable retention (:attr:`Settings.versions.max_per_document`,
:attr:`Settings.versions.max_age_days`) no longer keeps. Both are pure
functions over version rows and settings; the scanner and the
``apply_retention`` job handler call them.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

from mynotes.config import Settings
from mynotes.db.models import DocumentVersion


def should_coalesce(latest: DocumentVersion, now: datetime, settings: Settings) -> bool:
    """Decide whether a MODIFIED scan should overwrite the latest snapshot.

    An edit within ``versions.min_interval_hours`` of the latest snapshot's
    creation reuses that snapshot instead of adding a new one. The first
    snapshot (version 1) is always kept as its own point in history, and with
    ``versions.max_per_document = 1`` only the newest snapshot exists at all,
    so coalescing is disabled and every edit overwrites through retention.

    Args:
        latest: The document's latest version row.
        now: The scan time (UTC).
        settings: Application settings (the ``versions`` section).

    Returns:
        True when the edit should reuse ``latest`` instead of creating a new
        version.
    """
    if settings.versions.max_per_document == 1:
        return False
    if latest.version_no <= 1:
        return False
    return now - latest.created_at < timedelta(hours=settings.versions.min_interval_hours)


def versions_to_purge(
    versions: Sequence[DocumentVersion], now: datetime, settings: Settings
) -> list[int]:
    """Pick the version ids that retention should delete.

    The latest version is never purged. ``versions.max_per_document = N`` keeps
    the N newest snapshots (by ``version_no``) and marks the older ones for
    deletion; ``versions.max_age_days = D`` marks non-latest snapshots with
    ``created_at`` before ``now - D days``; when both are set the purge set is
    the union.

    Args:
        versions: All versions of one document (any order).
        now: The current time (UTC).
        settings: Application settings (the ``versions`` section).

    Returns:
        The ids of the versions to purge (may be empty), sorted ascending.
    """
    if not versions:
        return []
    latest = next((version for version in versions if version.is_latest), None)
    if latest is None:
        latest = max(versions, key=lambda version: version.version_no)
    purge: set[int] = set()
    max_keep = settings.versions.max_per_document
    if max_keep is not None:
        newest = sorted(versions, key=lambda version: version.version_no, reverse=True)[:max_keep]
        keep = {version.id for version in newest}
        purge.update(version.id for version in versions if version.id not in keep)
    max_age_days = settings.versions.max_age_days
    if max_age_days is not None:
        cutoff = now - timedelta(days=max_age_days)
        purge.update(
            version.id
            for version in versions
            if version is not latest
            and version.created_at is not None
            and version.created_at < cutoff
        )
    purge.discard(latest.id)
    return sorted(purge)
