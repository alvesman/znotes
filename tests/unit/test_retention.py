"""Unit tests for the version retention rules (task 33).

:func:`versions_to_purge` and :func:`should_coalesce` are pure functions over
version rows and settings, so these tests need neither Postgres nor
OpenSearch; ``history_timeline_text`` is pure too.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from mynotes.config import Settings
from mynotes.db.models import DocumentVersion
from mynotes.versioning.history import VersionInfo, history_timeline_text
from mynotes.versioning.retention import should_coalesce, versions_to_purge

_NOW = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def version(version_no: int, *, is_latest: bool = False, age_days: float = 0.0) -> DocumentVersion:
    """A transient version row with a synthetic id, ``age_days`` in the past."""
    return DocumentVersion(
        id=version_no,
        document_id=1,
        version_no=version_no,
        content_hash=f"hash-{version_no}",
        rel_path_at_version="notas.md",
        mtime_ns=0,
        status="embedded",
        is_latest=is_latest,
        created_at=_NOW - timedelta(days=age_days),
    )


def history_info(
    version_no: int,
    created_at: datetime,
    *,
    is_latest: bool = False,
    changed: list[str] | None = None,
    doc_date: datetime | None = None,
) -> VersionInfo:
    """A :class:`VersionInfo` with placeholder values for the timeline tests."""
    return VersionInfo(
        version_no=version_no,
        created_at=created_at,
        doc_date=doc_date,
        rel_path=f"v{version_no}.md",
        status="embedded",
        chunk_count=3,
        is_latest=is_latest,
        changed_sections=changed or [],
    )


def test_max_per_document_keeps_the_newest_versions() -> None:
    """max_per_document=3 with 5 versions purges exactly v1 and v2."""
    versions = [
        version(1, age_days=40),
        version(2, age_days=30),
        version(3, age_days=20),
        version(4, age_days=10),
        version(5, is_latest=True, age_days=0),
    ]
    settings = Settings(versions={"max_per_document": 3})
    assert versions_to_purge(versions, _NOW, settings) == [1, 2]


def test_max_age_days_purges_old_non_latest_versions_only() -> None:
    """Old snapshots are purged; the latest is kept even when it is old."""
    versions = [
        version(1, age_days=90),
        version(2, age_days=31),
        version(3, age_days=10),
        version(4, is_latest=True, age_days=45),
    ]
    settings = Settings(versions={"max_age_days": 30})
    assert versions_to_purge(versions, _NOW, settings) == [1, 2]


def test_both_limits_purge_the_union() -> None:
    """With both limits set, a version is purged when either rule hits it."""
    versions = [
        version(1, age_days=90),
        version(2, age_days=40),
        version(3, age_days=35),
        version(4, is_latest=True, age_days=50),
    ]
    settings = Settings(versions={"max_per_document": 2, "max_age_days": 36})
    assert versions_to_purge(versions, _NOW, settings) == [1, 2]


def test_no_limits_purge_nothing() -> None:
    """Both limits unset (null) means an unlimited history."""
    versions = [version(1, age_days=900), version(2, is_latest=True, age_days=800)]
    settings = Settings(versions={"max_per_document": None, "max_age_days": None})
    assert versions_to_purge(versions, _NOW, settings) == []


def test_latest_version_is_never_purged() -> None:
    """max_per_document=1 keeps only the latest snapshot, never purges it."""
    versions = [version(1, age_days=5), version(2, is_latest=True, age_days=1)]
    settings = Settings(versions={"max_per_document": 1})
    assert versions_to_purge(versions, _NOW, settings) == [1]


def test_edit_within_interval_coalesces() -> None:
    """An edit 6 h after the latest snapshot, interval 24 h: coalesce."""
    settings = Settings(versions={"min_interval_hours": 24})
    latest = version(2, is_latest=True, age_days=0.25)
    assert should_coalesce(latest, _NOW, settings) is True


def test_first_version_is_never_coalesced() -> None:
    """The first snapshot is always kept as its own point in history."""
    settings = Settings(versions={"min_interval_hours": 24})
    latest = version(1, is_latest=True, age_days=0.0)
    assert should_coalesce(latest, _NOW, settings) is False


def test_edit_after_the_interval_creates_a_version() -> None:
    """An edit 2 days after the latest snapshot, interval 24 h: no coalescing."""
    settings = Settings(versions={"min_interval_hours": 24})
    latest = version(2, is_latest=True, age_days=2.0)
    assert should_coalesce(latest, _NOW, settings) is False


def test_max_per_document_one_disables_coalescing() -> None:
    """With only one snapshot allowed, every edit overwrites via retention."""
    settings = Settings(versions={"min_interval_hours": 24, "max_per_document": 1})
    latest = version(3, is_latest=True, age_days=0.0)
    assert should_coalesce(latest, _NOW, settings) is False


def test_history_timeline_text_matches_the_plan_example() -> None:
    """The compact block: latest with changes, changed versions, plain dates."""
    infos = [
        history_info(
            3,
            datetime(2026, 9, 10, 8, 0, tzinfo=UTC),
            is_latest=True,
            changed=["Orçamento", "Prazos"],
        ),
        history_info(2, datetime(2026, 6, 2, 9, 30, tzinfo=UTC), changed=["Equipa"]),
        history_info(1, datetime(2026, 3, 15, 18, 45, tzinfo=UTC)),
    ]
    assert history_timeline_text(infos, "Plano Alfa") == (
        'Version history of "Plano Alfa": '
        "v3 2026-09-10 (latest; changed: Orçamento, Prazos) · "
        "v2 2026-06-02 (changed: Equipa) · "
        "v1 2026-03-15"
    )


def test_history_timeline_text_of_a_single_version() -> None:
    """The latest version without changes shows only the ``latest`` note."""
    infos = [history_info(1, datetime(2026, 3, 15, 12, 0, tzinfo=UTC), is_latest=True)]
    assert history_timeline_text(infos, "Notas") == (
        'Version history of "Notas": v1 2026-03-15 (latest)'
    )


def test_history_timeline_text_prefers_doc_date() -> None:
    """A version with a known doc_date shows it instead of the capture date."""
    infos = [
        history_info(
            2,
            datetime(2026, 9, 24, 8, 0, tzinfo=UTC),
            is_latest=True,
            doc_date=datetime(2026, 9, 1, 0, 0, tzinfo=UTC),
        )
    ]
    assert history_timeline_text(infos, "Notas") == (
        'Version history of "Notas": v2 2026-09-01 (latest)'
    )
