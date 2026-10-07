"""Unit tests for the pure detector parts (task 34): ordering and decisions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar

import pytest

from mynotes.config import Settings
from mynotes.versioning.detector import (
    Candidate,
    SideInfo,
    _archive_folders_related,
    _as_list,
    _folder_of,
    _in_archive,
    pick_newer,
)
from mynotes.versioning.similarity import pair_score

_TH = {"versions": {"auto_threshold": 0.80, "propose_threshold": 0.55}}


def candidate(
    *,
    score: float,
    order_reason: str = "version_number",
    older_id: int = 1,
    newer_id: int = 2,
) -> Candidate:
    """Build a Candidate with fixed bookkeeping fields."""
    return Candidate(
        older_id=older_id,
        newer_id=newer_id,
        score=score,
        jaccard=0.0,
        cosine=0.0,
        name_signal=False,
        archive_move=False,
        order_reason=order_reason,  # type: ignore[arg-type]
    )


def side(
    document_id: int = 1,
    version: int | None = None,
    date: datetime | None = None,
    mtime: int | None = None,
) -> SideInfo:
    """Build a SideInfo with the given ordering signals."""
    return SideInfo(document_id, version, date, mtime)


class TestPickNewer:
    """The ordering cascade: version number, doc_date, mtime."""

    def test_higher_version_number_wins(self) -> None:
        newer, reason = pick_newer(side(1, version=2), side(2, version=1))
        assert (newer.document_id, reason) == (1, "version_number")

    def test_missing_version_counts_as_oldest(self) -> None:
        newer, reason = pick_newer(side(1), side(2, version=1))
        assert (newer.document_id, reason) == (2, "version_number")

    def test_equal_versions_fall_through_to_doc_date(self) -> None:
        date_early = datetime(2026, 3, 1, tzinfo=UTC)
        date_late = datetime(2026, 9, 1, tzinfo=UTC)
        newer, reason = pick_newer(
            side(1, version=2, date=date_early), side(2, version=2, date=date_late)
        )
        assert (newer.document_id, reason) == (2, "doc_date")

    def test_doc_date_beats_mtime_only_when_versions_tie(self) -> None:
        date = datetime(2026, 5, 1, tzinfo=UTC)
        newer, reason = pick_newer(
            side(1, version=3, date=date, mtime=1), side(2, version=1, date=None, mtime=99)
        )
        assert (newer.document_id, reason) == (1, "version_number")

    def test_missing_doc_date_counts_as_oldest(self) -> None:
        newer, reason = pick_newer(
            side(1, date=None, mtime=1), side(2, date=datetime(2026, 1, 1, tzinfo=UTC), mtime=0)
        )
        assert (newer.document_id, reason) == (2, "doc_date")

    def test_mtime_decides_without_other_signals(self) -> None:
        newer, reason = pick_newer(side(1, mtime=100), side(2, mtime=200))
        assert (newer.document_id, reason) == (2, "mtime")

    def test_equal_signals_are_ambiguous(self) -> None:
        for first, second in (
            (side(1), side(2)),
            (side(1, version=2), side(2, version=2)),
            (side(1, mtime=5), side(2, mtime=5)),
        ):
            newer, reason = pick_newer(first, second)
            assert newer is None and reason == "ambiguous"


class TestDecide:
    """Score bands; ambiguous order is at most proposed."""

    def test_auto_band_with_clear_order(self) -> None:
        assert decide_score(0.80, "mtime") == "auto"
        assert decide_score(1.0, "doc_date") == "auto"

    def test_ambiguous_order_caps_at_proposed(self) -> None:
        assert decide_score(0.95, "ambiguous") == "proposed"
        assert decide_score(0.80, "ambiguous") == "proposed"

    def test_proposed_band(self) -> None:
        assert decide_score(0.55, "mtime") == "proposed"
        assert decide_score(0.79, "mtime") == "proposed"
        assert decide_score(0.54, "mtime") == "none"
        assert decide_score(0.0, "mtime") == "none"


def decide_score(score: float, order_reason: str) -> str:
    """Decide over a synthetic candidate with the default thresholds."""
    settings = Settings()
    settings.versions.auto_threshold = 0.80
    settings.versions.propose_threshold = 0.55
    return decide_of(candidate(score=score, order_reason=order_reason), settings)


def decide_of(c: Candidate, settings: Settings) -> str:
    """Import indirection so the test module reads top-down."""
    from mynotes.versioning.detector import decide

    return decide(c, settings)


class TestArchiveFolderHelpers:
    """The child/sibling archive-folder relation used for scope and bonus."""

    ARCHIVES: ClassVar[set[str]] = {"archive", "old", "versions", "arquivo"}

    def test_folder_of(self) -> None:
        assert _folder_of("a/b/c.md") == "a/b"
        assert _folder_of("c.md") == ""

    def test_child_archive(self) -> None:
        assert _archive_folders_related("docs", "docs/archive", self.ARCHIVES)
        assert _archive_folders_related("docs/archive", "docs", self.ARCHIVES)
        assert _archive_folders_related("docs", "docs/archive/deep", self.ARCHIVES)

    def test_sibling_archive(self) -> None:
        assert _archive_folders_related("docs/x", "docs/archive", self.ARCHIVES)
        assert _archive_folders_related("docs", "arquivo", self.ARCHIVES)
        assert _archive_folders_related("", "archive", self.ARCHIVES)

    def test_unrelated_folders_are_not_archive_pairs(self) -> None:
        assert not _archive_folders_related("docs", "outros", self.ARCHIVES)
        assert not _archive_folders_related("docs/x", "docs/y", self.ARCHIVES)
        assert not _archive_folders_related("docs", "docs", self.ARCHIVES)

    def test_in_archive_flag(self) -> None:
        assert _in_archive("docs/archive", self.ARCHIVES)
        assert _in_archive("arquivo/sub/x", self.ARCHIVES)
        assert not _in_archive("docs/archived", self.ARCHIVES)
        assert not _in_archive("", self.ARCHIVES)


def test_as_list_flattens_frontmatter_values() -> None:
    """Strings stay single; lists flatten; empties vanish."""
    assert _as_list(None) == []
    assert _as_list("idea.md") == ["idea.md"]
    assert _as_list(["a.md", " b.md "]) == ["a.md", " b.md "]
    assert _as_list(["a.md", ["nested.md"]]) == ["a.md", "nested.md"]
    assert _as_list("   ") == []


def test_pair_score_used_by_detector_matches_spec() -> None:
    """The detector's score is the similarity module's pair_score."""
    assert pair_score(0.5, 0.5, name_signal=True, archive_move=True) == pytest.approx(0.7)
