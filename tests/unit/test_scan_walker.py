"""Unit tests for the scanner walk: ignore rules, formats and FileStat."""

from __future__ import annotations

import os
import sys
import time
import unicodedata
from pathlib import Path

import pytest

from mynotes.config import Settings
from mynotes.scan.formats import format_for
from mynotes.scan.ignore import DEFAULT_IGNORES, IgnoreRules
from mynotes.scan.walker import FileStat, stat_paths, walk_root


def _rules(root: Path, globs: list[str] | None = None) -> IgnoreRules:
    """Build ignore rules for a test root (defaults from PLAN.md §12)."""
    return IgnoreRules(root, globs if globs is not None else DEFAULT_IGNORES)


def _walk(root: Path, rules: IgnoreRules | None = None) -> dict[str, FileStat]:
    """Walk a root with a fixed now and map the results by rel_path."""
    stats = walk_root(
        root, rules if rules is not None else _rules(root), settle_seconds=10, now_ns=time.time_ns()
    )
    return {stat.rel_path: stat for stat in stats}


def test_walk_yields_supported_and_unsupported_only(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.pdf").write_bytes(b"%PDF-1.4\n")
    (sub / "c.doc").write_bytes(b"legacy")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.md").write_text("x", encoding="utf-8")
    (tmp_path / ".DS_Store").write_bytes(b"\x00")
    (tmp_path / "~$tmp.docx").write_bytes(b"lock")
    (tmp_path / "img.png").write_bytes(b"\x89PNG")

    stats = _walk(tmp_path)

    assert set(stats) == {"a.md", "sub/b.pdf", "sub/c.doc"}
    assert stats["a.md"].format == "md"
    assert stats["sub/b.pdf"].format == "pdf"
    assert stats["sub/c.doc"].format == "unsupported"
    assert all(not stat.placeholder for stat in stats.values())
    assert stats["a.md"].abs_path == tmp_path / "a.md"


def test_mynotesignore_scopes_to_its_folder(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "z.pdf").write_bytes(b"%PDF")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / ".mynotesignore").write_text("*.pdf\n", encoding="utf-8")
    (sub / "b.pdf").write_bytes(b"%PDF")

    stats = _walk(tmp_path)

    assert set(stats) == {"a.md", "z.pdf"}


def test_mynotesignore_negation_and_comments(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / ".mynotesignore").write_text("# generated\n*.pdf\n!keep.pdf\n", encoding="utf-8")
    (sub / "b.pdf").write_bytes(b"%PDF")
    (sub / "keep.pdf").write_bytes(b"%PDF")
    (tmp_path / "z.pdf").write_bytes(b"%PDF")

    stats = _walk(tmp_path)

    assert set(stats) == {"z.pdf", "sub/keep.pdf"}


def test_is_ignored_gitwildmatch_semantics(tmp_path: Path) -> None:
    rules = _rules(tmp_path)
    assert rules.is_ignored("node_modules", True)
    assert not rules.is_ignored("node_modules", False)
    assert rules.is_ignored("node_modules/deep/pkg", True)
    assert rules.is_ignored("sub/node_modules", True)
    assert rules.is_ignored(".git", True)
    assert rules.is_ignored("notes/~$lock.docx", False)
    assert rules.is_ignored("build/thing.tmp", False)
    assert not rules.is_ignored("notes/a.md", False)
    assert not rules.is_ignored("a.md", True)


def test_settled_flag_uses_settle_seconds(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh.md"
    fresh.write_text("x", encoding="utf-8")
    old = tmp_path / "old.md"
    old.write_text("x", encoding="utf-8")
    now_ns = time.time_ns()
    os.utime(fresh, ns=(now_ns, now_ns))
    hour_ago_ns = now_ns - 3600 * 1_000_000_000
    os.utime(old, ns=(hour_ago_ns, hour_ago_ns))

    stats = _walk(tmp_path)

    assert stats["fresh.md"].settled is False
    assert stats["old.md"].settled is True


def test_zero_byte_document_is_placeholder(tmp_path: Path) -> None:
    (tmp_path / "empty.pdf").write_bytes(b"")
    (tmp_path / "full.pdf").write_bytes(b"%PDF")

    stats = _walk(tmp_path)

    assert stats["empty.pdf"].placeholder is True
    assert stats["full.pdf"].placeholder is False


def test_directory_symlink_loop_does_not_recurse(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "x.md").write_text("x", encoding="utf-8")
    try:
        os.symlink("..", sub / "loop", target_is_directory=True)
    except OSError:
        pytest.skip("symlinks cannot be created on this platform/account")

    stats = _walk(tmp_path)

    assert set(stats) == {"sub/x.md"}


@pytest.mark.skipif(sys.platform == "win32", reason="creating symlinks needs privileges on Windows")
def test_file_symlink_inside_root_is_followed(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "real.md").write_text("x", encoding="utf-8")
    (tmp_path / "link.md").symlink_to(notes / "real.md")

    stats = _walk(tmp_path)

    assert set(stats) == {"notes/real.md", "link.md"}
    assert stats["link.md"].abs_path == tmp_path / "link.md"
    assert stats["link.md"].rel_path == "link.md"


@pytest.mark.skipif(sys.platform == "win32", reason="creating symlinks needs privileges on Windows")
def test_file_symlink_outside_root_is_skipped(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    target = outside / "secret.md"
    target.write_text("x", encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.md").write_text("x", encoding="utf-8")
    (root / "link.md").symlink_to(target)

    stats = _walk(root)

    assert set(stats) == {"a.md"}


def test_stat_paths_expands_directories_and_reports_missing(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.pdf").write_bytes(b"%PDF")
    (sub / "c.doc").write_bytes(b"legacy")

    found, missing = stat_paths(
        tmp_path, ["a.md", "gone.md", "sub"], _rules(tmp_path), settle_seconds=10
    )

    rels = {stat.rel_path for stat in found}
    assert "a.md" in rels
    assert {"sub/b.pdf", "sub/c.doc"} <= rels
    assert missing == ["gone.md"]


def test_stat_paths_missing_for_ignored_and_non_documents(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "img.png").write_bytes(b"\x89PNG")
    (tmp_path / "~$x.docx").write_bytes(b"lock")
    (tmp_path / "node_modules").mkdir()

    found, missing = stat_paths(
        tmp_path,
        ["a.md", "img.png", "~$x.docx", "node_modules", "../escape.md"],
        _rules(tmp_path),
        settle_seconds=10,
    )

    assert {stat.rel_path for stat in found} == {"a.md"}
    assert missing == ["img.png", "~$x.docx", "node_modules", "../escape.md"]


def test_rel_path_is_nfc(tmp_path: Path) -> None:
    nfc_name = "caf\u00e9.md"
    nfd_name = unicodedata.normalize("NFD", nfc_name)
    (tmp_path / nfd_name).write_text("x", encoding="utf-8")

    stats = _walk(tmp_path)

    assert list(stats) == [nfc_name]
    assert stats[nfc_name].rel_path == unicodedata.normalize("NFC", nfc_name)


def test_walk_survives_unreadable_directory(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    locked = tmp_path / "locked"
    locked.mkdir()
    if sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0):
        pytest.skip("chmod-based locks do not apply on this platform/account")
    locked.chmod(0o000)

    try:
        stats = _walk(tmp_path)
    finally:
        locked.chmod(0o755)

    assert set(stats) == {"a.md"}


def test_format_for_is_case_insensitive_and_legacy_aware() -> None:
    assert format_for("notes/a.md") == "md"
    assert format_for("notes/A.MD") == "md"
    assert format_for("notes/b.markdown") == "md"
    assert format_for("notes/old.DOC") == "unsupported"
    assert format_for("notes/sheet.ods") == "unsupported"
    assert format_for("notes/img.png") is None
    assert format_for("no-extension") is None


def test_format_for_respects_settings_formats(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(formats=["md"])
    monkeypatch.setattr("mynotes.config.get_settings", lambda: settings)

    assert format_for("a.txt") is None
    assert format_for("a.md") == "md"
    assert format_for("legacy.doc") == "unsupported"
