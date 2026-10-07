"""Unit tests for mynotes.platform.fsinfo."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mynotes.platform import fsinfo
from mynotes.platform.fsinfo import (
    filesystem_type,
    inotify_watch_limit,
    is_cloud_placeholder,
    is_wsl_windows_mount,
    probe_case_sensitive,
    watch_backend_for,
)


def test_probe_case_sensitive_returns_bool_and_cleans_up(tmp_path: Path) -> None:
    result = probe_case_sensitive(tmp_path)

    assert isinstance(result, bool)
    assert list(tmp_path.iterdir()) == []


def test_probe_case_sensitive_falls_back_when_root_not_writable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(self: Path, mode: str = "r", *args: object, **kwargs: object) -> object:
        raise PermissionError(13, "permission denied")

    monkeypatch.setattr(Path, "open", denied)

    assert probe_case_sensitive(tmp_path) == (sys.platform == "linux")


def test_watch_backend_auto_polls_on_network_fs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fsinfo, "is_network_fs", lambda p: True)
    assert watch_backend_for(tmp_path, "auto") == "polling"


def test_watch_backend_auto_native_when_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fsinfo, "is_network_fs", lambda p: False)
    monkeypatch.setattr(fsinfo, "is_wsl_windows_mount", lambda p: False)
    monkeypatch.setattr(fsinfo, "in_container", lambda: False)
    assert watch_backend_for(tmp_path, "auto") == "native"


def test_watch_backend_returns_configured_backend(tmp_path: Path) -> None:
    assert watch_backend_for(tmp_path, "native") == "native"
    assert watch_backend_for(tmp_path, "polling") == "polling"


def test_is_cloud_placeholder_icloud_stub(tmp_path: Path) -> None:
    base = tmp_path / "base.pdf"
    base.write_bytes(b"%PDF-1.4")

    assert is_cloud_placeholder(tmp_path / "doc.x.pdf.icloud", base.stat()) is True


def test_is_cloud_placeholder_zero_byte_documents(tmp_path: Path) -> None:
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(b"")
    empty_md = tmp_path / "empty.md"
    empty_md.write_bytes(b"")

    assert is_cloud_placeholder(empty_pdf, empty_pdf.stat()) is True
    assert is_cloud_placeholder(empty_md, empty_md.stat()) is True


def test_is_cloud_placeholder_normal_file(tmp_path: Path) -> None:
    notes = tmp_path / "notes.md"
    notes.write_text("# Idea\n\nreal content", encoding="utf-8")

    assert is_cloud_placeholder(notes, notes.stat()) is False


def test_is_cloud_placeholder_zero_byte_non_document(tmp_path: Path) -> None:
    blob = tmp_path / "empty.bin"
    blob.write_bytes(b"")

    assert is_cloud_placeholder(blob, blob.stat()) is False


def test_filesystem_type_returns_string_or_none(tmp_path: Path) -> None:
    result = filesystem_type(tmp_path)

    assert result is None or isinstance(result, str)


def test_is_wsl_windows_mount(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(fsinfo.platform, "release", lambda: "5.15.167.4-microsoft-standard-WSL2")

    assert is_wsl_windows_mount(Path("/mnt/c/Users/me")) is True
    assert is_wsl_windows_mount(Path("/home/me/notes")) is False

    monkeypatch.setattr(fsinfo.platform, "release", lambda: "6.8.0-45-generic")
    assert is_wsl_windows_mount(Path("/mnt/c/Users/me")) is False


def test_inotify_watch_limit() -> None:
    limit = inotify_watch_limit()

    if sys.platform == "linux":
        assert limit is None or isinstance(limit, int)
    else:
        assert limit is None
