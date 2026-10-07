"""Unit tests for mynotes.platform.paths."""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import pytest

from mynotes.platform.paths import (
    is_within,
    long_path,
    path_ancestors,
    path_ancestors_folded,
    to_rel_path,
)


def test_to_rel_path_uses_posix_separators_and_nfc(tmp_path: Path) -> None:
    folder = tmp_path / "Projetos" / "Ação"
    folder.mkdir(parents=True)
    nfd_name = unicodedata.normalize("NFD", "café.md")
    file_path = folder / nfd_name
    file_path.write_text("x", encoding="utf-8")

    rel = to_rel_path(tmp_path, file_path)

    assert rel == "Projetos/Ação/café.md"
    assert "\\" not in rel
    assert rel == unicodedata.normalize("NFC", rel)


def test_to_rel_path_raises_outside_root(tmp_path: Path) -> None:
    sibling = tmp_path.parent / "outside-sibling.md"
    sibling.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError):
        to_rel_path(tmp_path, sibling)
    with pytest.raises(ValueError):
        to_rel_path(tmp_path, tmp_path / ".." / "outside-sibling.md")


def test_path_ancestors() -> None:
    assert path_ancestors("a/b/c.md") == ["a", "a/b"]
    assert path_ancestors("c.md") == []


def test_path_ancestors_folded() -> None:
    assert path_ancestors_folded("Projetos/Ação/x.md") == ["projetos", "projetos/acao"]


@pytest.mark.skipif(sys.platform == "win32", reason="creating symlinks needs privileges on Windows")
def test_is_within_rejects_symlink_escape(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside_dir = tmp_path_factory.mktemp("outside")
    target = outside_dir / "secret.md"
    target.write_text("x", encoding="utf-8")
    link = root / "link.md"
    link.symlink_to(target)

    assert not is_within(root, link)
    assert not is_within(root, root / ".." / "secret.md")

    inside = root / "real.md"
    inside.write_text("x", encoding="utf-8")
    assert is_within(root, inside)
    assert is_within(root, root)


def test_long_path_is_identity_on_non_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if sys.platform == "win32":
        monkeypatch.setattr(sys, "platform", "linux")
    file_path = tmp_path / "a.txt"
    assert long_path(file_path) is file_path


def test_long_path_prepends_prefix_for_long_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    deep = tmp_path / ("x" * 300)

    result = long_path(deep)

    assert str(result) == "\\\\?\\" + str(deep.resolve())


def test_long_path_keeps_short_paths_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    file_path = tmp_path / "a.txt"
    assert long_path(file_path) is file_path
