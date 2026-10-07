"""Unit tests for scanner hashing (blake3)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import blake3
import pytest

from mynotes.scan.hashing import FileLockedError, hash_file, hash_text


def test_hash_file_is_deterministic_and_content_sensitive(tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    first.write_bytes(b"hello world")
    second = tmp_path / "b.txt"
    second.write_bytes(b"hello world")
    third = tmp_path / "c.txt"
    third.write_bytes(b"hello worlds")

    assert hash_file(first) == hash_file(second)
    assert hash_file(first) != hash_file(third)


def test_hash_file_empty_file_matches_blake3_of_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")

    assert hash_file(empty) == blake3.blake3(b"").hexdigest()


def test_hash_file_streams_large_file_in_chunks(tmp_path: Path) -> None:
    data = os.urandom(5 * 1024 * 1024)
    big = tmp_path / "big.bin"
    big.write_bytes(data)

    assert hash_file(big, chunk_size=64 * 1024) == blake3.blake3(data).hexdigest()
    assert hash_file(big) == blake3.blake3(data).hexdigest()


def test_hash_file_raises_file_locked_error_on_permission_error(tmp_path: Path) -> None:
    if sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0):
        pytest.skip("chmod-based locks do not apply on this platform/account")
    locked = tmp_path / "locked.docx"
    locked.write_bytes(b"office file")
    locked.chmod(0o000)

    try:
        with pytest.raises(FileLockedError) as excinfo:
            hash_file(locked)
    finally:
        locked.chmod(0o644)

    assert excinfo.value.path == locked
    assert isinstance(excinfo.value.__cause__, PermissionError)


def test_hash_text_hashes_utf8_bytes() -> None:
    assert hash_text("ol\u00e1") == blake3.blake3("ol\u00e1".encode("utf-8")).hexdigest()
    assert hash_text("a") != hash_text("b")
    assert hash_text("") == blake3.blake3(b"").hexdigest()
