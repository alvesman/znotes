"""Unit tests for the content-addressed blob store of remote roots."""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import blake3
import pytest

from mynotes.storage.blobs import BlobStore, HashMismatch, TooLarge

DIGEST = blake3.blake3(b"hello notes").hexdigest()


def test_store_writes_the_blob_under_hash_and_format(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    stored = store.store(1, 2, "md", [b"hello ", b"notes"], expected_hash=DIGEST)
    assert stored.content_hash == DIGEST
    assert stored.size == 11
    assert stored.path == tmp_path / "1" / "2" / DIGEST[:2] / f"{DIGEST}.md"
    assert stored.path.read_bytes() == b"hello notes"
    assert store.exists(1, 2, DIGEST, "md")
    assert list(store.tmp_dir.iterdir()) == []  # no partial file left behind


def test_store_is_idempotent_and_keeps_the_first_copy(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    first = store.store(1, 2, "md", [b"hello notes"])
    mtime = first.path.stat().st_mtime_ns
    second = store.store(1, 2, "md", [b"hello notes"])
    assert second.path == first.path
    assert first.path.stat().st_mtime_ns == mtime


def test_hash_mismatch_and_size_limit_discard_the_bytes(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    with pytest.raises(HashMismatch):
        store.store(1, 2, "md", [b"hello notes"], expected_hash="00" * 32)
    with pytest.raises(TooLarge):
        store.store(1, 2, "md", [b"x" * 10, b"y" * 10], max_bytes=15)
    assert not store.exists(1, 2, DIGEST, "md")
    assert list(store.tmp_dir.iterdir()) == []


def test_path_for_rejects_bad_hashes_and_formats(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    with pytest.raises(ValueError):
        store.path_for(1, 2, "../etc/passwd", "md")
    with pytest.raises(ValueError):
        store.path_for(1, 2, DIGEST, "md/../x")


def test_sweep_tmp_removes_old_partial_uploads(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    store.tmp_dir.mkdir(parents=True)
    old = store.tmp_dir / "old.part"
    old.write_bytes(b"x")
    stale = time.time() - 2 * 24 * 3600
    os.utime(old, (stale, stale))
    (store.tmp_dir / "fresh.part").write_bytes(b"y")
    assert store.sweep_tmp() == 1
    assert sorted(p.name for p in store.tmp_dir.iterdir()) == ["fresh.part"]


def test_version_source_path_for_local_and_remote_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mynotes.config import Settings
    from mynotes.jobs.pipeline import version_source_path

    local = SimpleNamespace(kind="local", abs_path=str(tmp_path), workspace_id=1, id=3)
    assert version_source_path(local, "a/b.md", DIGEST, "md") == tmp_path / "a" / "b.md"

    remote = SimpleNamespace(kind="remote", abs_path=None, workspace_id=1, id=3)
    monkeypatch.setattr("mynotes.storage.blobs.get_settings", lambda: Settings())
    assert version_source_path(remote, "r/b.md", DIGEST, "md") is None  # no storage.dir
    settings = Settings(storage={"dir": str(tmp_path / "blobs")})
    monkeypatch.setattr("mynotes.storage.blobs.get_settings", lambda: settings)
    assert version_source_path(remote, "r/b.md", DIGEST, "md") == (
        tmp_path / "blobs" / "1" / "3" / DIGEST[:2] / f"{DIGEST}.md"
    )
    assert version_source_path(remote, "r/b.md", None, "md") is None
