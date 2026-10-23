"""Content-addressed blob store for remote roots (files uploaded by agents).

Layout: ``<storage.dir>/<workspace_id>/<root_id>/<hh>/<blake3>.<format>``. The
extension matters: the Office extractor picks its reader from the suffix. A
blob is written to ``<storage.dir>/tmp`` while its hash is computed and moved
into place with :func:`os.replace`, so a crash never leaves a half-written
blob where a reader could find it. Content addressing makes uploads
idempotent: a rename or a duplicate needs no new bytes, and old versions stay
downloadable until their rows are purged.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import blake3

from mynotes.config import get_settings

logger = logging.getLogger(__name__)

_HEX = frozenset("0123456789abcdef")
TMP_MAX_AGE_S = 24 * 3600


class StorageNotConfigured(RuntimeError):
    """``storage.dir`` is unset but a remote root needs the blob store."""


class HashMismatch(ValueError):
    """The uploaded bytes do not hash to the announced digest."""


class TooLarge(ValueError):
    """The upload exceeds ``storage.max_upload_mb``."""


@dataclass(frozen=True)
class StoredBlob:
    """Where an upload ended up."""

    content_hash: str
    size: int
    path: Path


def _check_hash(content_hash: str) -> str:
    digest = content_hash.strip().lower()
    if len(digest) != 64 or not set(digest) <= _HEX:
        raise ValueError(f"not a blake3 hex digest: {content_hash!r}")
    return digest


def _check_format(fmt: str) -> str:
    if not fmt or not fmt.isalnum():
        raise ValueError(f"not a format name: {fmt!r}")
    return fmt.lower()


class BlobStore:
    """The blob store rooted at one directory."""

    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)

    @property
    def tmp_dir(self) -> Path:
        return self.dir / "tmp"

    def path_for(self, workspace_id: int, root_id: int, content_hash: str, fmt: str) -> Path:
        """The blob path for a hash and format (whether or not it exists)."""
        digest = _check_hash(content_hash)
        return (
            self.dir
            / str(int(workspace_id))
            / str(int(root_id))
            / digest[:2]
            / (f"{digest}.{_check_format(fmt)}")
        )

    def exists(self, workspace_id: int, root_id: int, content_hash: str, fmt: str) -> bool:
        return self.path_for(workspace_id, root_id, content_hash, fmt).is_file()

    def store(
        self,
        workspace_id: int,
        root_id: int,
        fmt: str,
        chunks: Iterable[bytes],
        *,
        expected_hash: str | None = None,
        max_bytes: int | None = None,
    ) -> StoredBlob:
        """Write a stream of bytes as a blob, hashing on the way.

        Args:
            workspace_id: The root's workspace.
            root_id: The remote root.
            fmt: The file's format name (its extension in the store).
            chunks: The bytes, in order.
            expected_hash: The digest the uploader announced; a mismatch
                discards the bytes and raises :class:`HashMismatch`.
            max_bytes: Discard and raise :class:`TooLarge` beyond this size.

        Returns:
            The :class:`StoredBlob`. Storing bytes that already exist is a
            no-op that returns the existing path.
        """
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.tmp_dir / f"{uuid.uuid4().hex}.part"
        digest = blake3.blake3()
        size = 0
        try:
            with open(tmp, "wb") as handle:
                for chunk in chunks:
                    size += len(chunk)
                    if max_bytes is not None and size > max_bytes:
                        raise TooLarge(f"upload exceeds {max_bytes} bytes")
                    digest.update(chunk)
                    handle.write(chunk)
            content_hash = digest.hexdigest()
            if expected_hash is not None and _check_hash(expected_hash) != content_hash:
                raise HashMismatch(
                    f"uploaded bytes hash to {content_hash}, not {expected_hash.lower()}"
                )
            target = self.path_for(workspace_id, root_id, content_hash, fmt)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                tmp.unlink(missing_ok=True)
            else:
                os.replace(tmp, target)
            return StoredBlob(content_hash=content_hash, size=size, path=target)
        finally:
            tmp.unlink(missing_ok=True)

    def new_temp(self) -> Path:
        """A fresh path under ``tmp`` for a streaming upload to write to."""
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        return self.tmp_dir / f"{uuid.uuid4().hex}.part"

    def adopt(
        self,
        tmp: Path,
        workspace_id: int,
        root_id: int,
        fmt: str,
        content_hash: str,
        *,
        expected_hash: str | None = None,
    ) -> StoredBlob:
        """Move a completed temp file (already hashed by the caller) into place.

        The temp file is always consumed. Raises :class:`HashMismatch` when
        ``expected_hash`` differs from ``content_hash``.
        """
        try:
            if expected_hash is not None and _check_hash(expected_hash) != content_hash:
                raise HashMismatch(
                    f"uploaded bytes hash to {content_hash}, not {expected_hash.lower()}"
                )
            size = tmp.stat().st_size
            target = self.path_for(workspace_id, root_id, content_hash, fmt)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.replace(tmp, target)
            return StoredBlob(content_hash=content_hash, size=size, path=target)
        finally:
            tmp.unlink(missing_ok=True)

    def remove(self, workspace_id: int, root_id: int, content_hash: str, fmt: str) -> bool:
        """Delete one blob; True when it existed."""
        path = self.path_for(workspace_id, root_id, content_hash, fmt)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True

    def sweep_tmp(self, max_age_s: float = TMP_MAX_AGE_S) -> int:
        """Delete abandoned partial uploads older than ``max_age_s``; returns the count."""
        if not self.tmp_dir.is_dir():
            return 0
        cutoff = time.time() - max_age_s
        removed = 0
        for part in self.tmp_dir.iterdir():
            try:
                if part.stat().st_mtime < cutoff:
                    part.unlink()
                    removed += 1
            except OSError:
                logger.warning("could not remove partial upload %s", part, exc_info=True)
        return removed

    def total_size(self) -> int:
        """Bytes used by every blob (walks the store)."""
        total = 0
        for path in self.dir.rglob("*"):
            if path.is_file() and path.parent != self.tmp_dir:
                total += path.stat().st_size
        return total


def get_blob_store() -> BlobStore:
    """The configured blob store.

    Raises:
        StorageNotConfigured: When ``storage.dir`` is unset.
    """
    directory = get_settings().storage.dir
    if directory is None:
        raise StorageNotConfigured(
            "storage.dir is not set: remote roots need a blob directory "
            "(set storage.dir in mynotes.yaml or MYNOTES_STORAGE_DIR)"
        )
    return BlobStore(Path(directory).expanduser())
