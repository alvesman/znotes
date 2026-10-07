"""Content hashing for the scanner (blake3, PLAN.md §7.1 step 3)."""

from __future__ import annotations

from pathlib import Path

import blake3

from mynotes.platform.paths import long_path


class FileLockedError(Exception):
    """A file could not be read because it is locked or access was denied.

    Raised by :func:`hash_file` when opening a file raises
    ``PermissionError`` (e.g. a document locked by Office on Windows). The
    scanner catches this and retries the file on the next pass.
    """

    def __init__(self, path: Path) -> None:
        """Store the path that could not be read."""
        self.path = path
        super().__init__(f"file is locked or unreadable: {path}")


def hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the blake3 hex digest of a file's raw bytes, streamed.

    Args:
        path: The file to hash; opened through :func:`long_path` so deep
            Windows paths work.
        chunk_size: Read size in bytes (1 MiB by default).

    Returns:
        The blake3 digest as a hex string.

    Raises:
        FileLockedError: If the file cannot be opened due to permissions
            (locked by Office on Windows).
        OSError: For other I/O errors (missing file, I/O failure).
    """
    digest = blake3.blake3()
    try:
        with open(long_path(path), "rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except PermissionError as exc:
        raise FileLockedError(path) from exc
    return digest.hexdigest()


def hash_text(text: str) -> str:
    """Return the blake3 hex digest of the UTF-8 encoding of ``text``.

    Args:
        text: The text to hash.

    Returns:
        The blake3 digest as a hex string.
    """
    return blake3.blake3(text.encode("utf-8")).hexdigest()
