"""An on-disk cache of file hashes keyed by (path, size, mtime): periodic syncs rarely rehash."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class HashCache:
    """``(rel_path, size, mtime_ns) -> blake3`` as JSON; losing it only costs a rehash."""

    def __init__(self, file: Path | None) -> None:
        self.file = file
        self._entries: dict[str, list] = {}
        self._dirty = False
        if file is not None and file.exists():
            try:
                self._entries = json.loads(file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                logger.warning("ignoring unreadable hash cache %s", file)
                self._entries = {}

    def get(self, rel_path: str, size: int, mtime_ns: int) -> str | None:
        entry = self._entries.get(rel_path)
        if entry and entry[0] == size and entry[1] == mtime_ns:
            return str(entry[2])
        return None

    def put(self, rel_path: str, size: int, mtime_ns: int, content_hash: str) -> None:
        self._entries[rel_path] = [size, mtime_ns, content_hash]
        self._dirty = True

    def forget(self, rel_path: str) -> None:
        if self._entries.pop(rel_path, None) is not None:
            self._dirty = True

    def save(self) -> None:
        if self.file is None or not self._dirty:
            return
        try:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.file.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._entries), encoding="utf-8")
            tmp.replace(self.file)
            self._dirty = False
        except OSError:
            logger.warning("could not save the hash cache %s", self.file, exc_info=True)
