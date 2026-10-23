"""Keeping a remote root in step with the agent's folder: manifests and uploads."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mynotes.agent.client import ServerClient
from mynotes.agent.hashcache import HashCache
from mynotes.config import AgentConfig, ScanConfig
from mynotes.scan.hashing import FileLockedError, hash_file
from mynotes.scan.ignore import IgnoreRules
from mynotes.scan.walker import FileStat, stat_paths, walk_root
from mynotes.scan.watch import HintBatch

logger = logging.getLogger(__name__)


@dataclass
class SyncReport:
    """What one sync did."""

    scope: str
    listed: int = 0
    uploaded: int = 0
    failed: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    skipped_unsettled: int = 0
    skipped_large: int = 0


class AgentSync:
    """Lists the folder, sends manifests, uploads what the server asks for."""

    def __init__(
        self,
        agent: AgentConfig,
        scan: ScanConfig,
        client: ServerClient,
        cache: HashCache | None = None,
    ) -> None:
        self.agent = agent
        self.scan = scan
        self.client = client
        self.cache = cache or HashCache(None)
        self.path = Path(agent.path).expanduser()
        self.rules = IgnoreRules(self.path, scan.ignore)
        self.max_upload_bytes: int | None = None
        self._reported_large: set[str] = set()

    def learn_limits(self) -> dict[str, Any]:
        """Ask the server about this root; remember its upload limit. Returns the status."""
        status = self.client.status()
        limit_mb = status.get("max_upload_mb")
        if isinstance(limit_mb, int | float) and limit_mb > 0:
            self.max_upload_bytes = int(limit_mb * 1024 * 1024)
        return status

    def _too_large(self, stat: FileStat) -> bool:
        if self.max_upload_bytes is None or stat.size <= self.max_upload_bytes:
            return False
        if stat.rel_path not in self._reported_large:
            self._reported_large.add(stat.rel_path)
            logger.warning(
                "skipping %s: %d bytes, larger than the server accepts (%d MB)",
                stat.rel_path,
                stat.size,
                self.max_upload_bytes // (1024 * 1024),
            )
        return True

    # -- listing --------------------------------------------------------------

    def _entry(self, stat: FileStat) -> dict[str, Any] | None:
        """A manifest entry for one file, hashing it unless the cache knows it."""
        if stat.placeholder:
            return {
                "rel_path": stat.rel_path,
                "size": stat.size,
                "mtime_ns": stat.mtime_ns,
                "blake3": None,
                "placeholder": True,
            }
        digest = self.cache.get(stat.rel_path, stat.size, stat.mtime_ns)
        if digest is None:
            try:
                digest = hash_file(stat.abs_path)
            except (FileLockedError, OSError) as exc:
                logger.info("skipping %s for now: %s", stat.rel_path, exc)
                return None
            self.cache.put(stat.rel_path, stat.size, stat.mtime_ns, digest)
        return {
            "rel_path": stat.rel_path,
            "size": stat.size,
            "mtime_ns": stat.mtime_ns,
            "blake3": digest,
            "placeholder": False,
        }

    def list_all(self) -> tuple[list[dict[str, Any]], int]:
        """Every file of the folder as manifest entries; unsettled files are left out."""
        return self._entries_for(
            walk_root(
                self.path,
                self.rules,
                settle_seconds=self.agent.settle_seconds,
                skip_placeholders=self.scan.skip_cloud_placeholders,
            )
        )

    def list_paths(self, rel_paths: Sequence[str]) -> tuple[list[dict[str, Any]], int]:
        """The given paths that exist, as manifest entries."""
        stats, _missing = stat_paths(
            self.path, rel_paths, self.rules, settle_seconds=self.agent.settle_seconds
        )
        return self._entries_for(stats)

    def _entries_for(self, stats: Iterable[FileStat]) -> tuple[list[dict[str, Any]], int]:
        entries: list[dict[str, Any]] = []
        unsettled = 0
        self._last_skipped_large = 0
        for stat in stats:
            if not stat.settled:
                unsettled += 1
                continue
            if self._too_large(stat):
                self._last_skipped_large += 1
                continue
            entry = self._entry(stat)
            if entry is not None:
                entries.append(entry)
        self.cache.save()
        return entries, unsettled

    # -- syncing --------------------------------------------------------------

    def full_sync(self) -> SyncReport:
        """List everything, send a full manifest, upload what the server asks for."""
        entries, unsettled = self.list_all()
        report = SyncReport(
            scope="full",
            listed=len(entries),
            skipped_unsettled=unsettled,
            skipped_large=self._last_skipped_large,
        )
        answer = self.client.manifest(entries, scope="full")
        report.counts = dict(answer.get("counts", {}))
        self._upload_all(answer.get("upload", []), entries, report)
        return report

    def sync_paths(self, rel_paths: Sequence[str]) -> SyncReport:
        """Report the given paths (present or not) and upload what the server asks for."""
        paths = sorted(set(rel_paths))
        entries, unsettled = self.list_paths(paths)
        report = SyncReport(
            scope="paths",
            listed=len(entries),
            skipped_unsettled=unsettled,
            skipped_large=self._last_skipped_large,
        )
        answer = self.client.manifest(entries, scope="paths", paths=paths)
        report.counts = dict(answer.get("counts", {}))
        self._upload_all(answer.get("upload", []), entries, report)
        return report

    def handle_batch(self, batch: HintBatch) -> None:
        """The watcher's callback: an overflow means a full sync, else the changed paths."""
        report = self.full_sync() if batch.overflow else self.sync_paths(sorted(batch.rel_paths))
        _log_report(report)
        if report.failed:
            raise RuntimeError(f"{len(report.failed)} upload(s) failed")  # the watcher retries

    def _upload_all(
        self, rel_paths: Sequence[str], entries: Sequence[dict[str, Any]], report: SyncReport
    ) -> None:
        by_path = {entry["rel_path"]: entry for entry in entries}
        todo = [by_path[rel] for rel in rel_paths if rel in by_path]
        if not todo:
            return
        with ThreadPoolExecutor(max_workers=self.agent.upload_concurrency) as pool:
            for entry, ok in zip(todo, pool.map(self._upload_one, todo), strict=True):
                if ok:
                    report.uploaded += 1
                else:
                    report.failed.append(entry["rel_path"])

    def _upload_one(self, entry: dict[str, Any]) -> bool:
        rel = entry["rel_path"]
        try:
            self.client.upload(
                rel,
                self.path / rel,
                content_hash=entry["blake3"],
                mtime_ns=entry["mtime_ns"],
                size=entry["size"],
            )
            logger.info("uploaded %s (%d bytes)", rel, entry["size"])
            return True
        except Exception:
            logger.warning("upload of %s failed", rel, exc_info=True)
            return False


def _log_report(report: SyncReport) -> None:
    logger.info(
        "sync (%s): %d file(s) listed, %d uploaded, %d failed, %d unsettled, %d too large; "
        "server: %s",
        report.scope,
        report.listed,
        report.uploaded,
        len(report.failed),
        report.skipped_unsettled,
        report.skipped_large,
        report.counts,
    )
