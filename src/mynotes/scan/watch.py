"""File watcher: filesystem events to targeted scans (PLAN.md §7.8).

One :class:`RootWatcher` per watched source root turns ``watchfiles`` events
into debounced *path hints* (:class:`HintBatch`), filters them through the
same ignore rules as the scanner and hands them to a scan callback. Watching
only accelerates updates; the scheduled scans (task 25) guarantee convergence.
Native inotify/FSEvents/ReadDirectoryChangesW is the default backend; polling
is forced on unreliable filesystems, when configured, or when the Linux
inotify watch limit is too small for the root. Crashes and a missing root
leave the watcher ``down`` with exponential backoff and a catch-up scan
request once it recovers.

The worker registers one background thread per watched root via
:func:`register_watchers` and publishes the per-root state to
``watchers.json`` in the user state directory every 30 s so
``mynotes watch status`` can report it from another process.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import platformdirs
from watchfiles import Change, DefaultFilter, watch

from mynotes.config import RootConfig, Settings, get_settings
from mynotes.platform.fsinfo import WatchBackend, inotify_watch_limit, watch_backend_for
from mynotes.platform.paths import to_rel_path
from mynotes.scan.ignore import IgnoreRules
from mynotes.scan.scanner import scan_root

logger = logging.getLogger(__name__)

WATCHERS: dict[str, RootWatcher] = {}
"""Worker-side registry of running root watchers, keyed by root name."""

HEARTBEAT_INTERVAL_S = 30.0
"""How often the worker refreshes the ``watchers.json`` state file."""

STALE_AFTER_S = 120.0
"""``watch status`` flags state entries older than this as stale."""

_RETRY_DELAY_S = 10.0
_RETRY_POLL_S = 0.5
_MONITOR_INTERVAL_S = 1.0
_MISSING_POLL_S = 1.0


@dataclass
class HintBatch:
    """Debounced watcher hints for one source root.

    Attributes:
        root_id: The ``source_root.id`` the paths belong to.
        rel_paths: Relevant paths relative to the root (POSIX, NFC); empty
            when ``overflow`` is set.
        overflow: True when the batch exceeded ``max_batch_paths``; the
            handler must scan the whole root instead of the single paths.
    """

    root_id: int
    rel_paths: set[str]
    overflow: bool = False


def batch_changes(
    root: Path,
    root_id: int,
    changes: Iterable[tuple[Change, str]],
    rules: IgnoreRules,
    max_batch_paths: int,
) -> HintBatch | None:
    """Turn raw watcher changes into one hint batch (pure, no side effects).

    Added, modified and deleted paths are all treated as "look at this path";
    ``stat_paths`` expands directories and reports deletions as missing. A
    deletion of a directory yields the directory path itself.

    Args:
        root: The source root the watched paths live under.
        root_id: The ``source_root.id`` recorded on the batch.
        changes: ``(Change, absolute path)`` pairs as yielded by
            ``watchfiles.watch``.
        rules: Ignore rules anchored at ``root``.
        max_batch_paths: Batch size limit; more distinct relevant paths than
            this empty the batch and set ``overflow``.

    Returns:
        The hint batch, or None when every change was irrelevant (outside the
        root or ignored).
    """
    rel_paths: set[str] = set()
    for _change, raw_path in changes:
        rel = _relevant_rel(root, raw_path, rules)
        if rel is None or rel in rel_paths:
            continue
        rel_paths.add(rel)
        if len(rel_paths) > max_batch_paths:
            return HintBatch(root_id, set(), overflow=True)
    if not rel_paths:
        return None
    return HintBatch(root_id, rel_paths)


def _relevant_rel(root: Path, raw_path: str, rules: IgnoreRules) -> str | None:
    """Return the canonical rel_path for a change, or None when irrelevant.

    Paths outside ``root`` (including relative ones) and paths ignored as a
    file or as a directory are dropped; the root itself (``.``) is dropped
    because a targeted scan of it is an incremental scan's job.
    """
    try:
        rel = to_rel_path(root, Path(raw_path))
    except ValueError:
        return None
    if not rel or rel == ".":
        return None
    if rules.is_ignored(rel, False) or rules.is_ignored(rel, True):
        return None
    return rel


class MynotesFilter(DefaultFilter):
    """``watchfiles`` filter that also applies the scanner's ignore rules."""

    def __init__(self, rules: IgnoreRules) -> None:
        """Store the rules the filter consults for every raw event.

        Args:
            rules: Ignore rules anchored at the watched root.
        """
        super().__init__()
        self._rules = rules

    def __call__(self, change: Change, path: str) -> bool:
        """Drop ignored paths and paths outside the root, keep the rest."""
        if not super().__call__(change, path):
            return False
        return _relevant_rel(self._rules.root, path, self._rules) is not None


def handle_batch(batch: HintBatch) -> None:
    """Default ``on_batch``: scan exactly what the hints ask for.

    Overflow batches request an incremental scan of the whole root, others a
    targeted scan of the sorted rel paths (``scan_root`` handles settle time,
    delete grace and the advisory lock). When the lock is busy a scheduled
    scan is running, the batch is merged into an in-memory retry list and
    re-submitted after :data:`_RETRY_DELAY_S` by :func:`retry_deferred`.
    Files the scan skipped as still being written (modified less than
    ``settle_seconds`` ago) are re-submitted the same way once they have had
    time to settle, so a new or just-saved file doesn't wait for the next
    scheduled scan.
    """
    if batch.overflow:
        summary = scan_root(batch.root_id, kind="incremental", trigger="watch")
    else:
        summary = scan_root(
            batch.root_id, kind="targeted", trigger="watch", rel_paths=sorted(batch.rel_paths)
        )
    if summary.counts.get("skipped_locked"):
        logger.info(
            "scan of root %d skipped: lock busy; retrying batch in %.0fs",
            batch.root_id,
            _RETRY_DELAY_S,
        )
        _defer_retry(batch)
    elif summary.unsettled:
        delay_s = get_settings().scan.settle_seconds + 1.0
        logger.info(
            "%d file(s) in root %d still being written; re-scanning in %.0fs",
            len(summary.unsettled),
            batch.root_id,
            delay_s,
        )
        _defer_retry(HintBatch(batch.root_id, set(summary.unsettled)), delay_s=delay_s)


@dataclass
class _PendingRetry:
    """A batch waiting for the advisory lock, merged per root."""

    rel_paths: set[str] = field(default_factory=set)
    overflow: bool = False
    due: float = 0.0


_PENDING: dict[int, _PendingRetry] = {}
_PENDING_LOCK = threading.Lock()
_RETRY_THREAD: threading.Thread | None = None
_RETRY_THREAD_LOCK = threading.Lock()


def _defer_retry(batch: HintBatch, *, delay_s: float = _RETRY_DELAY_S) -> None:
    """Merge ``batch`` into the pending retry list, due in ``delay_s`` (10 s by default)."""
    due = time.monotonic() + delay_s
    with _PENDING_LOCK:
        pending = _PENDING.get(batch.root_id)
        if pending is None:
            _PENDING[batch.root_id] = _PendingRetry(set(batch.rel_paths), batch.overflow, due)
        else:
            pending.rel_paths |= batch.rel_paths
            pending.overflow = pending.overflow or batch.overflow
            pending.due = min(pending.due, due)
    logger.debug("batch for root %d queued for retry", batch.root_id)


def _take_due_batch() -> HintBatch | None:
    """Pop the oldest due pending batch, or None when nothing is due."""
    now = time.monotonic()
    with _PENDING_LOCK:
        for root_id, pending in _PENDING.items():
            if pending.due <= now:
                del _PENDING[root_id]
                return HintBatch(root_id, pending.rel_paths, pending.overflow)
    return None


def retry_deferred(stop: threading.Event, *, interval_s: float = _RETRY_POLL_S) -> None:
    """Re-submit hint batches whose advisory-lock retry is due, until ``stop``."""
    while not stop.is_set():
        batch = _take_due_batch()
        if batch is None:
            stop.wait(interval_s)
            continue
        try:
            handle_batch(batch)
        except Exception:
            logger.exception("retry scan of root %d failed; deferring again", batch.root_id)
            _defer_retry(batch)


def _ensure_retry_thread(stop: threading.Event) -> None:
    """Start the shared retry loop once per process."""
    global _RETRY_THREAD
    with _RETRY_THREAD_LOCK:
        if _RETRY_THREAD is not None and _RETRY_THREAD.is_alive():
            return
        _RETRY_THREAD = threading.Thread(
            target=retry_deferred, args=(stop,), name="mynotes-watch-retry", daemon=True
        )
        _RETRY_THREAD.start()


class _ChainedEvent:
    """Duck-typed ``threading.Event`` set when any wrapped event is set.

    ``watchfiles`` only calls ``is_set()`` on the stop event, so this chains
    the worker's stop event with the watcher's internal restart event.
    """

    def __init__(self, *events: threading.Event) -> None:
        self._events = events

    def is_set(self) -> bool:
        return any(event.is_set() for event in self._events)


def _count_directories(root: Path) -> int:
    """Count the directories under ``root`` (not following symlinks)."""
    count = 0
    for _dirpath, dirs, _files in os.walk(root):
        count += len(dirs)
    return count


class RootWatcher:
    """Watches one source root and emits debounced hint batches.

    The watcher never scans itself: batches go to ``on_batch`` (by default
    :func:`handle_batch`). It survives crashes with exponential backoff
    (5 s to 5 min) and a catch-up overflow batch once it recovers, and it
    waits for a missing root to come back instead of spinning.
    """

    _BACKOFF_START_S = 5.0
    _BACKOFF_MAX_S = 300.0

    def __init__(
        self,
        root_id: int,
        root_path: Path,
        settings: Settings,
        on_batch: Callable[[HintBatch], None] | None = None,
    ) -> None:
        """Store the root configuration and resolve the watch backend.

        Args:
            root_id: The ``source_root.id`` of the root.
            root_path: The absolute path of the root.
            settings: Application settings (``scan.watch`` plus the per-root
                overrides in ``roots``).
            on_batch: Batch callback; defaults to :func:`handle_batch`.
        """
        self.root_id = root_id
        self.root_path = Path(root_path)
        self.settings = settings
        self.on_batch = on_batch if on_batch is not None else handle_batch
        root_config = _root_config_for(settings, self.root_path)
        configured = (
            root_config.watch_backend if root_config is not None else None
        ) or settings.scan.watch.backend
        self.backend: WatchBackend = watch_backend_for(self.root_path, configured)
        self.state: Literal["starting", "native", "polling", "down", "disabled"] = "starting"
        self._internal_stop = threading.Event()
        self._catchup_pending = False
        self._backend_resolved = False
        self._backoff_s = self._BACKOFF_START_S

    def run(self, stop: threading.Event) -> None:
        """Watch the root until ``stop`` is set, recovering from failures.

        Args:
            stop: Event set by the worker on shutdown; the method returns
                soon after it is set.
        """
        logger.info(
            "watcher for root %d starting on %s (backend %s)",
            self.root_id,
            self.root_path,
            self.backend,
        )
        self._set_state("starting")
        _ensure_retry_thread(stop)
        rules = IgnoreRules(self.root_path, self.settings.scan.ignore)
        watch_filter = MynotesFilter(rules)
        helpers = [
            threading.Thread(
                target=self._monitor_loop,
                args=(stop,),
                name=f"watch-monitor-{self.root_id}",
                daemon=True,
            ),
            threading.Thread(
                target=self._heartbeat_loop,
                args=(stop,),
                name=f"watch-heartbeat-{self.root_id}",
                daemon=True,
            ),
        ]
        for helper in helpers:
            helper.start()
        try:
            while not stop.is_set():
                if not self.root_path.is_dir():
                    self._set_state("down")
                    self._catchup_pending = True
                    if self._wait_for_root(stop):
                        break
                    continue
                if not self._backend_resolved:
                    self._apply_backend()
                self._set_state(self.backend)
                if self._catchup_pending:
                    self._catchup_pending = False
                    self._emit_catchup()
                self._internal_stop.clear()
                started = time.monotonic()
                try:
                    self._watch_loop(stop, rules, watch_filter)
                except Exception:
                    logger.exception("watcher for root %d crashed; backing off", self.root_id)
                    self._set_state("down")
                    self._catchup_pending = True
                    if not self._backoff_wait(stop):
                        break
                    continue
                if time.monotonic() - started > self._BACKOFF_START_S:
                    self._backoff_s = self._BACKOFF_START_S
                if stop.is_set():
                    break
                logger.warning(
                    "watch on %s ended unexpectedly; treating root as down", self.root_path
                )
                self._set_state("down")
                self._catchup_pending = True
        finally:
            self._internal_stop.set()
        logger.info("watcher for root %d stopped", self.root_id)

    def _watch_loop(
        self, stop: threading.Event, rules: IgnoreRules, watch_filter: MynotesFilter
    ) -> None:
        """Run the ``watchfiles`` generator until it or ``stop`` ends it."""
        watch_config = self.settings.scan.watch
        for changes in watch(
            self.root_path,
            watch_filter=watch_filter,
            debounce=watch_config.debounce_ms,
            step=50,
            stop_event=_ChainedEvent(self._internal_stop, stop),
            force_polling=(self.backend == "polling"),
            poll_delay_ms=watch_config.poll_delay_ms,
            recursive=True,
            raise_interrupt=False,
            yield_on_timeout=False,
        ):
            if not self.root_path.is_dir():
                return
            batch = batch_changes(
                self.root_path, self.root_id, changes, rules, watch_config.max_batch_paths
            )
            if batch is not None:
                self._dispatch(batch)

    def _dispatch(self, batch: HintBatch) -> None:
        """Hand one batch to ``on_batch``, deferring a retry when it fails."""
        try:
            self.on_batch(batch)
        except Exception:
            logger.exception("batch handler failed for root %d; deferring retry", self.root_id)
            _defer_retry(batch)

    def _emit_catchup(self) -> None:
        """Request a catch-up scan of the whole root after a failure."""
        logger.info("watcher for root %d requesting catch-up scan", self.root_id)
        try:
            self.on_batch(HintBatch(self.root_id, set(), overflow=True))
        except Exception:
            logger.exception("catch-up scan request failed for root %d", self.root_id)

    def _apply_backend(self) -> None:
        """Downgrade to polling when the Linux inotify limit is too small."""
        if self.backend == "native" and sys.platform == "linux":
            limit = inotify_watch_limit()
            if limit is not None:
                directories = _count_directories(self.root_path)
                if limit < directories:
                    logger.warning(
                        "root %s has %d directories but fs.inotify.max_user_watches"
                        " is %d; switching to polling (fix:"
                        " sudo sysctl fs.inotify.max_user_watches=524288)",
                        self.root_path,
                        directories,
                        limit,
                    )
                    self.backend = "polling"
        self._backend_resolved = True

    def _monitor_loop(self, stop: threading.Event) -> None:
        """Interrupt the blocked watch when the root disappears or we stop."""
        while not stop.wait(_MONITOR_INTERVAL_S):
            if self.state in ("native", "polling") and not self.root_path.is_dir():
                logger.warning("watch root %s disappeared; forcing restart", self.root_path)
                self._set_state("down")
                self._catchup_pending = True
                self._internal_stop.set()
        self._internal_stop.set()

    def _heartbeat_loop(self, stop: threading.Event) -> None:
        """Refresh the shared state file so ``watch status`` can spot a dead worker."""
        while not stop.wait(HEARTBEAT_INTERVAL_S):
            self._publish_state()

    def _wait_for_root(self, stop: threading.Event) -> bool:
        """Wait for the root to exist again; True when ``stop`` was set."""
        logger.warning("watch root %s is missing; waiting for it to come back", self.root_path)
        while not stop.is_set() and not self.root_path.is_dir():
            stop.wait(_MISSING_POLL_S)
        return stop.is_set()

    def _backoff_wait(self, stop: threading.Event) -> bool:
        """Sleep with exponential backoff (5 s to 5 min); False when stopped."""
        delay = min(self._backoff_s, self._BACKOFF_MAX_S)
        self._backoff_s = min(self._backoff_s * 2, self._BACKOFF_MAX_S)
        return not stop.wait(delay)

    def _set_state(
        self, state: Literal["starting", "native", "polling", "down", "disabled"]
    ) -> None:
        """Update the state and publish it to the shared file."""
        self.state = state
        self._publish_state()

    def _publish_state(self) -> None:
        publish_states()


def _root_config_for(settings: Settings, root_path: Path) -> RootConfig | None:
    """Return the configured root matching ``root_path``, or None."""
    wanted = os.path.normpath(str(root_path))
    for root_config in settings.roots:
        if os.path.normpath(str(root_config.path)) == wanted:
            return root_config
    return None


def _root_config_by_name(settings: Settings, name: str) -> RootConfig | None:
    """Return the configured root with the given name, or None."""
    for root_config in settings.roots:
        if root_config.name == name:
            return root_config
    return None


def watched_roots(settings: Settings) -> list[tuple[str, int, Path]]:
    """List the enabled source roots that should be watched.

    Args:
        settings: Application settings; the per-root ``watch`` flag
            (``roots[].watch``) wins over ``scan.watch.enabled``.

    Returns:
        ``(name, source_root.id, abs_path)`` tuples in root id order.
    """
    from sqlalchemy import select

    from mynotes.db.models import SourceRoot, Workspace
    from mynotes.db.session import session_scope

    stmt = (
        select(SourceRoot.id, SourceRoot.name, SourceRoot.abs_path)
        .join(Workspace, Workspace.id == SourceRoot.workspace_id)
        .where(
            Workspace.name == settings.workspace,
            SourceRoot.enabled.is_(True),
            SourceRoot.kind == "local",
        )
        .order_by(SourceRoot.id)
    )
    with session_scope() as session:
        rows = session.execute(stmt).all()
    watched: list[tuple[str, int, Path]] = []
    for root_id, name, abs_path in rows:
        root_config = _root_config_by_name(settings, str(name))
        enabled = root_config.watch if root_config is not None else None
        if enabled is None:
            enabled = settings.scan.watch.enabled
        if enabled:
            watched.append((str(name), int(root_id), Path(str(abs_path))))
    return watched


def _watch_target(
    root_name: str, root_id: int, root_path: Path
) -> Callable[[threading.Event], None]:
    """Build the background-thread target for one root's watcher."""

    def target(stop: threading.Event) -> None:
        watcher = RootWatcher(root_id, root_path, get_settings())
        WATCHERS[root_name] = watcher
        try:
            watcher.run(stop)
        finally:
            WATCHERS.pop(root_name, None)

    return target


def register_watchers() -> list[str]:
    """Register one ``watch:<name>`` background thread per watched root.

    The ``worker`` CLI calls this before starting the worker; each thread
    runs :meth:`RootWatcher.run` with the worker's stop event.

    Returns:
        The registered background thread names.
    """
    from mynotes.jobs.worker import register_background_thread

    settings = get_settings()
    names: list[str] = []
    for root_name, root_id, root_path in watched_roots(settings):
        thread_name = f"watch:{root_name}"
        register_background_thread(thread_name, _watch_target(root_name, root_id, root_path))
        names.append(thread_name)
    if names:
        logger.info("registered watcher thread(s): %s", ", ".join(names))
    return names


def watchers_state_file() -> Path:
    """Return the JSON file where the worker publishes watcher states."""
    return Path(platformdirs.user_state_dir("mynotes")) / "watchers.json"


_PUBLISH_LOCK = threading.Lock()
STATE_KEY = "watchers"
"""The ``app_state`` key the states are published under (the file is the fallback)."""


def publish_states() -> None:
    """Write every registered watcher's state to ``watchers.json`` atomically.

    ``since`` is the write time (UTC), refreshed by the workers' 30 s
    heartbeat, so readers can detect a dead worker. Best effort: write
    failures are logged, never raised.
    """
    with _PUBLISH_LOCK:
        now = datetime.now(UTC).isoformat()
        payload = {
            name: {"state": watcher.state, "backend": watcher.backend, "since": now}
            for name, watcher in sorted(WATCHERS.items())
        }
        path = watchers_state_file()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            logger.warning("could not write watcher state to %s", path, exc_info=True)
        try:  # the shared table lets a web process on another machine show the states
            from mynotes.db.state import set_state

            set_state(STATE_KEY, payload)
        except Exception:
            logger.debug("watcher state not written to app_state", exc_info=True)


def read_watchers_state() -> dict[str, Any]:
    """Read the worker-published watcher states.

    Returns:
        The parsed JSON mapping (root name to state entry), or an empty dict
        when the file is missing or invalid.
    """
    try:
        from mynotes.db.state import get_state

        shared = get_state(STATE_KEY)
    except Exception:
        shared = None
    if isinstance(shared, dict):
        return shared
    try:
        payload = json.loads(watchers_state_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_since(value: str) -> datetime | None:
    """Parse an ISO ``since`` timestamp; naive values are taken as UTC."""
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def watcher_status_rows(now: datetime | None = None) -> list[dict[str, Any]]:
    """Parse the state file into rows for ``mynotes watch status``.

    Args:
        now: Current time; defaults to the real clock (tests inject a fake).

    Returns:
        One row per state entry with ``root``, ``state``, ``backend``,
        ``since`` and ``stale`` (True when the entry is older than
        :data:`STALE_AFTER_S`, i.e. the worker is probably not running).
    """
    payload = read_watchers_state()
    current = now if now is not None else datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    for name in sorted(payload):
        entry = payload[name] if isinstance(payload[name], dict) else {}
        since_raw = str(entry.get("since", ""))
        since = _parse_since(since_raw)
        stale = since is None or (current - since).total_seconds() > STALE_AFTER_S
        rows.append(
            {
                "root": name,
                "state": str(entry.get("state", "unknown")),
                "backend": str(entry.get("backend", "unknown")),
                "since": since_raw or "(missing)",
                "stale": stale,
            }
        )
    return rows
