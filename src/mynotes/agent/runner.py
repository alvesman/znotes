"""``mynotes agent run``: the long-running agent process."""

from __future__ import annotations

import logging
import platform
import threading
import time
from pathlib import Path

import platformdirs

from mynotes import __version__
from mynotes.agent.client import AgentAuthError, ServerClient, ServerError
from mynotes.agent.hashcache import HashCache
from mynotes.agent.sync import AgentSync, SyncReport, _log_report
from mynotes.config import AgentConfig, Settings

logger = logging.getLogger(__name__)

_HEARTBEAT_S = 30.0
_BACKOFF_START_S = 5.0
_BACKOFF_MAX_S = 300.0


class AgentNotConfigured(RuntimeError):
    """``agent`` settings are missing."""


def agent_config(settings: Settings) -> AgentConfig:
    if settings.agent is None:
        raise AgentNotConfigured(
            "no agent settings: set agent.server_url, agent.root, agent.path and "
            "MYNOTES_AGENT_TOKEN (see mynotes.example.yaml)"
        )
    if settings.agent.token is None:
        raise AgentNotConfigured("agent.token is not set (MYNOTES_AGENT_TOKEN)")
    return settings.agent


def build_sync(settings: Settings) -> AgentSync:
    agent = agent_config(settings)
    assert agent.token is not None
    client = ServerClient(
        agent.server_url, agent.token.get_secret_value(), agent.root, verify_tls=agent.verify_tls
    )
    cache_file = (
        Path(platformdirs.user_state_dir("mynotes")) / f"agent-{agent.root}-hashes.json"
        if agent.hash_cache
        else None
    )
    return AgentSync(agent, settings.scan, client, HashCache(cache_file))


def sync_once(settings: Settings) -> SyncReport:
    """One full sync (``mynotes agent sync``)."""
    sync = build_sync(settings)
    try:
        sync.learn_limits()
        report = sync.full_sync()
        _log_report(report)
        return report
    finally:
        sync.client.close()


def run_agent(settings: Settings, stop: threading.Event) -> None:
    """Full sync, then watch and re-sync until ``stop`` is set."""
    sync = build_sync(settings)
    agent = sync.agent
    try:
        _with_backoff(stop, "connecting to the server", sync.learn_limits)
        logger.info("agent for root %r on %s, server %s", agent.root, sync.path, agent.server_url)
        _with_backoff(stop, "full sync", lambda: _log_report(sync.full_sync()))
        threads: list[threading.Thread] = []
        heartbeat = threading.Thread(
            target=_heartbeat_loop, args=(sync, stop), name="agent-heartbeat", daemon=True
        )
        heartbeat.start()
        threads.append(heartbeat)
        if agent.watch:
            from mynotes.scan.watch import RootWatcher

            watcher = RootWatcher(0, sync.path, settings, on_batch=sync.handle_batch)
            watch_thread = threading.Thread(
                target=watcher.run, args=(stop,), name="agent-watch", daemon=True
            )
            watch_thread.start()
            threads.append(watch_thread)
        interval = agent.scan_interval_minutes * 60.0
        while not stop.wait(interval):
            _with_backoff(stop, "periodic sync", lambda: _log_report(sync.full_sync()))
        for thread in threads:
            thread.join(timeout=5.0)
    finally:
        sync.client.close()


def _heartbeat_loop(sync: AgentSync, stop: threading.Event) -> None:
    info = {
        "agent_version": __version__,
        "hostname": platform.node(),
        "path": str(sync.path),
        "watch_state": "watching" if sync.agent.watch else "polling",
    }
    while not stop.is_set():
        try:
            sync.client.heartbeat(info)
        except Exception:
            logger.debug("heartbeat failed", exc_info=True)
        stop.wait(_HEARTBEAT_S)


def _with_backoff(stop: threading.Event, what: str, action) -> None:
    """Run ``action`` until it succeeds, waiting longer after each network or server error."""
    delay = _BACKOFF_START_S
    while not stop.is_set():
        try:
            action()
            return
        except AgentAuthError:
            raise
        except (ServerError, OSError) as exc:
            logger.warning("%s failed (%s); retrying in %.0f s", what, exc, delay)
        except Exception:
            logger.exception("%s failed; retrying in %.0f s", what, delay)
        if stop.wait(delay):
            return
        delay = min(delay * 2, _BACKOFF_MAX_S)


def wait_forever(stop: threading.Event) -> None:
    while not stop.is_set():
        time.sleep(0.5)
