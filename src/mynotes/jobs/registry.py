"""Job type priorities, handler registry and the per-job execution context.

Handlers are plain callables registered globally with the :func:`handler`
decorator (usually at import time of the module implementing them). The
worker only claims job types that have a registered handler, so job types
whose handlers arrive in later tasks simply stay queued.

Handlers must be idempotent: running the same job twice must not duplicate
rows, because crash recovery (:func:`mynotes.jobs.queue.reap_stale`) can
re-queue a job that was already executed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from mynotes.config import Settings
from mynotes.embed.client import Embedder, get_embedder


class Priority(IntEnum):
    """Job priorities; lower numbers are claimed first (see PLAN.md §7.7)."""

    EXTRACT = 10
    CHUNK = 20
    EMBED = 30
    VERSION_DETECT = 50
    TAG_LLM = 60
    KG_EXTRACT = 70
    RESOLVE_ENTITIES = 75
    SUMMARIES = 80
    MAINTENANCE = 90


class RetryLater(Exception):
    """Raised by a handler to requeue the job after a delay, without consuming an attempt."""

    def __init__(self, delay_s: float, reason: str) -> None:
        """Store the delay in seconds and a short reason for ``job.last_error``."""
        super().__init__(reason)
        self.delay_s = delay_s
        self.reason = reason


@dataclass
class JobContext:
    """Per-job execution context handed to handlers.

    ``embedder`` defaults to the process-wide embedder from
    :func:`mynotes.embed.client.get_embedder`; tests pass a fake instead.
    Later tasks add shared clients here (chat, ...).
    """

    worker_id: str
    settings: Settings
    embedder: Embedder = field(default_factory=get_embedder)


Handler = Callable[[dict[str, Any], JobContext], None]

_HANDLERS: dict[str, Handler] = {}


def handler(job_type: str) -> Callable[[Handler], Handler]:
    """Register the decorated function as the handler for ``job_type``.

    Args:
        job_type: The job type string stored in ``job.type``.

    Returns:
        A decorator that registers the function and returns it unchanged.
    """

    def decorator(fn: Handler) -> Handler:
        _HANDLERS[job_type] = fn
        return fn

    return decorator


def get_handler(job_type: str) -> Handler:
    """Return the handler registered for ``job_type``.

    Raises:
        KeyError: When no handler is registered for the job type.
    """
    try:
        return _HANDLERS[job_type]
    except KeyError:
        raise KeyError(f"no handler registered for job type {job_type!r}") from None


def registered_types() -> list[str]:
    """Return the sorted list of job types that currently have a handler."""
    return sorted(_HANDLERS)
