"""Shared endpoint guards: rate limiting, retries, circuit breaker and stats.

Every model-endpoint client (tasks 06-07) sends its requests through an
:class:`EndpointGuard` obtained from :func:`get_guard`. The guard bounds
concurrency and request rate per endpoint, retries transient failures with
exponential backoff (honouring ``Retry-After``), opens a circuit breaker when
an endpoint keeps failing, and records per-endpoint statistics for
``mynotes status``.

All classes are thread-safe and synchronous; blocking uses
:class:`threading.Semaphore` and the injected ``sleep`` callable, never
asyncio. Tests inject a fake ``clock``/``sleep`` so nothing really waits.
"""

from __future__ import annotations

import email.utils
import math
import random
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import httpx

from mynotes.config import EndpointLimits, LLMLimits, RetryConfig, get_settings
from mynotes.llm.errors import EndpointError, EndpointUnavailable

type Clock = Callable[[], float]
type Sleep = Callable[[float], None]
type BreakerState = Literal["closed", "open", "half_open"]
type Priority = Literal["interactive", "background"]

_TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})
_BODY_EXCERPT_CHARS = 300
_STATS_CAPACITY = 200
_BACKGROUND_POLL_S = 0.05


def _truncate(text: str, limit: int = _BODY_EXCERPT_CHARS) -> str:
    """Return at most ``limit`` characters of ``text``."""
    return text if len(text) <= limit else text[:limit]


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header value (seconds or an HTTP date).

    Args:
        value: The raw header value, or None when absent.

    Returns:
        The delay in seconds (never negative), or None when missing/unparsable.
    """
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0.0, (parsed - datetime.now(UTC)).total_seconds())


class RateLimiter:
    """Token bucket limiting requests per minute; starts full, refills continuously.

    With ``max_per_minute=None`` :meth:`acquire` returns immediately. The
    bucket capacity is ``max_per_minute`` tokens refilling at
    ``max_per_minute / 60`` tokens per second, so a quiet endpoint allows an
    immediate burst and then paces requests evenly.
    """

    def __init__(
        self, max_per_minute: int | None, clock: Clock = time.monotonic, sleep: Sleep = time.sleep
    ):
        """Create a limiter for one endpoint.

        Args:
            max_per_minute: Requests allowed per minute, or None for no limit.
            clock: Monotonic time source.
            sleep: Blocking wait function.

        Raises:
            ValueError: If ``max_per_minute`` is not None and not positive.
        """
        if max_per_minute is not None and max_per_minute <= 0:
            raise ValueError("max_per_minute must be positive or None")
        self._max = max_per_minute
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._tokens = float(max_per_minute) if max_per_minute is not None else 0.0
        self._refill_per_s = (max_per_minute / 60.0) if max_per_minute is not None else 0.0
        self._updated = clock()

    def acquire(self) -> None:
        """Block (via ``sleep``) until a token is available; thread-safe."""
        if self._max is None:
            return
        while True:
            with self._lock:
                now = self._clock()
                self._tokens = min(
                    float(self._max), self._tokens + (now - self._updated) * self._refill_per_s
                )
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait_s = (1.0 - self._tokens) / self._refill_per_s
            self._sleep(wait_s)


class CircuitBreaker:
    """Opens after consecutive failures, half-opens after a cooldown.

    States: ``closed`` (calls allowed) -> ``open`` after
    ``failure_threshold`` consecutive failures -> ``half_open`` once
    ``cooldown_s`` elapsed -> a trial call then closes it on success or
    re-opens it on failure.
    """

    def __init__(
        self, failure_threshold: int = 5, cooldown_s: float = 60, clock: Clock = time.monotonic
    ):
        """Create a breaker; pass the same ``clock`` as the owning guard.

        Args:
            failure_threshold: Consecutive failures before opening.
            cooldown_s: Seconds before an open breaker becomes half-open.
            clock: Monotonic time source.
        """
        self._failure_threshold = failure_threshold
        self._cooldown_s = cooldown_s
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._state: BreakerState = "closed"
        self._opened_at = 0.0

    def _effective_state(self) -> BreakerState:
        if self._state == "open" and self._clock() - self._opened_at >= self._cooldown_s:
            return "half_open"
        return self._state

    @property
    def state(self) -> BreakerState:
        """The current breaker state, honouring the cooldown."""
        with self._lock:
            return self._effective_state()

    def allow(self) -> bool:
        """Whether a call may be attempted right now."""
        with self._lock:
            return self._effective_state() != "open"

    def record_success(self) -> None:
        """Record a successful call; closes the breaker and resets the count."""
        with self._lock:
            self._failures = 0
            self._state = "closed"

    def record_failure(self) -> None:
        """Record an exhausted call; opens the breaker once the threshold is hit."""
        with self._lock:
            if self._state == "open":
                self._opened_at = self._clock()
                return
            self._failures += 1
            if self._failures >= self._failure_threshold:
                self._state = "open"
                self._opened_at = self._clock()
                self._failures = 0


@dataclass
class EndpointSnapshot:
    """Point-in-time view of one endpoint for ``mynotes status``."""

    name: str
    state: str
    calls: int
    p50_ms: float | None
    p95_ms: float | None
    error_rate: float
    last_error: str | None


class EndpointStats:
    """Thread-safe record of call outcomes; keeps the last 200 latencies.

    ``calls`` and ``error_rate`` count every recorded outcome, while the
    p50/p95 percentiles are computed over the most recent 200 latencies.
    """

    def __init__(self, capacity: int = _STATS_CAPACITY):
        """Create an empty stats record.

        Args:
            capacity: How many recent latencies to keep for percentiles.
        """
        self._lock = threading.Lock()
        self._latencies: deque[float] = deque(maxlen=capacity)
        self._calls = 0
        self._errors = 0
        self._last_error: str | None = None

    def record(self, latency_ms: float, *, ok: bool, error: str | None = None) -> None:
        """Record one call outcome.

        Args:
            latency_ms: Round-trip latency in milliseconds.
            ok: Whether the call succeeded (2xx).
            error: Short error description for failed calls.
        """
        with self._lock:
            self._calls += 1
            self._latencies.append(latency_ms)
            if ok:
                return
            self._errors += 1
            if error:
                self._last_error = error

    def snapshot(self, name: str, state: str) -> EndpointSnapshot:
        """Build a snapshot labelled with the endpoint ``name`` and breaker ``state``."""
        with self._lock:
            ordered = sorted(self._latencies)
            calls = self._calls
            error_rate = self._errors / calls if calls else 0.0
            return EndpointSnapshot(
                name=name,
                state=state,
                calls=calls,
                p50_ms=_percentile(ordered, 0.50),
                p95_ms=_percentile(ordered, 0.95),
                error_rate=error_rate,
                last_error=self._last_error,
            )


def _percentile(ordered: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile of an already sorted list."""
    if not ordered:
        return None
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


class EndpointGuard:
    """Limits, retries and observes the calls to one model endpoint.

    ``slot()`` bounds concurrency (semaphore) and pace (token bucket);
    :meth:`request` wraps an ``httpx`` call with retries on 429/5xx and
    transport errors, exponential backoff honouring ``Retry-After``, and
    circuit-breaker bookkeeping. Retry waits happen outside the slot so a
    backing-off thread never blocks other callers.
    """

    def __init__(
        self,
        name: str,
        limits: EndpointLimits | LLMLimits,
        retry: RetryConfig,
        breaker: CircuitBreaker | None = None,
        clock: Clock = time.monotonic,
        sleep: Sleep = time.sleep,
    ):
        """Create a guard for one endpoint.

        Args:
            name: Endpoint identifier used in errors and snapshots.
            limits: Concurrency, rate and timeout limits for the endpoint.
            retry: Retry and backoff policy.
            breaker: Optional pre-built breaker; a default one is created otherwise.
            clock: Monotonic time source (inject a fake in tests).
            sleep: Blocking wait function (inject a fake in tests).
        """
        self.name = name
        self.limits = limits
        self.retry = retry
        self.breaker = breaker if breaker is not None else CircuitBreaker(clock=clock)
        self._clock = clock
        self._sleep = sleep
        self._stats = EndpointStats()
        self._limiter = RateLimiter(limits.max_requests_per_minute, clock=clock, sleep=sleep)
        self._init_concurrency(limits)

    def _init_concurrency(self, limits: EndpointLimits | LLMLimits) -> None:
        """Create the concurrency gate; overridden by :class:`LLMGuard`."""
        self._slots = threading.Semaphore(limits.concurrency)  # type: ignore[union-attr]

    @contextmanager
    def slot(self) -> Iterator[None]:
        """Hold one concurrency slot plus one rate-limiter token."""
        self._slots.acquire()
        try:
            self._limiter.acquire()
            yield
        finally:
            self._slots.release()

    def request(self, client: httpx.Client, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Send one request with retries; see :meth:`_request_with_retries`."""
        return self._request_with_retries(client, method, url, slot_kwargs={}, **kwargs)

    def _request_with_retries(
        self,
        client: httpx.Client,
        method: str,
        url: str,
        *,
        slot_kwargs: dict[str, Any],
        **kwargs: Any,
    ) -> httpx.Response:
        """Run the retry loop shared by :class:`EndpointGuard` and :class:`LLMGuard`."""
        if not self.breaker.allow():
            raise EndpointUnavailable(self.name)
        last_reason = "no attempt was made"
        last_status: int | None = None
        for attempt in range(1, self.retry.max_attempts + 1):
            response, failure, latency_ms = self._send_one(
                client, method, url, slot_kwargs, **kwargs
            )
            if response is not None and failure is None:
                if response.is_success:
                    self._stats.record(latency_ms, ok=True)
                    self.breaker.record_success()
                    return response
                detail = _truncate(response.text)
                failure = f"HTTP {response.status_code}: {detail}"
                if response.status_code not in _TRANSIENT_STATUSES:
                    self._stats.record(latency_ms, ok=False, error=failure)
                    raise EndpointError(self.name, response.status_code, detail)
            self._stats.record(latency_ms, ok=False, error=failure)
            last_reason = failure or "unknown failure"
            if response is not None:
                last_status = response.status_code
            if attempt < self.retry.max_attempts:
                self._sleep(self._delay_for(attempt, response))
        self.breaker.record_failure()
        raise EndpointError(self.name, last_status, f"retries exhausted: {last_reason}")

    def _send_one(
        self,
        client: httpx.Client,
        method: str,
        url: str,
        slot_kwargs: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[httpx.Response | None, str | None, float]:
        """Send once inside a slot; return (response, failure, latency_ms)."""
        with self.slot(**slot_kwargs):
            started = self._clock()
            try:
                response = client.request(method, url, timeout=self.limits.timeout_s, **kwargs)
            except httpx.TransportError as exc:
                return (
                    None,
                    _truncate(f"{type(exc).__name__}: {exc}"),
                    (self._clock() - started) * 1000.0,
                )
            return response, None, (self._clock() - started) * 1000.0

    def _delay_for(self, attempt: int, response: httpx.Response | None) -> float:
        """Wait before retrying: ``Retry-After`` when honoured, else capped backoff with jitter."""
        if response is not None and self.retry.honor_retry_after:
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            if retry_after is not None:
                return min(self.retry.backoff_max_s, retry_after)
        base = min(self.retry.backoff_max_s, self.retry.backoff_base_s * 2 ** (attempt - 1))
        return base + random.uniform(0, 0.5)

    def snapshot(self) -> EndpointSnapshot:
        """Current stats and breaker state for this endpoint."""
        return self._stats.snapshot(self.name, self.breaker.state)


class LLMGuard(EndpointGuard):
    """EndpointGuard with separate interactive and background slot pools.

    A background call never starts while an interactive caller is waiting
    for a slot, so ``ask`` stays responsive during background work. When
    ``max_total_concurrency`` is set, a shared semaphore additionally caps
    both pools together. The guard never waits for the background time
    window; callers check :meth:`background_allowed_now` and re-queue.
    """

    def __init__(
        self,
        limits: LLMLimits,
        retry: RetryConfig,
        breaker: CircuitBreaker | None = None,
        clock: Clock = time.monotonic,
        sleep: Sleep = time.sleep,
    ):
        """Create the LLM guard with its two priority pools.

        Args:
            limits: Interactive/background concurrency and window settings.
            retry: Retry and backoff policy.
            breaker: Optional pre-built breaker; a default one is created otherwise.
            clock: Monotonic time source (inject a fake in tests).
            sleep: Blocking wait function (inject a fake in tests).
        """
        super().__init__("llm", limits, retry, breaker=breaker, clock=clock, sleep=sleep)
        self._interactive_slots = threading.Semaphore(limits.interactive_concurrency)
        self._background_slots = threading.Semaphore(limits.background_concurrency)
        self._total_slots = (
            threading.Semaphore(limits.max_total_concurrency)
            if limits.max_total_concurrency
            else None
        )
        self._waiting_interactive = 0
        self._waiting_lock = threading.Lock()

    def _init_concurrency(self, limits: EndpointLimits | LLMLimits) -> None:
        """No single pool; the priority pools are set up in :meth:`__init__`."""

    @contextmanager
    def slot(self, priority: Priority = "interactive") -> Iterator[None]:
        """Hold a pool slot (plus shared and rate-limiter capacity) for one call."""
        if priority == "interactive":
            pool = self._interactive_slots
            with self._waiting_lock:
                self._waiting_interactive += 1
            try:
                pool.acquire()
            finally:
                with self._waiting_lock:
                    self._waiting_interactive -= 1
        elif priority == "background":
            pool = self._background_slots
            while True:
                with self._waiting_lock:
                    waiting = self._waiting_interactive
                if waiting == 0:
                    break
                self._sleep(_BACKGROUND_POLL_S)
            pool.acquire()
        else:
            raise ValueError(f"unknown priority: {priority!r}")
        if self._total_slots is not None:
            self._total_slots.acquire()
        try:
            self._limiter.acquire()
            yield
        finally:
            if self._total_slots is not None:
                self._total_slots.release()
            pool.release()

    def request(
        self,
        client: httpx.Client,
        method: str,
        url: str,
        *,
        priority: Priority = "interactive",
        **kwargs: Any,
    ) -> httpx.Response:
        """Send one LLM request with retries through the given priority pool."""
        return self._request_with_retries(
            client, method, url, slot_kwargs={"priority": priority}, **kwargs
        )

    def background_allowed_now(self, now: datetime | None = None) -> bool:
        """Whether background LLM work may start at ``now`` (local time).

        Args:
            now: The moment to check; defaults to the current local time.

        Returns:
            True when no ``background_window`` is configured or ``now`` is
            inside the configured '20:00-07:00' style window.
        """
        if self.limits.background_window is None:
            return True
        return in_window(
            self.limits.background_window, now if now is not None else datetime.now().astimezone()
        )


_WINDOW_RE = re.compile(r"(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})")


def in_window(window: str, now: datetime) -> bool:
    """Whether ``now`` falls inside a '20:00-07:00' style local-time window.

    Args:
        window: 'HH:MM-HH:MM' local-time window; a wrapped range
            (``start > end``) spans midnight. ``start == end`` means all day.
        now: The moment to test; only the local hour and minute are used.

    Returns:
        True when ``now`` is inside the window.

    Raises:
        ValueError: If ``window`` is not a valid 'HH:MM-HH:MM' range.
    """
    match = _WINDOW_RE.fullmatch(window.strip())
    if match is None:
        raise ValueError(f"bad window format: {window!r} (expected 'HH:MM-HH:MM')")
    start_h, start_m, end_h, end_m = (int(group) for group in match.groups())
    if start_h > 23 or end_h > 23 or start_m > 59 or end_m > 59:
        raise ValueError(f"bad window time: {window!r}")
    start = start_h * 60 + start_m
    end = end_h * 60 + end_m
    minute = now.hour * 60 + now.minute
    if start == end:
        return True
    if start < end:
        return start <= minute < end
    return minute >= start or minute < end


_GUARDS: dict[str, EndpointGuard] = {}
_GUARDS_LOCK = threading.Lock()


def _build_guard(name: str) -> EndpointGuard:
    """Build the guard for ``name`` from the current settings."""
    settings = get_settings()
    retry = settings.models.retry
    if name == "llm":
        return LLMGuard(settings.models.llm.limits, retry)
    if name == "embedding":
        return EndpointGuard(name, settings.models.embedding.limits, retry)
    if name == "reranker":
        return EndpointGuard(name, settings.models.reranker.limits, retry)
    raise ValueError(f"unknown endpoint: {name!r} (expected 'embedding', 'reranker' or 'llm')")


def get_guard(name: Literal["embedding", "reranker", "llm"]) -> EndpointGuard:
    """Return the process-wide guard singleton for ``name``, built on first use."""
    with _GUARDS_LOCK:
        guard = _GUARDS.get(name)
        if guard is None:
            guard = _build_guard(name)
            _GUARDS[name] = guard
        return guard


def reset_guards() -> None:
    """Drop the cached guards (used by tests and config reloads)."""
    with _GUARDS_LOCK:
        _GUARDS.clear()


def all_snapshots() -> list[EndpointSnapshot]:
    """Snapshots for every guard created so far in this process."""
    with _GUARDS_LOCK:
        guards = list(_GUARDS.values())
    return [guard.snapshot() for guard in guards]
