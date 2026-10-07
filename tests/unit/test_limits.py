"""Unit tests for the shared endpoint guards (limits, retries, breaker).

All tests use ``httpx.MockTransport`` and a fake clock/sleep, so no test
touches the network or really sleeps; the concurrency tests use real threads.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import datetime

import httpx
import pytest

from mynotes.config import EndpointLimits, LLMLimits, RetryConfig
from mynotes.llm.errors import EndpointError, EndpointUnavailable
from mynotes.llm.limits import CircuitBreaker, EndpointGuard, LLMGuard, RateLimiter, in_window

URL = "https://endpoint.test/v1/call"


class FakeClock:
    """Deterministic clock: ``sleep`` records the delay and advances time."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def status_handler(
    statuses: list[int],
    calls: list[httpx.Request],
    headers: dict[str, str] | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    """Build a MockTransport handler returning the given statuses in order.

    Once the list is exhausted, the last status repeats, so a single-entry
    list means "always this status".
    """
    state = {"index": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        index = min(state["index"], len(statuses) - 1)
        state["index"] += 1
        return httpx.Response(statuses[index], headers=headers, text="endpoint body")

    return handler


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_retries_transient_status_then_succeeds() -> None:
    """503 twice then 200: three transport calls, two recorded sleeps."""
    fake = FakeClock()
    calls: list[httpx.Request] = []
    guard = EndpointGuard(
        "embedding",
        EndpointLimits(concurrency=2),
        RetryConfig(max_attempts=4),
        clock=fake.clock,
        sleep=fake.sleep,
    )
    client = make_client(status_handler([503, 503, 200], calls))

    response = guard.request(client, "POST", URL)

    assert response.status_code == 200
    assert len(calls) == 3
    assert len(fake.sleeps) == 2
    assert all(delay > 0 for delay in fake.sleeps)


def test_retry_after_header_wins_over_backoff() -> None:
    """A 429 with 'Retry-After: 7' sleeps exactly 7 seconds, not exponentially."""
    fake = FakeClock()
    calls: list[httpx.Request] = []
    guard = EndpointGuard(
        "embedding",
        EndpointLimits(),
        RetryConfig(max_attempts=4, backoff_base_s=30),
        clock=fake.clock,
        sleep=fake.sleep,
    )
    client = make_client(status_handler([429, 200], calls, headers={"Retry-After": "7"}))

    response = guard.request(client, "POST", URL)

    assert response.status_code == 200
    assert fake.sleeps == [7.0]


def test_non_retryable_4xx_raises_after_one_call() -> None:
    """A 400 is our error: raised immediately, no retry, breaker untouched."""
    fake = FakeClock()
    calls: list[httpx.Request] = []
    guard = EndpointGuard(
        "embedding",
        EndpointLimits(),
        RetryConfig(max_attempts=4),
        clock=fake.clock,
        sleep=fake.sleep,
    )
    client = make_client(status_handler([400], calls))

    with pytest.raises(EndpointError) as exc_info:
        guard.request(client, "POST", URL)

    assert exc_info.value.endpoint == "embedding"
    assert exc_info.value.status == 400
    assert "endpoint body" in exc_info.value.message
    assert len(exc_info.value.message) <= 300
    assert len(calls) == 1
    assert fake.sleeps == []
    assert guard.breaker.state == "closed"


def test_retries_exhausted_raises_endpoint_error() -> None:
    """Always 503 with max_attempts=4: four calls, then 'retries exhausted'."""
    fake = FakeClock()
    calls: list[httpx.Request] = []
    guard = EndpointGuard(
        "embedding",
        EndpointLimits(),
        RetryConfig(max_attempts=4),
        clock=fake.clock,
        sleep=fake.sleep,
    )
    client = make_client(status_handler([503], calls))

    with pytest.raises(EndpointError, match="retries exhausted"):
        guard.request(client, "POST", URL)

    assert len(calls) == 4
    assert len(fake.sleeps) == 3


def test_circuit_breaker_opens_half_opens_and_closes() -> None:
    """Threshold 2: two exhausted calls open the breaker; a trial call closes it."""
    fake = FakeClock()
    calls: list[httpx.Request] = []
    mode = {"status": 503}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(mode["status"], text="endpoint body")

    breaker = CircuitBreaker(failure_threshold=2, cooldown_s=60, clock=fake.clock)
    guard = EndpointGuard(
        "llm",
        EndpointLimits(),
        RetryConfig(max_attempts=1),
        breaker=breaker,
        clock=fake.clock,
        sleep=fake.sleep,
    )
    client = make_client(handler)

    for _ in range(2):
        with pytest.raises(EndpointError, match="retries exhausted"):
            guard.request(client, "POST", URL)
    assert guard.breaker.state == "open"

    calls_before = len(calls)
    with pytest.raises(EndpointUnavailable):
        guard.request(client, "POST", URL)
    assert len(calls) == calls_before, "open breaker must not call the transport"

    fake.now += 61
    assert guard.breaker.state == "half_open"

    mode["status"] = 200
    response = guard.request(client, "POST", URL)
    assert response.status_code == 200
    assert guard.breaker.state == "closed"


def test_slot_bounds_concurrency() -> None:
    """20 threads with concurrency=3 never hold more than 3 slots at once."""
    guard = EndpointGuard("embedding", EndpointLimits(concurrency=3), RetryConfig(max_attempts=1))
    active = 0
    max_active = 0
    errors: list[Exception] = []
    lock = threading.Lock()

    def work() -> None:
        nonlocal active, max_active
        try:
            with guard.slot():
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.02)
                with lock:
                    active -= 1
        except Exception as exc:  # surfaced after join so thread failures fail the test
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert max_active <= 3


def test_rate_limiter_allows_burst_then_paces() -> None:
    """60 acquires run instantly, the 61st waits about one second in total."""
    fake = FakeClock()
    limiter = RateLimiter(60, clock=fake.clock, sleep=fake.sleep)

    for _ in range(60):
        limiter.acquire()
    assert fake.sleeps == []

    limiter.acquire()
    assert sum(fake.sleeps) > 0.9
    assert all(sleep < 2.0 for sleep in fake.sleeps)


def test_rate_limiter_without_limit_never_sleeps() -> None:
    """max_per_minute=None means no limit: acquire returns immediately."""
    fake = FakeClock()
    limiter = RateLimiter(None, clock=fake.clock, sleep=fake.sleep)

    for _ in range(1000):
        limiter.acquire()

    assert fake.sleeps == []


def test_llm_guard_background_yields_to_waiting_interactive() -> None:
    """With one pool slot each, background cannot enter while interactive waits."""
    guard = LLMGuard(
        LLMLimits(interactive_concurrency=1, background_concurrency=1), RetryConfig(max_attempts=1)
    )
    holder_has_slot = threading.Event()
    release_holder = threading.Event()
    interactive_acquired = threading.Event()
    background_acquired = threading.Event()
    order: list[str] = []
    order_lock = threading.Lock()
    errors: list[Exception] = []

    def record(name: str, event: threading.Event) -> None:
        with order_lock:
            order.append(name)
        event.set()

    def holder() -> None:
        try:
            with guard.slot("interactive"):
                holder_has_slot.set()
                release_holder.wait(5)
        except Exception as exc:  # surfaced after join so thread failures fail the test
            errors.append(exc)

    def interactive_waiter() -> None:
        try:
            with guard.slot("interactive"):
                record("interactive", interactive_acquired)
        except Exception as exc:  # surfaced after join so thread failures fail the test
            errors.append(exc)

    def background_worker() -> None:
        try:
            with guard.slot("background"):
                record("background", background_acquired)
        except Exception as exc:  # surfaced after join so thread failures fail the test
            errors.append(exc)

    holder_thread = threading.Thread(target=holder)
    holder_thread.start()
    assert holder_has_slot.wait(5), "holder never got the interactive slot"

    interactive_thread = threading.Thread(target=interactive_waiter)
    interactive_thread.start()
    deadline = time.monotonic() + 5
    while guard._waiting_interactive == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert guard._waiting_interactive >= 1, "interactive caller never started waiting"

    background_thread = threading.Thread(target=background_worker)
    background_thread.start()
    time.sleep(0.2)
    assert not background_acquired.is_set(), "background entered while interactive was waiting"

    release_holder.set()
    assert interactive_acquired.wait(5)
    interactive_thread.join(5)
    assert background_acquired.wait(5), "background never ran after interactive finished"
    holder_thread.join(5)
    background_thread.join(5)

    assert errors == []
    assert order.index("interactive") < order.index("background")


def test_llm_guard_background_allowed_now_uses_window() -> None:
    """background_allowed_now() delegates to in_window(); None window is always True."""
    guard = LLMGuard(LLMLimits(background_window="20:00-07:00"), RetryConfig(max_attempts=1))

    assert guard.background_allowed_now(datetime(2026, 9, 24, 23, 30)) is True
    assert guard.background_allowed_now(datetime(2026, 9, 24, 12, 0)) is False

    unlimited = LLMGuard(LLMLimits(), RetryConfig(max_attempts=1))
    assert unlimited.background_allowed_now(datetime(2026, 9, 24, 12, 0)) is True


def test_in_window() -> None:
    """Wrapped windows span midnight; plain ranges exclude the end; bad input raises."""
    assert in_window("20:00-07:00", datetime(2026, 9, 24, 23, 30)) is True
    assert in_window("20:00-07:00", datetime(2026, 9, 24, 12, 0)) is False
    assert in_window("20:00-07:00", datetime(2026, 9, 24, 6, 59)) is True
    assert in_window("20:00-07:00", datetime(2026, 9, 24, 7, 0)) is False
    assert in_window("09:00-17:00", datetime(2026, 9, 24, 10, 0)) is True
    assert in_window("09:00-17:00", datetime(2026, 9, 24, 17, 0)) is False
    with pytest.raises(ValueError):
        in_window("25:00-07:00", datetime(2026, 9, 24, 10, 0))
    with pytest.raises(ValueError):
        in_window("nine-to-five", datetime(2026, 9, 24, 10, 0))
    with pytest.raises(ValueError):
        in_window("", datetime(2026, 9, 24, 10, 0))


def test_snapshot_reports_calls_errors_and_state() -> None:
    """snapshot() counts both outcomes, the error rate and the breaker state."""
    fake = FakeClock()
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200 if len(calls) == 1 else 400, text="endpoint body")

    guard = EndpointGuard(
        "embedding",
        EndpointLimits(),
        RetryConfig(max_attempts=1),
        clock=fake.clock,
        sleep=fake.sleep,
    )
    client = make_client(handler)
    guard.request(client, "POST", URL)
    with pytest.raises(EndpointError):
        guard.request(client, "POST", URL)

    snapshot = guard.snapshot()

    assert snapshot.name == "embedding"
    assert snapshot.state == "closed"
    assert snapshot.calls == 2
    assert snapshot.error_rate == 0.5
    assert snapshot.last_error is not None
    assert "400" in snapshot.last_error
    assert snapshot.p50_ms is not None
    assert snapshot.p95_ms is not None


def test_llm_guard_request_uses_priority_and_kwargs() -> None:
    """LLMGuard.request routes through the chosen pool and forwards kwargs."""
    guard = LLMGuard(
        LLMLimits(interactive_concurrency=1, background_concurrency=1), RetryConfig(max_attempts=1)
    )
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200)

    client = make_client(handler)
    response = guard.request(client, "POST", URL, priority="background", json={"x": 1})

    assert response.status_code == 200
    assert seen == ["/v1/call"]


def test_transport_error_is_retried() -> None:
    """httpx.ConnectError counts as transient and is retried with backoff."""
    fake = FakeClock()
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200)

    guard = EndpointGuard(
        "embedding",
        EndpointLimits(),
        RetryConfig(max_attempts=2),
        clock=fake.clock,
        sleep=fake.sleep,
    )
    client = make_client(handler)
    response = guard.request(client, "POST", URL)

    assert response.status_code == 200
    assert len(calls) == 2
    assert len(fake.sleeps) == 1
    snapshot = guard.snapshot()
    assert snapshot.last_error is not None
    assert "ConnectError" in snapshot.last_error


def test_unknown_priority_raises() -> None:
    """slot() rejects priorities other than interactive/background."""
    guard = LLMGuard(LLMLimits(), RetryConfig(max_attempts=1))

    with pytest.raises(ValueError, match="priority"), guard.slot("urgent"):
        pass


def test_guard_singletons_and_helpers() -> None:
    """get_guard caches singletons; reset_guards drops them; names are checked."""
    from mynotes.llm import limits as limits_module

    limits_module.reset_guards()
    try:
        guard = limits_module.get_guard("embedding")
        assert limits_module.get_guard("embedding") is guard
        limits_module.reset_guards()
        assert limits_module.get_guard("embedding") is not guard

        snapshots = limits_module.all_snapshots()
        assert [snapshot.name for snapshot in snapshots] == ["embedding"]
    finally:
        limits_module.reset_guards()

    with pytest.raises(ValueError, match="unknown endpoint"):
        limits_module.get_guard("whisper")
