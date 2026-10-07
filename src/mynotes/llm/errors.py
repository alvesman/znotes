"""Exceptions raised by the shared model-endpoint client layer.

Both errors carry the endpoint name and, when known, the HTTP status. The
``message`` attribute holds at most 300 characters of the response body and
never includes request headers, so the exceptions are safe to log.
"""

from __future__ import annotations


class EndpointError(Exception):
    """Non-retryable HTTP error (4xx other than 429) or retries exhausted."""

    def __init__(self, endpoint: str, status: int | None, message: str) -> None:
        """Store the endpoint name, status code and truncated reason."""
        super().__init__(f"{endpoint}: {message}")
        self.endpoint = endpoint
        self.status = status
        self.message = message


class EndpointUnavailable(EndpointError):
    """Circuit breaker is open; the call was not attempted."""

    def __init__(
        self,
        endpoint: str,
        status: int | None = None,
        message: str = "circuit breaker is open",
    ) -> None:
        """Store the endpoint name; status is always None (nothing was sent)."""
        super().__init__(endpoint, status, message)
