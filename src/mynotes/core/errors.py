"""Domain errors raised by the facade and mapped to HTTP status codes by the API.

``NotFound`` and ``Ambiguous`` describe reference resolution problems
(unknown, or several candidates); ``Forbidden`` denies an access outside the
principal's path prefixes. They carry user-safe messages and never leak
internals.
"""

from __future__ import annotations

from typing import Any


class NotFound(LookupError):
    """The requested document, tag, version relation or entity does not exist."""


class Ambiguous(Exception):
    """A reference matched several documents or entities.

    Attributes:
        candidates: The matching items as JSON-safe dicts (at least
            ``id``/``document_id``, ``title``/``name`` and ``path`` where
            applicable) so the caller can retry with a specific one.
    """

    def __init__(self, message: str, candidates: list[dict[str, Any]]) -> None:
        """Store the message and the candidate list."""
        super().__init__(message)
        self.candidates = candidates


class Forbidden(PermissionError):
    """The principal's path prefixes do not cover the requested document."""
