"""Structured output: validated Pydantic objects from JSON-schema prompts.

Calls a :class:`~mynotes.llm.chat.ChatModel` with a vLLM ``response_format``
JSON schema, extracts the JSON from the answer (bare, fenced or wrapped in
prose), validates it against the Pydantic schema and makes exactly one repair
retry (feeding the bad output back) before raising
:class:`StructuredOutputError`. When the server rejects ``response_format``
(HTTP 400 mentioning it), the client is remembered and later calls rely on the
prompt plus the repair path only.
"""

from __future__ import annotations

import logging
import re
import weakref
from collections.abc import Sequence

from pydantic import BaseModel, ValidationError

from mynotes.llm.chat import ChatMessage, ChatModel, ChatResult
from mynotes.llm.errors import EndpointError
from mynotes.llm.limits import Priority

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_RAW_TEXT_CHARS = 500
_ERROR_CHARS = 200

_REJECTED_RESPONSE_FORMAT: weakref.WeakSet[ChatModel] = weakref.WeakSet()


class StructuredOutputError(Exception):
    """The model could not produce schema-valid JSON, even after one repair."""

    def __init__(self, raw_text: str, error: str) -> None:
        """Store the truncated raw answer and the validation error message."""
        super().__init__(f"invalid structured output: {error}")
        self.raw_text = raw_text
        self.error = error


def _shorten(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` characters."""
    return text[:limit]


def _validate[T: BaseModel](text: str, schema: type[T]) -> tuple[T | None, str]:
    """Try the whole text, then the first ``{...}`` block, against ``schema``.

    Returns:
        ``(instance, "")`` on success, else ``(None, validation error message)``.
    """
    candidates = [text.strip()]
    match = _JSON_BLOCK_RE.search(text)
    if match is not None and match.group(0) != candidates[0]:
        candidates.append(match.group(0))
    error = "no JSON object found in the answer"
    for candidate in candidates:
        try:
            return schema.model_validate_json(candidate), ""
        except ValidationError as exc:
            error = str(exc)
    return None, error


def _complete[T: BaseModel](
    chat: ChatModel,
    messages: list[ChatMessage],
    schema: type[T],
    *,
    priority: Priority,
    max_tokens: int,
) -> ChatResult:
    """Call ``chat.complete`` with the schema directive, falling back once.

    The fallback (and the memory that ``response_format`` is unsupported)
    triggers when the server answers HTTP 400 mentioning ``response_format``.
    """
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
    }
    if chat in _REJECTED_RESPONSE_FORMAT:
        return chat.complete(
            messages, priority=priority, max_tokens=max_tokens, response_format=None
        )
    try:
        return chat.complete(
            messages, priority=priority, max_tokens=max_tokens, response_format=response_format
        )
    except EndpointError as exc:
        if exc.status != 400 or "response_format" not in exc.message:
            raise
        _REJECTED_RESPONSE_FORMAT.add(chat)
        logger.warning(
            "chat endpoint rejected response_format; "
            "retrying without it and relying on the prompt + repair (remembered for this client)"
        )
        return chat.complete(
            messages, priority=priority, max_tokens=max_tokens, response_format=None
        )


def complete_structured[T: BaseModel](
    chat: ChatModel,
    messages: Sequence[ChatMessage],
    schema: type[T],
    *,
    priority: Priority = "background",
    max_tokens: int = 4096,
) -> T:
    """Return a ``schema`` instance validated from the model's JSON answer.

    Args:
        chat: The chat model to call.
        messages: The conversation so far (must ask for the JSON output).
        schema: Pydantic model the answer must validate against.
        priority: Guard pool for the calls (defaults to ``background``).
        max_tokens: Output cap passed to every call.

    Returns:
        One validated ``schema`` instance.

    Raises:
        StructuredOutputError: When the answer (and the single repair retry)
            still fails to parse or validate.
        EndpointError: On HTTP failure from the chat endpoint.
    """
    original = list(messages)
    result = _complete(chat, original, schema, priority=priority, max_tokens=max_tokens)
    value, error = _validate(result.text, schema)
    if value is not None:
        return value
    repair_prompt = (
        "Your previous answer was not valid JSON for the schema. "
        f"Error: {_shorten(error, _ERROR_CHARS)}. Reply with only the corrected JSON."
    )
    repair_messages = [
        *original,
        ChatMessage(role="assistant", content=result.text),
        ChatMessage(role="user", content=repair_prompt),
    ]
    result = _complete(chat, repair_messages, schema, priority=priority, max_tokens=max_tokens)
    value, error = _validate(result.text, schema)
    if value is None:
        raise StructuredOutputError(raw_text=_shorten(result.text, _RAW_TEXT_CHARS), error=error)
    return value
