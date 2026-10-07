"""Chat client for the OpenAI-compatible ``/chat/completions`` endpoint.

Non-streaming and streaming completions against an OpenAI-compatible chat endpoint, with
thinking disabled through ``chat_template_kwargs`` when the configured model
supports it (``cfg.thinking_model``) and a one-time fallback when the server
rejects that field. Every HTTP call goes through the shared
:class:`~mynotes.llm.limits.LLMGuard` from task 05: interactive and background
calls use separate priority pools, transient failures are retried with backoff
and a circuit breaker protects the endpoint.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal, Protocol, cast

import httpx
from pydantic import SecretStr

from mynotes.config import LLMConfig, ModelsNotConfigured, RetryConfig, get_settings
from mynotes.llm.errors import EndpointError, EndpointUnavailable
from mynotes.llm.limits import LLMGuard, Priority, get_guard

logger = logging.getLogger(__name__)

_MAX_DEFAULT_TOKENS = 4096
_BODY_EXCERPT_CHARS = 300
_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class ChatMessage:
    """One chat message: a role and its text content."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class ChatResult:
    """A completed chat turn: answer text, finish reason and token usage."""

    text: str
    finish_reason: str | None
    usage: dict[str, int]


class ChatModel(Protocol):
    """Anything that can complete (and stream) a list of chat messages."""

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        priority: Priority = "interactive",
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict | None = None,
    ) -> ChatResult: ...

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        priority: Priority = "interactive",
        max_tokens: int | None = None,
    ) -> Iterator[str]: ...


def _chat_url(base_url: str) -> str:
    """Join ``base_url`` (with or without a trailing slash) with ``/chat/completions``."""
    return f"{base_url.rstrip('/')}/chat/completions"


def _auth_headers(api_key: SecretStr | None) -> dict[str, str]:
    """Build the request headers; ``Authorization`` only when a key is set."""
    if api_key is None:
        return {}
    key = api_key.get_secret_value()
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


def _strip_think_block(text: str) -> str:
    """Remove one leading ``<think>...</think>`` block (models that ignore the flag)."""
    return _THINK_BLOCK_RE.sub("", text, count=1)


def _excerpt(text: str) -> str:
    """Return at most 300 characters of ``text`` (safe to log or raise)."""
    return text[:_BODY_EXCERPT_CHARS]


def _backoff_delay(attempt: int, retry: RetryConfig) -> float:
    """Capped exponential backoff for the streaming retry loop."""
    return min(retry.backoff_max_s, retry.backoff_base_s * 2 ** (attempt - 1))


def _iter_sse_pieces(response: httpx.Response) -> Iterator[str]:
    """Yield ``choices[0].delta.content`` from SSE lines; stop at ``data: [DONE]``."""
    for line in response.iter_lines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload:
            continue
        if payload == "[DONE]":
            return
        try:
            event = json.loads(payload)
            chunk = (event.get("choices") or [{}])[0]
            content = (chunk.get("delta") or {}).get("content")
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            logger.debug("ignoring unparsable SSE chunk: %.200s", payload)
            continue
        if content:
            yield content


class ChatClient:
    """Client for one chat-completion endpoint; implements :class:`ChatModel`."""

    def __init__(
        self,
        cfg: LLMConfig,
        guard: LLMGuard | None = None,
        http: httpx.Client | None = None,
    ) -> None:
        """Build a client for one endpoint configuration.

        Args:
            cfg: Chat endpoint, model parameters and limits.
            guard: Shared LLM guard; defaults to ``get_guard("llm")``.
            http: HTTP client to use; a plain ``httpx.Client`` is created when None.
        """
        missing = [
            f"models.llm.{key}"
            for key, value in (("base_url", cfg.base_url), ("model", cfg.model))
            if not value
        ]
        if missing:
            raise ModelsNotConfigured(missing)
        self.cfg = cfg
        self.guard = guard if guard is not None else cast(LLMGuard, get_guard("llm"))
        self._owns_http = http is None
        self._http = http if http is not None else httpx.Client()
        self._url = _chat_url(cfg.base_url)
        self._sends_thinking_kwargs = cfg.thinking_model

    def close(self) -> None:
        """Close the owned HTTP client; an injected client is left open."""
        if self._owns_http:
            self._http.close()

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        priority: Priority = "interactive",
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict | None = None,
    ) -> ChatResult:
        """Send one chat completion and return the parsed result.

        Args:
            messages: The conversation so far.
            priority: Guard pool to use (``interactive`` or ``background``).
            max_tokens: Output cap; defaults to ``min(cfg.max_tokens, 4096)``.
                Callers pass larger values explicitly.
            temperature: Sampling temperature; defaults to ``cfg.temperature``.
            response_format: Optional vLLM JSON-schema directive, passed through.

        Returns:
            The completed turn with the ``<think>...</think>`` block stripped from the text.

        Raises:
            EndpointError: On HTTP failure or a malformed response body.
        """
        payload = self._payload(messages, max_tokens=max_tokens, temperature=temperature)
        if response_format is not None:
            payload["response_format"] = response_format
        headers = _auth_headers(self.cfg.api_key)
        try:
            response = self._post(payload, headers=headers, priority=priority)
        except EndpointError as exc:
            if not (
                exc.status == 400
                and self._sends_thinking_kwargs
                and "chat_template_kwargs" in exc.message
            ):
                raise
            self._disable_thinking_kwargs()
            payload.pop("chat_template_kwargs", None)
            response = self._post(payload, headers=headers, priority=priority)
        return self._parse_completion(response)

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        priority: Priority = "interactive",
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Stream one chat completion, yielding text pieces as they arrive.

        The priority slot is held for the whole stream. Retries (on transient
        HTTP failures, transport errors and a ``chat_template_kwargs``
        rejection) happen only before the first piece is yielded; a failure
        afterwards propagates as :class:`EndpointError`.

        Args:
            messages: The conversation so far.
            priority: Guard pool to use (``interactive`` or ``background``).
            max_tokens: Output cap; defaults to ``min(cfg.max_tokens, 4096)``.

        Yields:
            Non-empty ``delta.content`` pieces, in server order.

        Raises:
            EndpointError: When the stream cannot be opened after the retries,
                or breaks after the first piece was yielded.
        """
        payload = self._payload(messages, max_tokens=max_tokens)
        payload["stream"] = True
        headers = _auth_headers(self.cfg.api_key)
        guard = self.guard
        if not guard.breaker.allow():
            raise EndpointUnavailable(guard.name)
        last_failure = "no attempt was made"
        for attempt in range(1, guard.retry.max_attempts + 1):
            yielded = False
            try:
                with (
                    guard.slot(priority),
                    self._http.stream(
                        "POST",
                        self._url,
                        json=payload,
                        headers=headers,
                        timeout=guard.limits.timeout_s,
                    ) as response,
                ):
                    if not response.is_success:
                        body = _excerpt(response.read().decode("utf-8", errors="replace"))
                        if (
                            response.status_code == 400
                            and self._sends_thinking_kwargs
                            and "chat_template_kwargs" in body
                        ):
                            self._disable_thinking_kwargs()
                            payload.pop("chat_template_kwargs", None)
                            continue
                        if response.status_code not in _TRANSIENT_STATUSES:
                            guard.breaker.record_failure()
                            raise EndpointError(guard.name, response.status_code, body)
                        last_failure = f"HTTP {response.status_code}: {body}"
                    else:
                        for piece in _iter_sse_pieces(response):
                            yielded = True
                            yield piece
                        guard.breaker.record_success()
                        return
            except httpx.TransportError as exc:
                detail = _excerpt(f"{type(exc).__name__}: {exc}")
                if yielded:
                    raise EndpointError(guard.name, None, detail) from exc
                last_failure = detail
            if attempt < guard.retry.max_attempts:
                time.sleep(_backoff_delay(attempt, guard.retry))
        guard.breaker.record_failure()
        raise EndpointError(guard.name, None, f"retries exhausted: {last_failure}")

    def _payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int | None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Build the request body (thinking kwarg only for thinking models)."""
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": (
                max_tokens
                if max_tokens is not None
                else min(self.cfg.max_tokens, _MAX_DEFAULT_TOKENS)
            ),
        }
        if self._sends_thinking_kwargs:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    def _post(
        self, payload: dict[str, Any], *, headers: dict[str, str], priority: Priority
    ) -> httpx.Response:
        """Send one completion request through the guard with the priority pool."""
        return self.guard.request(
            self._http, "POST", self._url, json=payload, headers=headers, priority=priority
        )

    def _disable_thinking_kwargs(self) -> None:
        """Stop sending ``chat_template_kwargs``; warn once per client."""
        if self._sends_thinking_kwargs:
            self._sends_thinking_kwargs = False
            logger.warning(
                "LLM endpoint rejected chat_template_kwargs; "
                "retrying without it and not sending it again on this client"
            )

    def _parse_completion(self, response: httpx.Response) -> ChatResult:
        """Extract text (``<think>``-stripped), finish reason and usage."""
        try:
            data = response.json()
            choice = data["choices"][0]
            raw_usage = data.get("usage") or {}
            usage = {
                key: int(raw_usage[key])
                for key in ("prompt_tokens", "completion_tokens")
                if key in raw_usage
            }
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            msg = f"malformed response body: {type(exc).__name__}: {exc}"
            raise EndpointError(self.guard.name, response.status_code, msg) from exc
        text = choice.get("message", {}).get("content") or ""
        return ChatResult(
            text=_strip_think_block(text),
            finish_reason=choice.get("finish_reason"),
            usage=usage,
        )


@cache
def get_chat(purpose: Literal["answer", "extract"] = "answer") -> ChatModel:
    """Return the cached chat client for ``purpose``.

    ``'answer'`` uses ``models.llm``; ``'extract'`` uses ``models.extract_llm``
    when configured, falling back to ``models.llm``.

    Args:
        purpose: What the client is used for; selects the endpoint config.

    Returns:
        A process-wide :class:`ChatClient` for that purpose.

    Raises:
        ValueError: If ``purpose`` is not ``'answer'`` or ``'extract'``.
    """
    settings = get_settings()
    if purpose == "answer":
        cfg = settings.models.llm
    elif purpose == "extract":
        cfg = settings.models.extract_llm or settings.models.llm
    else:
        raise ValueError(f"unknown chat purpose: {purpose!r} (expected 'answer' or 'extract')")
    return ChatClient(cfg)
