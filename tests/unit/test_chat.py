"""Unit tests for the chat client (MockTransport, no network)."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import httpx
import pytest

from fakes import mock_transport
from mynotes.config import LLMConfig, LLMLimits, RetryConfig, Settings
from mynotes.llm.chat import ChatClient, ChatMessage, ChatResult, get_chat
from mynotes.llm.errors import EndpointError
from mynotes.llm.limits import LLMGuard

type Call = dict[str, Any]

USER = ChatMessage(role="user", content="oi")


def chat_response(
    text: str = "olá", finish_reason: str | None = "stop", usage: dict[str, int] | None = None
) -> httpx.Response:
    """A 200 chat.completion response with one choice."""
    body: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


def make_chat(
    settings: Settings,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    retry: RetryConfig | None = None,
    **cfg_updates: object,
) -> tuple[ChatClient, list[Call]]:
    """Build a client on a MockTransport; ``calls`` records every request."""
    cfg: LLMConfig = (
        settings.models.llm.model_copy(update=cfg_updates) if cfg_updates else settings.models.llm
    )
    guard = LLMGuard(LLMLimits(), retry if retry is not None else RetryConfig(max_attempts=1))
    calls: list[Call] = []

    def record(request: httpx.Request) -> httpx.Response:
        calls.append({"request": request, "body": json.loads(request.content)})
        return handler(request)

    http = httpx.Client(transport=mock_transport({("POST", "/chat/completions"): record}))
    return ChatClient(cfg, guard=guard, http=http), calls


def sse_response(*chunks: str) -> httpx.Response:
    """A 200 SSE body with one data chunk per piece, ending with [DONE]."""
    lines = [
        f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]})}" for piece in chunks
    ]
    lines.append("data: [DONE]")
    return httpx.Response(200, text="\n\n".join(lines) + "\n\n")


def test_payload_includes_chat_template_kwargs_only_for_thinking_models(
    test_settings: Settings,
) -> None:
    """thinking_model=True sends enable_thinking=false; False omits the field."""

    def handler(request: httpx.Request) -> httpx.Response:
        return chat_response()

    thinking_client, thinking_calls = make_chat(test_settings, handler, thinking_model=True)
    thinking_client.complete([USER])
    body = thinking_calls[0]["body"]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["model"] == test_settings.models.llm.model
    assert body["messages"] == [{"role": "user", "content": "oi"}]
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == min(test_settings.models.llm.max_tokens, 4096)

    plain_client, plain_calls = make_chat(test_settings, handler, thinking_model=False)
    plain_client.complete([USER])
    assert "chat_template_kwargs" not in plain_calls[0]["body"]


def test_400_mentioning_chat_template_kwargs_retries_once_without_it(
    test_settings: Settings,
) -> None:
    """The first 400 strips the kwarg and retries; later calls never send it."""
    state = {"thinking_rejected": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if not state["thinking_rejected"]:
            state["thinking_rejected"] = True
            return httpx.Response(
                400,
                text='{"object": "error", "message": "Extra inputs are not permitted: '
                'chat_template_kwargs", "type": "invalid_request_error"}',
            )
        return chat_response("olá")

    client, calls = make_chat(test_settings, handler, thinking_model=True)
    result = client.complete([USER])

    assert result.text == "olá"
    assert len(calls) == 2
    assert "chat_template_kwargs" in calls[0]["body"]
    assert "chat_template_kwargs" not in calls[1]["body"]

    client.complete([USER])
    assert len(calls) == 3, "the fallback must be remembered for the client lifetime"
    assert "chat_template_kwargs" not in calls[2]["body"]


def test_leading_think_block_is_stripped(test_settings: Settings) -> None:
    """A leading <think>...</think> block is removed from the returned text."""
    client, _ = make_chat(test_settings, lambda request: chat_response("<think>abc</think>Olá"))

    result = client.complete([USER])

    assert result.text == "Olá"
    assert result.finish_reason == "stop"
    assert result.usage == {}


def test_usage_and_temperature_passthrough(test_settings: Settings) -> None:
    """usage tokens are surfaced and an explicit temperature overrides the config."""
    usage = {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    client, calls = make_chat(test_settings, lambda request: chat_response("olá", usage=usage))

    result = client.complete([USER], temperature=0.3, max_tokens=200)

    assert result.usage == {"prompt_tokens": 5, "completion_tokens": 2}
    assert calls[0]["body"]["temperature"] == 0.3
    assert calls[0]["body"]["max_tokens"] == 200


def test_response_format_is_passed_through(test_settings: Settings) -> None:
    """A response_format directive is forwarded verbatim in the payload."""
    fmt = {"type": "json_schema", "json_schema": {"name": "Out", "schema": {"type": "object"}}}
    client, calls = make_chat(test_settings, lambda request: chat_response())

    client.complete([USER], response_format=fmt)

    assert calls[0]["body"]["response_format"] == fmt


def test_malformed_completion_body_raises_endpoint_error(test_settings: Settings) -> None:
    """A 200 without choices fails with a clear EndpointError."""
    client, _ = make_chat(test_settings, lambda request: httpx.Response(200, json={"oops": True}))

    with pytest.raises(EndpointError, match="malformed response body") as exc_info:
        client.complete([USER])

    assert exc_info.value.endpoint == "llm"


def test_streaming_yields_delta_pieces_in_order(test_settings: Settings) -> None:
    """Three data chunks plus [DONE] yield three pieces in server order."""
    client, calls = make_chat(test_settings, lambda request: sse_response("Uma", "duas", "três"))

    pieces = list(client.stream([USER]))

    assert pieces == ["Uma", "duas", "três"]
    assert calls[0]["body"]["stream"] is True
    assert "response_format" not in calls[0]["body"]


def test_streaming_skips_empty_deltas_and_done(test_settings: Settings) -> None:
    """Chunks without content (role-only, empty lines) yield nothing."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n'
            "\n\n"
            'data: {"choices": [{"delta": {"content": "olá"}}]}\n\n'
            "data: not-json\n\n"
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=body)

    client, _ = make_chat(test_settings, handler)

    assert list(client.stream([USER])) == ["olá"]


def test_stream_retries_transient_failure_before_first_chunk(test_settings: Settings) -> None:
    """A 500 on the first attempt is retried; the stream then yields normally."""
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(500, text="boom")
        return sse_response("a", "b", "c")

    client, calls = make_chat(
        test_settings, handler, retry=RetryConfig(max_attempts=2, backoff_base_s=0.0)
    )

    pieces = list(client.stream([USER]))

    assert pieces == ["a", "b", "c"]
    assert len(calls) == 2


def test_stream_non_transient_error_raises_without_retry(test_settings: Settings) -> None:
    """A 400 that does not mention chat_template_kwargs fails immediately."""
    client, calls = make_chat(
        test_settings, lambda request: httpx.Response(400, text="bad messages")
    )

    with pytest.raises(EndpointError, match="bad messages"):
        list(client.stream([USER]))

    assert len(calls) == 1


def test_priority_routes_to_the_background_pool(test_settings: Settings) -> None:
    """complete() and stream() both acquire the guard slot with their priority."""
    client, _ = make_chat(
        test_settings,
        lambda request: chat_response(),
    )
    priorities: list[str] = []
    original_slot = client.guard.slot

    def spy(*args: object, **kwargs: object) -> Any:
        priorities.append(args[0] if args else kwargs.get("priority", "interactive"))
        return original_slot(*args, **kwargs)

    client.guard.slot = spy

    client.complete([USER])
    client.complete([USER], priority="background")
    list(client.stream([USER], priority="background"))

    assert priorities == ["interactive", "background", "background"]


def test_get_chat_caches_per_purpose_and_falls_back(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """'extract' uses extract_llm when set, else models.llm; clients are cached."""
    get_chat.cache_clear()
    try:
        answer = get_chat("answer")
        extract = get_chat("extract")
        assert answer is not extract
        assert answer.cfg is extract.cfg is test_settings.models.llm
        assert get_chat("answer") is answer
    finally:
        get_chat.cache_clear()

    settings = test_settings.model_copy(deep=True)
    settings.models.extract_llm = LLMConfig(
        base_url="http://extract.test/v1", model="extract-model"
    )
    monkeypatch.setattr("mynotes.llm.chat.get_settings", lambda: settings)
    get_chat.cache_clear()
    try:
        dedicated = get_chat("extract")
        assert dedicated.cfg is settings.models.extract_llm
    finally:
        get_chat.cache_clear()


def test_get_chat_rejects_unknown_purpose(test_settings: Settings) -> None:
    """An unknown purpose raises ValueError before any client is built."""
    get_chat.cache_clear()
    with pytest.raises(ValueError, match="unknown chat purpose"):
        get_chat("translate")  # type: ignore[arg-type]


def test_chat_result_is_immutable() -> None:
    """ChatMessage and ChatResult are frozen dataclasses."""
    message = ChatMessage(role="user", content="oi")
    result = ChatResult(text="olá", finish_reason="stop", usage={})

    with pytest.raises(AttributeError):
        message.content = "x"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        result.text = "x"  # type: ignore[misc]


def test_stream_slot_held_while_consuming(test_settings: Settings) -> None:
    """The guard slot is still held while the consumer iterates the stream."""
    client, _ = make_chat(test_settings, lambda request: sse_response("a", "b", "c"))
    holder: list[bool] = []
    original_slot = client.guard.slot

    @contextmanager
    def spy(*args: object, **kwargs: object) -> Iterator[None]:
        with original_slot(*args, **kwargs):
            holder.append(True)
            try:
                yield
            finally:
                holder.pop()

    client.guard.slot = spy

    stream = client.stream([USER])
    next(iter(stream))
    assert holder == [True], "slot must be held between pieces"
    for _ in stream:
        pass
    assert holder == [], "slot must be released after the stream ends"
