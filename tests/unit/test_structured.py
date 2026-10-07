"""Unit tests for the structured-output helper (FakeChat, no network)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from fakes import FakeChat
from mynotes.llm.chat import ChatMessage
from mynotes.llm.errors import EndpointError
from mynotes.llm.structured import StructuredOutputError, complete_structured

QUESTION = ChatMessage(role="user", content="Devolve um JSON com nome e idade.")
VALID = '{"name": "Ada", "age": 36}'


class Person(BaseModel):
    """Test schema for structured answers."""

    name: str
    age: int


def expected_error(raw: str) -> str:
    """The full validation error message pydantic produces for ``raw``."""
    try:
        Person.model_validate_json(raw)
    except ValidationError as exc:
        return str(exc)
    raise AssertionError("raw must be invalid")


def test_valid_json_returns_model_instance() -> None:
    """A bare JSON answer validates on the first call with the schema directive."""
    chat = FakeChat([VALID])

    person = complete_structured(chat, [QUESTION], Person)

    assert person == Person(name="Ada", age=36)
    assert len(chat.calls) == 1
    fmt = chat.kwargs[0]["response_format"]
    assert fmt == {
        "type": "json_schema",
        "json_schema": {"name": "Person", "schema": Person.model_json_schema()},
    }
    assert chat.kwargs[0]["priority"] == "background"
    assert chat.kwargs[0]["max_tokens"] == 4096


def test_invalid_json_is_repaired_once() -> None:
    """One repair retry: the bad output is fed back with the error message."""
    bad = "not json at all"
    chat = FakeChat([bad, '{"name": "Grace", "age": 45}'])

    person = complete_structured(chat, [QUESTION], Person)

    assert person == Person(name="Grace", age=45)
    assert len(chat.calls) == 2, "exactly one repair retry"
    assert chat.calls[1] == [
        QUESTION,
        ChatMessage(role="assistant", content=bad),
        ChatMessage(
            role="user",
            content=(
                "Your previous answer was not valid JSON for the schema. "
                f"Error: {expected_error(bad)[:200]}. Reply with only the corrected JSON."
            ),
        ),
    ]
    assert chat.kwargs[1]["response_format"] == chat.kwargs[0]["response_format"]


def test_invalid_twice_raises_structured_output_error() -> None:
    """After the repair retry fails too, StructuredOutputError carries both parts."""
    second = "still not json"
    chat = FakeChat(["garbage {}", second])

    with pytest.raises(StructuredOutputError) as exc_info:
        complete_structured(chat, [QUESTION], Person)

    error = exc_info.value
    assert error.raw_text == second
    assert error.error == expected_error(second)
    assert len(chat.calls) == 2, "no retries beyond the single repair"


def test_json_wrapped_in_prose_is_extracted() -> None:
    """The first {...} block is taken from a prose answer."""
    chat = FakeChat(['Here you go: {"name": "Linus", "age": 30}. Hope that helps!'])

    person = complete_structured(chat, [QUESTION], Person)

    assert person == Person(name="Linus", age=30)
    assert len(chat.calls) == 1


def test_json_in_code_fence_is_extracted() -> None:
    """Fenced JSON (```json ... ```) is also found by the block extraction."""
    chat = FakeChat(["```json\n" + VALID + "\n```"])

    person = complete_structured(chat, [QUESTION], Person)

    assert person == Person(name="Ada", age=36)


def test_repair_prompt_is_passed_with_the_original_conversation() -> None:
    """The repair call keeps the original messages before assistant + user."""
    chat = FakeChat(["nope", VALID])

    complete_structured(chat, [QUESTION], Person)

    assert chat.calls[1][:1] == [QUESTION]
    assert chat.calls[1][1].role == "assistant"
    assert chat.calls[1][2].role == "user"
    assert "Reply with only the corrected JSON." in chat.calls[1][2].content


def test_response_format_rejection_falls_back_and_is_remembered() -> None:
    """A 400 mentioning response_format retries without it and remembers."""

    class RejectingChat(FakeChat):
        """FakeChat that rejects response_format like an old vLLM server."""

        def __init__(self, responses):
            super().__init__(responses)
            self.rejected = 0

        def complete(self, messages, **kwargs):
            if kwargs.get("response_format") is not None:
                self.rejected += 1
                raise EndpointError("llm", 400, "response_format is not supported")
            return super().complete(messages, **kwargs)

    chat = RejectingChat([VALID, VALID])

    person = complete_structured(chat, [QUESTION], Person)

    assert person == Person(name="Ada", age=36)
    assert chat.rejected == 1
    assert len(chat.calls) == 1, "the rejected attempt is not a completed call"
    assert chat.kwargs[0]["response_format"] is None

    complete_structured(chat, [QUESTION], Person)
    assert chat.rejected == 1, "the fallback must be remembered for the client lifetime"
    assert len(chat.calls) == 2
    assert chat.kwargs[1]["response_format"] is None


def test_other_endpoint_errors_propagate() -> None:
    """Errors unrelated to response_format are not swallowed."""
    chat = FakeChat([VALID])

    def raise_500(messages, **kwargs):
        raise EndpointError("llm", 500, "retries exhausted: HTTP 500")

    chat.complete = raise_500

    with pytest.raises(EndpointError):
        complete_structured(chat, [QUESTION], Person)


def test_callable_responses_work_too() -> None:
    """FakeChat also accepts a callable producing answers from the messages."""
    chat = FakeChat(lambda messages: VALID)

    person = complete_structured(chat, [QUESTION], Person)

    assert person == Person(name="Ada", age=36)
    assert chat.calls[0] == [QUESTION]
