"""Follow-up questions in a chat: rewrite them for retrieval, give the answer the earlier turns.

A follow-up ("e a porta?", "and who maintains it?") cannot be searched as typed:
the words that matter are in the earlier turns. Two steps, both only when the
chat already has turns:

1. :func:`condense_question` asks the model for a standalone version of the
   question (one short call, thinking off), used for retrieval only. A question
   that already stands alone comes back unchanged.
2. :func:`history_messages` turns the last turns into chat messages placed before
   the grounded prompt, so the answer can refer to what was said. Earlier ``[n]``
   markers are removed: they pointed at that turn's sources, not this one's.

Both use at most ``retrieval.followup_turns`` turns (0 turns the feature off).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from mynotes.llm.chat import ChatMessage, ChatModel

logger = logging.getLogger(__name__)

_ANSWER_CHARS = 1200
"""Characters of each earlier answer kept (rewrite prompt and answer history)."""

_REWRITE_MAX_TOKENS = 200
_MARKER_RE = re.compile(r"\s*\[\d+\]")

REWRITE_PROMPT = """You turn the last question of a conversation about the user's notes into a \
standalone search question.

Rules:
1. Resolve pronouns and references ("it", "that project", "e a porta?", "e ele?") using the \
earlier turns, so the question can be understood without them.
2. If the question already stands alone, return it unchanged.
3. Keep the question's language and wording; do not answer it, do not add facts.
4. Keep filter tokens exactly as written (as-of:, since:, until:, in:, #tag, history:).
5. Reply with the question only, on one line."""


@dataclass(frozen=True)
class Turn:
    """One earlier question and its answer."""

    question: str
    answer: str


def _short(answer: str) -> str:
    text = _MARKER_RE.sub("", answer).strip()
    return text if len(text) <= _ANSWER_CHARS else text[:_ANSWER_CHARS].rstrip() + " …"


def condense_question(question: str, turns: Sequence[Turn], chat: ChatModel) -> str:
    """A standalone version of ``question`` given the earlier ``turns``.

    Returns ``question`` unchanged when there are no turns, when the model call
    fails, or when the reply is empty or implausibly long.
    """
    if not turns:
        return question
    conversation = "\n\n".join(
        f"User: {turn.question}\nAssistant: {_short(turn.answer)}" for turn in turns
    )
    messages = [
        ChatMessage(role="system", content=REWRITE_PROMPT),
        ChatMessage(
            role="user",
            content=f"Earlier turns:\n{conversation}\n\nLast question: {question}\n\n"
            "Standalone question:",
        ),
    ]
    try:
        result = chat.complete(
            messages, priority="interactive", max_tokens=_REWRITE_MAX_TOKENS, temperature=0.0
        )
    except Exception:
        logger.warning("follow-up rewrite failed; searching the question as typed", exc_info=True)
        return question
    lines = [line.strip() for line in (result.text or "").strip().splitlines() if line.strip()]
    rewritten = lines[0].strip().strip('"“”«»').strip() if lines else ""
    if rewritten.lower().startswith(("standalone question:", "pergunta:")):
        rewritten = rewritten.split(":", 1)[1].strip()
    if not rewritten or len(rewritten) > 3 * len(question) + 300:
        return question
    return rewritten


def history_messages(turns: Sequence[Turn]) -> list[ChatMessage]:
    """The earlier turns as alternating user/assistant messages, oldest first."""
    messages: list[ChatMessage] = []
    for turn in turns:
        messages.append(ChatMessage(role="user", content=turn.question))
        messages.append(ChatMessage(role="assistant", content=_short(turn.answer)))
    return messages
