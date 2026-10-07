"""Token counting for chunk budgets (PLAN.md §7.3).

The production counter is the embedding model's own tokenizer (task 12 uses
the model card's ``tokenizer.json``); tiktoken's ``cl100k_base`` and a
whitespace estimate serve as fallbacks so chunking always works, even
offline with a cold cache.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Protocol

import tiktoken
from tokenizers import Tokenizer

from mynotes.config import get_settings

logger = logging.getLogger(__name__)

_TIKTOKEN_ENCODING = "cl100k_base"
_WHITESPACE_INFLATION = 1.3
"""Rough tokens-per-word ratio used by the whitespace fallback."""


class TokenCounter(Protocol):
    """Anything that can count tokens of a text."""

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text``."""
        ...


class WhitespaceCounter:
    """Approximate counter: ``ceil(words * 1.3)``.

    Used in tests and as the last fallback when no real tokenizer is
    available.
    """

    def count(self, text: str) -> int:
        """Return the estimated token count of ``text``.

        Args:
            text: The text to count.

        Returns:
            ``ceil(len(text.split()) * 1.3)``; 0 for empty text.
        """
        return math.ceil(len(text.split()) * _WHITESPACE_INFLATION)


class HFCounter:
    """Counter backed by a Hugging Face ``tokenizer.json``.

    The tokenizer downloads once into the HF cache, then works offline.
    """

    def __init__(self, model_name: str) -> None:
        """Load the tokenizer shipped with ``model_name``.

        Args:
            model_name: Hugging Face model id, e.g. the embedding model.

        Raises:
            Exception: If the tokenizer cannot be downloaded or loaded.
        """
        self._tokenizer = Tokenizer.from_pretrained(model_name)

    def count(self, text: str) -> int:
        """Return the exact token count of ``text``.

        Args:
            text: The text to count.

        Returns:
            Number of tokens produced by the model tokenizer.
        """
        return len(self._tokenizer.encode(text).ids)


class TiktokenCounter:
    """Counter backed by tiktoken's ``cl100k_base``.

    The BPE file downloads on first use, then works offline.
    """

    def __init__(self) -> None:
        """Load the ``cl100k_base`` encoding.

        Raises:
            Exception: If the BPE file cannot be downloaded or loaded.
        """
        self._encoding = tiktoken.get_encoding(_TIKTOKEN_ENCODING)

    def count(self, text: str) -> int:
        """Return the token count of ``text`` under ``cl100k_base``.

        Args:
            text: The text to count.

        Returns:
            Number of tokens produced by the encoding.
        """
        return len(self._encoding.encode(text))


_token_counter: TokenCounter | None = None
_token_counter_lock = threading.Lock()


def _resolve_token_counter() -> TokenCounter:
    """Build the process-wide counter, trying HF, then tiktoken, then whitespace.

    Returns:
        The first counter that loads successfully; every failure is logged
        and swallowed so a missing network or cache never crashes chunking.
    """
    model_name = get_settings().models.embedding.model
    try:
        counter = HFCounter(model_name)
        logger.info("token counter: HF tokenizer for model %s", model_name)
        return counter
    except Exception:
        logger.debug("HF tokenizer unavailable for %s", model_name, exc_info=True)
    try:
        counter = TiktokenCounter()
        logger.info("token counter: tiktoken %s", _TIKTOKEN_ENCODING)
        return counter
    except Exception:
        logger.debug("tiktoken %s unavailable", _TIKTOKEN_ENCODING, exc_info=True)
    logger.info("token counter: whitespace fallback (approximate counts)")
    return WhitespaceCounter()


def get_token_counter() -> TokenCounter:
    """Return the cached process-wide token counter, resolving it on first use.

    Returns:
        The counter installed via :func:`set_token_counter`, or the first of
        HF / tiktoken / whitespace that loads.
    """
    global _token_counter
    if _token_counter is None:
        with _token_counter_lock:
            if _token_counter is None:
                _token_counter = _resolve_token_counter()
    return _token_counter


def set_token_counter(counter: TokenCounter) -> None:
    """Install ``counter`` as the process-wide counter (used by tests).

    Args:
        counter: The counter to install; it replaces any cached one.
    """
    global _token_counter
    _token_counter = counter


def reset_token_counter() -> None:
    """Clear the cached counter so the next :func:`get_token_counter`
    re-resolves it (used by tests)."""
    global _token_counter
    _token_counter = None
