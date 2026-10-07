"""Language detection restricted to Portuguese and English (PLAN.md §7.2)."""

from __future__ import annotations

from functools import lru_cache

from lingua import Language, LanguageDetector, LanguageDetectorBuilder

_MIN_LETTERS = 20
"""Below this many letters the text is too short to detect reliably."""
_MIN_CONFIDENCE = 0.6
"""Minimum confidence of the top language for a detection to be returned."""


@lru_cache(maxsize=1)
def _detector() -> LanguageDetector:
    """Build the two-language detector once and cache it."""
    return LanguageDetectorBuilder.from_languages(Language.PORTUGUESE, Language.ENGLISH).build()


def detect_language(text: str) -> str | None:
    """Detect whether a text is Portuguese or English.

    Args:
        text: The text to classify (plain text, may be long).

    Returns:
        ``"pt"`` or ``"en"`` when the top language reaches the confidence
        threshold; None when the text has fewer than 20 letters or the top
        language's confidence is below 0.6.
    """
    letter_count = sum(1 for char in text if char.isalpha())
    if letter_count < _MIN_LETTERS:
        return None
    values = _detector().compute_language_confidence_values(text)
    if not values or values[0].value < _MIN_CONFIDENCE:
        return None
    return values[0].language.iso_code_639_1.name.lower()
