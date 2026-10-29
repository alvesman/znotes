"""Language detection restricted to Portuguese and English (PLAN.md §7.2).

fastText's compressed ``lid.176.ftz`` model, bundled by fast-langdetect (about 1 MB),
decides between the two; lingua did the same job with 290 MB of models in the image.
"""

from __future__ import annotations

import re
from functools import lru_cache

from fast_langdetect import LangDetectConfig, LangDetector

_MIN_LETTERS = 20
"""Below this many letters the text is too short to detect reliably."""
_MIN_CONFIDENCE = 0.6
"""Minimum share of the top language, among Portuguese and English, for a decision."""
_MIN_MASS = 0.3
"""Portuguese and English together must hold this much of the model's probability;
below it the text is in some other language and no decision is made."""
_TOP_K = 176
"""Ask for every label the model has, so both scores are always present."""
_WORD_RE = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)?")
"""Words of letters (with an inner apostrophe): the model sees these only, since
Markdown tables, URLs, numbers and code pull it towards English."""


@lru_cache(maxsize=1)
def _detector() -> LangDetector:
    """Build the detector once: the bundled small model, never a download, whole texts."""
    return LangDetector(LangDetectConfig(model="lite", max_input_length=None))


def detect_language(text: str) -> str | None:
    """Detect whether a text is Portuguese or English.

    Args:
        text: The text to classify (plain text, may be long).

    Returns:
        ``"pt"`` or ``"en"`` when that language holds at least 60% of the
        probability the model gives the two together; None when the text has
        fewer than 20 letters, when the two together hold less than 30% (the
        text is in another language), or when neither reaches 60%.
    """
    letter_count = sum(1 for char in text if char.isalpha())
    if letter_count < _MIN_LETTERS:
        return None
    words = " ".join(_WORD_RE.findall(text))
    scores = {str(row["lang"]): float(row["score"]) for row in _detector().detect(words, k=_TOP_K)}
    portuguese, english = scores.get("pt", 0.0), scores.get("en", 0.0)
    mass = portuguese + english
    if mass < _MIN_MASS:
        return None
    language, share = ("pt", portuguese / mass) if portuguese >= english else ("en", english / mass)
    return language if share >= _MIN_CONFIDENCE else None
