"""Unit tests for the Portuguese/English language detector."""

from __future__ import annotations

from mynotes.text.language import detect_language

_PT_SENTENCE = (
    "Esta frase serve para testar o detetor de idiomas do mynotes e tem "
    "vocabulário suficiente em português para a decisão ser confiável."
)
_EN_SENTENCE = (
    "This sentence exercises the mynotes language detector and carries enough "
    "vocabulary in English for the decision to be reliable."
)


def test_detects_portuguese() -> None:
    assert detect_language(_PT_SENTENCE) == "pt"


def test_detects_english() -> None:
    assert detect_language(_EN_SENTENCE) == "en"


def test_short_text_returns_none() -> None:
    assert detect_language("ok") is None
    assert detect_language("") is None
    assert detect_language("hello world") is None  # 10 letters, below the minimum


def test_other_language_returns_none() -> None:
    french = (
        "Cette phrase est écrite en français pour vérifier que le détecteur ne la "
        "classe ni comme portugais ni comme anglais."
    )
    assert detect_language(french) is None


def test_short_queries_are_detected() -> None:
    assert detect_language("reunião de projeto com a equipa") == "pt"
    assert detect_language("project meeting notes for the team") == "en"
