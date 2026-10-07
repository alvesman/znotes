"""Case- and accent-insensitive normalization shared by every lookup path.

``normalize_key()`` implements PLAN.md §6.1: the same folding used for the
``*_norm`` columns in Postgres (§5) and mirrored by the OpenSearch normalizer
(task 10). ``FOLDING_TEST_STRINGS`` is the shared corpus of (input, expected)
pairs; task 12 reuses it for the parity test against OpenSearch.
"""

from __future__ import annotations

import unicodedata

_NON_DECOMPOSING: dict[str, str] = {
    "ø": "o",
    "æ": "ae",
    "œ": "oe",
    "ł": "l",
    "đ": "d",
    "ð": "d",
    "þ": "th",
    "ı": "i",  # noqa: RUF001
}
_FOLD_TABLE: dict[int, str] = {ord(ch): replacement for ch, replacement in _NON_DECOMPOSING.items()}

FOLDING_TEST_STRINGS: list[tuple[str, str]] = [
    ("Ação", "acao"),
    ("CORAÇÃO", "coracao"),
    ("Frigorífico", "frigorifico"),
    ("FRIGORIFICO", "frigorifico"),
    ("Área-Técnica", "area-tecnica"),
    ("Ærø", "aero"),
    ("straße", "strasse"),
    ("Łódź", "lodz"),
    ("Crème Brûlée", "creme brulee"),
    ("  múltiplos  espaços ", "multiplos espacos"),
    ("naïve café", "naive cafe"),
    ("Øresund", "oresund"),
    ("", ""),
]


def normalize_key(s: str | None) -> str:
    """Fold text for case- and accent-insensitive matching.

    Args:
        s: The input text, or None.

    Returns:
        The normalized key: NFKD decomposition, combining marks dropped,
        casefolded, letters that do not decompose mapped (``ø→o``, ``æ→ae``,
        ``œ→oe``, ``ł→l``, ``đ→d``, ``ð→d``, ``þ→th``, ``ı→i``), and every
        whitespace run (including NBSP and other Unicode spaces) collapsed to
        a single space with the ends stripped. ``None`` becomes ``""``.
    """  # noqa: RUF002
    if s is None:
        return ""
    decomposed = unicodedata.normalize("NFKD", s)
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    mapped = without_marks.casefold().translate(_FOLD_TABLE)
    return " ".join(mapped.split())


def normalize_path_key(rel_path: str) -> str:
    """Normalize each ``/``-separated segment of a relative path.

    Args:
        rel_path: A relative path with ``/`` separators.

    Returns:
        The path with every segment passed through :func:`normalize_key` and
        rejoined with ``/``; whitespace inside segments is preserved as single
        spaces.
    """
    return "/".join(normalize_key(segment) for segment in rel_path.split("/"))
