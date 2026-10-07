"""Retrieval and answer metrics for the eval harness.

All metric functions take the golden-set question dict and the recorded ranked
list (see ``run_eval.py`` for the record shape) and return plain floats / bools
so aggregation stays trivial. Documents are matched by **rel path** (stable
across re-ingestion); a question without ``expected_docs`` has no measurable
retrieval target and its retrieval metrics are ``None`` so they never poison
the aggregate means (the same for ``page_hit@k`` without ``expected_pages``).
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any

from mynotes.retrieval.prompts import NOT_FOUND
from mynotes.text.normalize import normalize_key

K_VALUES = (1, 3, 5, 10)

_RETRIEVAL_KEYS: tuple[str, ...] = (
    "mrr",
    *(f"doc_hit@{k}" for k in K_VALUES),
    *(f"page_hit@{k}" for k in K_VALUES),
)


def strip_accents(s: str) -> str:
    """Remove combining marks without changing the case.

    Used for the accent-stripped robustness variant; :func:`normalize_key`
    cannot be used because it also lowercases.

    Args:
        s: The input text.

    Returns:
        The text with NFKD combining marks dropped, original case kept.
    """
    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def _expected_docs(question: dict[str, Any]) -> list[str]:
    return list(question.get("expected_docs") or [])


def _doc_matches(question: dict[str, Any], hit: dict[str, Any]) -> bool:
    """A hit matches when its rel path is one of the expected rel paths."""
    return hit.get("path") in _expected_docs(question)


def _page_matches(question: dict[str, Any], hit: dict[str, Any]) -> bool:
    expected_pages = question.get("expected_pages") or []
    if not expected_pages:
        return False
    return _doc_matches(question, hit) and (
        hit.get("page") in expected_pages or hit.get("slide") in expected_pages
    )


def doc_hit_at_k(question: dict[str, Any], ranked: list[dict[str, Any]], k: int) -> bool | None:
    """Whether any of the first ``k`` hits is an expected document.

    Args:
        question: The golden-set question.
        ranked: The ranked hit records.
        k: The cutoff rank.

    Returns:
        True/False, or ``None`` when the question has no ``expected_docs``.
    """
    if not _expected_docs(question):
        return None
    return any(_doc_matches(question, hit) for hit in ranked[:k])


def page_hit_at_k(question: dict[str, Any], ranked: list[dict[str, Any]], k: int) -> bool | None:
    """Whether any of the first ``k`` hits is an expected (doc, page/slide).

    Args:
        question: The golden-set question.
        ranked: The ranked hit records.
        k: The cutoff rank.

    Returns:
        True/False, or ``None`` when the question has no ``expected_pages``.
    """
    if not (question.get("expected_pages") or []):
        return None
    return any(_page_matches(question, hit) for hit in ranked[:k])


def reciprocal_rank(question: dict[str, Any], ranked: list[dict[str, Any]]) -> float | None:
    """RR of the first (doc, page) match; doc-only match when no expected pages.

    Args:
        question: The golden-set question.
        ranked: The ranked hit records.

    Returns:
        ``1/rank`` of the first match, ``0.0`` when nothing matches, or
        ``None`` when the question has no ``expected_docs``.
    """
    if not _expected_docs(question):
        return None
    has_pages = bool(question.get("expected_pages"))
    for position, hit in enumerate(ranked, start=1):
        if has_pages:
            if _page_matches(question, hit):
                return 1.0 / position
        elif _doc_matches(question, hit):
            return 1.0 / position
    return 0.0


def question_retrieval_metrics(
    question: dict[str, Any], ranked: list[dict[str, Any]]
) -> dict[str, Any]:
    """All retrieval metrics for one question at ``K_VALUES``.

    Args:
        question: The golden-set question.
        ranked: The ranked hit records.

    Returns:
        ``mrr``, ``doc_hit@k`` and ``page_hit@k`` for every ``k``; all values
        are ``None`` when the question has no ``expected_docs``.
    """
    if not _expected_docs(question):
        return dict.fromkeys(_RETRIEVAL_KEYS, None)
    metrics: dict[str, Any] = {"mrr": reciprocal_rank(question, ranked)}
    for k in K_VALUES:
        metrics[f"doc_hit@{k}"] = doc_hit_at_k(question, ranked, k)
        metrics[f"page_hit@{k}"] = page_hit_at_k(question, ranked, k)
    return metrics


def source_attribution_correct(question: dict[str, Any], sources: list[dict[str, Any]]) -> bool:
    """Do the answer's emitted sources include an expected doc (and page)?

    Args:
        question: The golden-set question.
        sources: Serialized citation dicts with ``path``/``page``/``slide``.

    Returns:
        True when one cited source names an expected rel path (and page/slide
        when the question expects pages).
    """
    expected_docs = _expected_docs(question)
    expected_pages = question.get("expected_pages") or []
    for source in sources or []:
        if source.get("path") not in expected_docs:
            continue
        if (
            not expected_pages
            or source.get("page") in expected_pages
            or source.get("slide") in expected_pages
        ):
            return True
    return False


def current_version_precision(
    question: dict[str, Any], citations: list[dict[str, Any]]
) -> float | None:
    """Fraction of cited sources that are current and not a superseded path.

    Args:
        question: The golden-set question (``stale_docs`` optional).
        citations: Serialized citation dicts with ``path`` and ``is_current``.

    Returns:
        The share of cited sources whose path is not in ``stale_docs`` and
        whose ``is_current`` is True; ``None`` when nothing is cited.
    """
    if not citations:
        return None
    stale = set(question.get("stale_docs") or [])
    current = [
        citation
        for citation in citations
        if citation.get("is_current") and citation.get("path") not in stale
    ]
    return len(current) / len(citations)


def stale_answer(question: dict[str, Any], answer_text: str | None) -> bool | None:
    """Whether the answer states a superseded value (stale-answer detection).

    Values are compared on :func:`normalize_key`, so case and accent
    differences in the answer still match.

    Args:
        question: The golden-set question (``stale_values`` optional).
        answer_text: The produced answer text.

    Returns:
        True when any stale value appears in the answer, False when none does,
        or ``None`` when the question has no ``stale_values``.
    """
    stale_values = question.get("stale_values") or []
    if not stale_values:
        return None
    answer_key = normalize_key(answer_text or "")
    return any(normalize_key(value) in answer_key for value in stale_values)


# Phrases (matched on normalize_key text, so accents and case don't matter) with which
# the answer model declines on its own: "The provided notes do not contain ...",
# "A informação ... não consta nas notas fornecidas", "... is not mentioned".
_DECLINE = re.compile(
    r"\b("
    r"(do|does|did) not (contain|mention|include|specify|provide|state|say|identify|cover)"
    r"|(is|are|was|were) not (mentioned|specified|provided|stated|included|available|found|given)"
    r"|(isn't|aren't|wasn't) (mentioned|specified|provided|stated|available)"
    r"|no (information|mention|details?|data) (about|on|regarding|of)"
    r"|not (found|available) in the (provided )?(notes|context|documents)"
    r"|nao (consta|constam|contem|ha|existe|existem|menciona|mencionam|refere|referem"
    r"|identifica|identificam|especifica|especificam|indica|indicam"
    r"|esta disponivel|estao disponiveis"
    r"|e mencionad[oa]|sao mencionad[oa]s)"
    r"|nao se encontra|nao foi possivel encontrar|nao encontrei|sem informacao"
    r")\b"
)


def is_decline(answer: str) -> bool:
    """True when an answer says the notes don't contain the requested information.

    Recognizes the fixed not-found texts and the model's own phrasing in Portuguese
    and English. Meant for ``no-answer`` questions only.
    """
    answer_key = normalize_key(answer or "")
    if not answer_key:
        return False
    if answer_key in {normalize_key(text) for text in NOT_FOUND.values()}:
        return True
    return bool(_DECLINE.search(answer_key))


def no_answer_correct(question: dict[str, Any], result: dict[str, Any]) -> bool | None:
    """For a ``no-answer`` question: did the system correctly refuse?

    Args:
        question: The golden-set question.
        result: The answer row with ``answer`` (text) and ``relevant`` (bool).

    Returns:
        True iff the retrieval gate marked the question not relevant, or the answer
        declines (the fixed not-found text or the model's own wording, see
        :func:`is_decline`); ``None`` for every other category.
    """
    if question.get("category") != "no-answer":
        return None
    if result.get("relevant") is False:
        return True
    return is_decline(result.get("answer") or "")


def _mean_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean of every numeric/bool metric over the given rows, plus ``n``."""
    if not rows:
        return {}
    keys: set[str] = set()
    for row in rows:
        keys.update(
            key for key, value in row["metrics"].items() if isinstance(value, (int, float, bool))
        )
    block: dict[str, Any] = {}
    for key in sorted(keys):
        values = [
            float(row["metrics"][key])
            for row in rows
            if isinstance(row["metrics"].get(key), (int, float, bool))
        ]
        if values:
            block[key] = round(sum(values) / len(values), 4)
    block["n"] = len(rows)
    return block


def aggregate(per_question: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-question rows: overall means plus slices.

    Args:
        per_question: Rows with ``question`` and ``metrics`` dicts.

    Returns:
        ``{"overall": ..., "slices": {"language:pt": ..., "category:factual":
        ..., "tag:table": ...}}``; ``None`` metrics are skipped in the means.
    """
    slices: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_question:
        question = row["question"]
        slices[f"language:{question.get('language', '?')}"].append(row)
        if question.get("category"):
            slices[f"category:{question['category']}"].append(row)
        for tag in question.get("tags") or []:
            slices[f"tag:{tag}"].append(row)
    return {
        "overall": _mean_block(per_question),
        "slices": {name: _mean_block(rows) for name, rows in sorted(slices.items())},
    }


PARITY_METRICS: tuple[str, ...] = ("doc_hit@5", "mrr")
"""The metrics the case/accent robustness parity is reported on."""


def parity_block(
    original_rows: list[dict[str, Any]], variant_rows: list[dict[str, Any]]
) -> dict[str, float | None]:
    """Mean per-question delta of the variant vs the original question.

    Args:
        original_rows: Rows of the unmodified questions.
        variant_rows: Rows of one variant (questions matched by ``id``).

    Returns:
        For each :data:`PARITY_METRICS` entry the mean of
        ``variant - original`` over the questions where both are measurable,
        or ``None`` when no pair is measurable.
    """
    by_id = {row["question"]["id"]: row for row in original_rows}
    deltas: dict[str, list[float]] = {metric: [] for metric in PARITY_METRICS}
    for row in variant_rows:
        base = by_id.get(row["question"]["id"])
        if base is None:
            continue
        for metric in PARITY_METRICS:
            base_value = base["metrics"].get(metric)
            variant_value = row["metrics"].get(metric)
            if isinstance(base_value, (int, float)) and isinstance(variant_value, (int, float)):
                deltas[metric].append(float(variant_value) - float(base_value))
    return {
        metric: round(sum(values) / len(values), 4) if values else None
        for metric, values in deltas.items()
    }
