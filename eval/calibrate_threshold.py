"""Threshold calibration for the ``min_rerank_score`` "not found" gate.

    uv run python -m eval.calibrate_threshold --results eval/results/<run>.json

From a retrieval eval run, each question's best rerank score separates the
answerable questions (should be >= t) from the ``no-answer`` questions (should
be < t). The chosen t maximizes balanced accuracy; ties keep the lowest t
(the task prefers avoiding false "not found" answers). The optimum is reported
overall and for the cross-lingual slice (tag ``cross-lingual``) separately,
because the M0 spike showed cross-lingual rerank scores run lower.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

CROSS_LINGUAL_TAG = "cross-lingual"
SLICE_GAP = 0.05
"""Cross-lingual optimum at least this much lower than overall -> recommend it."""


def best_rerank_score(ranked: Sequence[dict[str, Any]]) -> float:
    """The best rerank score of one question's ranked list.

    Args:
        ranked: Hit records with optional ``rerank_score`` (see run_eval).

    Returns:
        The maximum score, or ``0.0`` when nothing was reranked.
    """
    scores = [hit["rerank_score"] for hit in ranked if hit.get("rerank_score") is not None]
    return max(scores) if scores else 0.0


def balanced_accuracy(pairs: Sequence[tuple[float, bool]], threshold: float) -> float:
    """Balanced accuracy of the gate ``answerable iff score >= threshold``.

    Args:
        pairs: ``(best_rerank_score, is_answerable)`` per question.
        threshold: The candidate threshold.

    Returns:
        ``(TPR + TNR) / 2``; ``0.0`` when one class is missing (the gate is
        not calibratable without both answerable and no-answer questions).
    """
    positives = [score for score, answerable in pairs if answerable]
    negatives = [score for score, answerable in pairs if not answerable]
    if not positives or not negatives:
        return 0.0
    tpr = sum(score >= threshold for score in positives) / len(positives)
    tnr = sum(score < threshold for score in negatives) / len(negatives)
    return (tpr + tnr) / 2


def choose_threshold(pairs: Sequence[tuple[float, bool]]) -> tuple[float, float]:
    """Pick the threshold with the best balanced accuracy (lowest on ties).

    Args:
        pairs: ``(best_rerank_score, is_answerable)`` per question.

    Returns:
        ``(threshold, balanced_accuracy)``; ``(0.0, 0.0)`` when uncalibratable.
    """
    best: tuple[float, float] = (0.0, 0.0)
    for threshold in sorted({score for score, _ in pairs}):
        accuracy = balanced_accuracy(pairs, threshold)
        if accuracy > best[1]:
            best = (threshold, accuracy)
    return best


def collect_pairs(rows: Sequence[dict[str, Any]]) -> list[tuple[float, bool]]:
    """Extract the ``(score, answerable)`` pairs from retrieval run rows.

    Args:
        rows: The ``retrieval.per_question`` rows of a results JSON.

    Returns:
        One pair per question; questions without ``expected_docs`` (the
        ``no-answer`` slice) are labelled ``False``.
    """
    return [
        (best_rerank_score(row.get("ranked") or []), bool(row["question"].get("expected_docs")))
        for row in rows
    ]


def _slice_rows(rows: Sequence[dict[str, Any]], tag: str) -> list[dict[str, Any]]:
    return [row for row in rows if tag in (row["question"].get("tags") or [])]


def calibrate(rows: Sequence[dict[str, Any]], tag: str = CROSS_LINGUAL_TAG) -> dict[str, Any]:
    """Calibrate the gate on a retrieval run, overall and per slice.

    Args:
        rows: The ``retrieval.per_question`` rows of a results JSON.
        tag: The free tag identifying the cross-lingual questions.

    Returns:
        A report dict with the overall and cross-lingual optima, the
        recommendation for ``min_rerank_score`` and its rationale.
    """
    overall_rows = list(rows)
    overall_pairs = collect_pairs(overall_rows)
    overall_threshold, overall_accuracy = choose_threshold(overall_pairs)
    report: dict[str, Any] = {
        "overall": {
            "n": len(overall_rows),
            "threshold": overall_threshold,
            "balanced_accuracy": overall_accuracy,
        },
        "cross_lingual": None,
        "recommendation": overall_threshold,
        "rationale": "",
    }

    cross_rows = _slice_rows(rows, tag)
    if cross_rows:
        # The slice alone has no no-answer questions, so its balanced accuracy
        # is degenerate: pool the slice's answerable questions with the run's
        # full no-answer set (the same negative distribution the gate faces).
        cross_answerable = [row for row in cross_rows if row["question"].get("expected_docs")]
        no_answer_rows = [row for row in rows if not row["question"].get("expected_docs")]
        cross_pairs = collect_pairs(cross_answerable) + collect_pairs(no_answer_rows)
        cross_threshold, cross_accuracy = choose_threshold(cross_pairs)
        below_overall = [
            best_rerank_score(row.get("ranked") or [])
            for row in cross_answerable
            if best_rerank_score(row.get("ranked") or []) < overall_threshold
        ]
        report["cross_lingual"] = {
            "n": len(cross_rows),
            "n_answerable": len(cross_answerable),
            "threshold": cross_threshold,
            "balanced_accuracy": cross_accuracy,
            "below_overall_optimum": len(below_overall),
        }
        if overall_threshold - cross_threshold >= SLICE_GAP:
            report["recommendation"] = cross_threshold
            report["rationale"] = (
                f"The cross-lingual optimum ({cross_threshold:.2f}) runs "
                f"{overall_threshold - cross_threshold:.2f} below the overall optimum "
                f"({overall_threshold:.2f}): cross-lingual rerank scores are lower, so the "
                "global gate would wrongly refuse answerable cross-lingual questions "
                f"({len(below_overall)} of {len(cross_answerable)} fall below the overall "
                'optimum; false "not found"). Recommending the lower value; the trade-off '
                "is a slightly higher risk of weakly grounded answers on truly absent topics."
            )
        elif cross_threshold - overall_threshold >= SLICE_GAP:
            report["rationale"] = (
                f"The cross-lingual optimum ({cross_threshold:.2f}) is above the overall optimum "
                f"({overall_threshold:.2f}), so the overall gate is already more permissive for "
                "cross-lingual questions; no separate cross-lingual gate is needed."
            )
        else:
            report["rationale"] = (
                f"The cross-lingual optimum ({cross_threshold:.2f}) is within {SLICE_GAP:.2f} of "
                f"the overall optimum ({overall_threshold:.2f}); no separate cross-lingual gate "
                "is needed."
            )
    else:
        report["rationale"] = (
            f"No questions carry the {tag!r} tag; only the overall optimum is reported."
        )
    return report


def _print_report(report: dict[str, Any]) -> None:
    """Render the calibration report as Rich tables."""
    table = Table(title="min_rerank_score calibration (best rerank score per question)")
    for column in ("slice", "n", "threshold", "balanced accuracy"):
        table.add_column(column, justify="right" if column != "slice" else "left")
    overall = report["overall"]
    table.add_row(
        "overall",
        str(overall.get("n", "?")),
        f"{overall['threshold']:.3f}",
        f"{overall['balanced_accuracy']:.3f}",
    )
    cross = report["cross_lingual"]
    if cross:
        table.add_row(
            CROSS_LINGUAL_TAG,
            str(cross["n"]),
            f"{cross['threshold']:.3f}",
            f"{cross['balanced_accuracy']:.3f}",
        )
    console.print(table)
    console.print(f"[bold]recommended min_rerank_score:[/] {report['recommendation']:.3f}")
    console.print(report["rationale"])


def main() -> None:
    """Calibrate from a results JSON and print (and save) the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="retrieval results JSON")
    parser.add_argument(
        "--tag", default=CROSS_LINGUAL_TAG, help="tag marking cross-lingual questions"
    )
    args = parser.parse_args()

    data = json.loads(args.results.read_text(encoding="utf-8"))
    rows = (data.get("retrieval") or {}).get("per_question") or []
    if not rows:
        raise SystemExit(f"{args.results} has no retrieval.per_question rows")

    report = calibrate(rows, tag=args.tag)
    report["source"] = str(args.results)
    report["run_tag"] = data.get("tag")
    _print_report(report)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.results.parent / f"{stamp}_threshold.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]Report written to[/] {out_path}")


if __name__ == "__main__":
    main()
