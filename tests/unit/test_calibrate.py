"""Unit tests for the ``min_rerank_score`` calibration (task 32).

The synthetic scores mirror the task spec: answerable ``[0.9, 0.8, 0.4]``,
no-answer ``[0.1, 0.3]``; the optimal gate separates them perfectly.
"""

from __future__ import annotations

from eval.calibrate_threshold import (
    balanced_accuracy,
    best_rerank_score,
    calibrate,
    choose_threshold,
    collect_pairs,
)


def test_best_rerank_score_ignores_unreranked_hits() -> None:
    ranked = [{"rerank_score": 0.3}, {"rerank_score": 0.7}, {"rerank_score": None}]
    assert best_rerank_score(ranked) == 0.7
    assert best_rerank_score([{"rerank_score": None}]) == 0.0
    assert best_rerank_score([]) == 0.0


def test_threshold_separates_synthetic_scores_perfectly() -> None:
    pairs = [(0.9, True), (0.8, True), (0.4, True), (0.1, False), (0.3, False)]
    threshold, accuracy = choose_threshold(pairs)
    assert accuracy == 1.0
    assert 0.3 < threshold <= 0.4


def test_balanced_accuracy_at_extremes() -> None:
    pairs = [(0.9, True), (0.8, True), (0.4, True), (0.1, False), (0.3, False)]
    # Everything predicted answerable: perfect recall, zero specificity.
    assert balanced_accuracy(pairs, 0.05) == 0.5
    # 0.3 is a no-answer score: it must stay below the gate.
    assert balanced_accuracy(pairs, 0.3) == 0.75


def test_choose_threshold_keeps_lowest_tie() -> None:
    pairs = [(0.9, True), (0.5, True), (0.8, False), (0.2, False)]
    # t=0.5 and t=0.9 both score balanced accuracy 0.75 (tie); the lower one
    # is kept because it maximizes TPR (fewest false "not found" answers).
    threshold, accuracy = choose_threshold(pairs)
    assert threshold == 0.5
    assert accuracy == 0.75


def test_collect_pairs_labels_no_answer_false() -> None:
    rows = [
        {"question": {"id": "d001", "expected_docs": ["a.md"]}, "ranked": [{"rerank_score": 0.8}]},
        {"question": {"id": "d002", "category": "no-answer"}, "ranked": [{"rerank_score": 0.1}]},
    ]
    assert collect_pairs(rows) == [(0.8, True), (0.1, False)]


def _row(qid: str, score: float, *, answerable: bool = True, cross: bool = False) -> dict:
    question: dict = {"id": qid, "tags": ["cross-lingual"] if cross else []}
    if answerable:
        question["expected_docs"] = [f"{qid}.md"]
    else:
        question["category"] = "no-answer"
    return {"question": question, "ranked": [{"rerank_score": score}]}


def test_calibrate_cross_lingual_above_overall_needs_no_separate_gate() -> None:
    rows = [
        _row("a1", 0.9, cross=True),
        _row("a2", 0.8, cross=True),
        _row("a3", 0.3),
        _row("n1", 0.1, answerable=False),
        _row("n2", 0.25, answerable=False),
    ]
    report = calibrate(rows)
    assert report["cross_lingual"]["threshold"] > report["recommendation"]
    assert "above the overall optimum" in report["rationale"]
    assert "close to" not in report["rationale"]
