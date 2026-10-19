"""Compare two eval result files: aggregate + slice deltas, per-question changes.

Usage (from the repo root):
    uv run python -m eval.compare eval/results/<baseline>.json eval/results/<candidate>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()

_SECTIONS: tuple[tuple[str, str], ...] = (
    ("retrieval", "mrr"),
    ("answers", "judge_correctness"),
)


def load(path: Path) -> dict[str, Any]:
    """Load one results JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _delta_table(base: dict[str, Any], cand: dict[str, Any], title: str) -> Table:
    """One Rich table of metric deltas between two aggregate blocks."""
    table = Table(title=title, expand=False)
    table.add_column("metric")
    table.add_column("baseline", justify="right")
    table.add_column("candidate", justify="right")
    table.add_column("delta", justify="right")
    keys = sorted(set(base) | set(cand))
    for key in keys:
        base_value, cand_value = base.get(key), cand.get(key)
        if not isinstance(base_value, (int, float)) or not isinstance(cand_value, (int, float)):
            continue
        delta = cand_value - base_value
        marker = " " if abs(delta) < 1e-9 else ("+" if delta > 0 else "-")
        color = "green" if delta > 0 else ("red" if delta < 0 else "white")
        table.add_row(
            key,
            f"{base_value:.4f}",
            f"{cand_value:.4f}",
            f"[{color}]{marker}{abs(delta):.4f}[/]",
        )
    return table


def _aggregate_delta(base: dict[str, Any], cand: dict[str, Any], title: str) -> None:
    """Print the delta table for one aggregate block."""
    table = _delta_table(base, cand, title)
    if table.row_count:
        console.print(table)


def _slice_deltas(base: dict[str, Any], cand: dict[str, Any]) -> None:
    """Print delta tables for the slices present in both runs."""
    shared = sorted(set(base.get("slices") or {}) & set(cand.get("slices") or {}))
    for name in shared:
        _aggregate_delta(base["slices"][name], cand["slices"][name], f"slice {name}")


def _per_question_changes(base_rows: list, cand_rows: list, metric: str) -> None:
    """Print per-question regressions and improvements on ``metric``."""
    base_by_id = {row["question"]["id"]: row for row in base_rows}
    regressions: list[tuple[str, float, float, str]] = []
    improvements: list[tuple[str, float, float, str]] = []
    for row in cand_rows:
        question_id = row["question"]["id"]
        base_row = base_by_id.get(question_id)
        if not base_row:
            continue
        base_value = float(base_row["metrics"].get(metric, 0) or 0)
        cand_value = float(row["metrics"].get(metric, 0) or 0)
        text = row["question"]["question"][:60]
        if cand_value < base_value:
            regressions.append((question_id, base_value, cand_value, text))
        elif cand_value > base_value:
            improvements.append((question_id, base_value, cand_value, text))
    for label, entries in (("REGRESSIONS", regressions), ("improvements", improvements)):
        if entries:
            console.print(f"\n{label} ({metric}):")
            for question_id, base_value, cand_value, text in entries:
                console.print(f"  {question_id}: {base_value:.3f} -> {cand_value:.3f}  {text}")


def main() -> None:
    """Print the overall + slice deltas of two result files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()

    base, cand = load(args.baseline), load(args.candidate)

    def _label(run: dict[str, Any]) -> str:
        dirty = " (dirty)" if run["git"]["dirty"] else ""
        return f"{run['tag']} @ {run['git']['commit'][:9]}{dirty}"

    console.print(f"baseline:  {_label(base)}")
    console.print(f"candidate: {_label(cand)}")
    if base.get("dataset_sha256") != cand.get("dataset_sha256"):
        console.print(
            "[red]WARNING: the golden set differs between runs; deltas are not comparable![/]"
        )
    base_split, cand_split = base.get("split", "all"), cand.get("split", "all")
    if base_split != cand_split:
        console.print(
            f"[red]WARNING: different golden-set splits ({base_split} vs {cand_split}); "
            "deltas are not comparable![/]"
        )

    for section, key_metric in _SECTIONS:
        if section in base and section in cand:
            console.print(f"\n=== {section} ===")
            _aggregate_delta(
                base[section]["aggregate"]["overall"],
                cand[section]["aggregate"]["overall"],
                f"{section} overall",
            )
            if "aggregate_fused" in base[section] and "aggregate_fused" in cand[section]:
                _aggregate_delta(
                    base[section]["aggregate_fused"]["overall"],
                    cand[section]["aggregate_fused"]["overall"],
                    "retrieval before rerank (fused; pool_hit, window_hit)",
                )
            _slice_deltas(base[section]["aggregate"], cand[section]["aggregate"])
            _per_question_changes(
                base[section]["per_question"], cand[section]["per_question"], key_metric
            )


if __name__ == "__main__":
    main()
