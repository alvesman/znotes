"""RRF parameter sweep over the golden set (task 32, Part B step 2).

    uv run python -m eval.sweep_rrf [--tag m2-sweep] [--split dev] [--limit N]

For ``rank_constant`` in {20, 40, 60, 80} x ``hybrid_weights`` in
{[0.3,0.7], [0.5,0.5], [0.7,0.3]}: build the settings with
``Settings.model_copy(update=...)``, patch every module-level ``get_settings``
reference, re-PUT the OpenSearch RRF pipeline (``ensure_pipeline``) and run a
retrieval-only eval (fused + reranked per question). Configurations are ranked
by **pre-rerank** MRR and doc_hit@5 (the fused candidate list the reranker
sees), then by post-rerank MRR. The sweep runs on the ``dev`` split by
default, so the hold-out questions stay unseen. The original pipeline and settings are always
restored (``try/finally``). A per-config summary table is printed and the full
per-question results are written to ``eval/results/<stamp>_<tag>.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

import mynotes.config
import mynotes.embed.client
import mynotes.index.opensearch
import mynotes.llm.chat
import mynotes.llm.limits
import mynotes.retrieval.analyze
import mynotes.retrieval.assemble
import mynotes.retrieval.hybrid
import mynotes.retrieval.pipeline
import mynotes.retrieval.rerank_client
from eval import metrics as m
from eval.run_eval import (
    DEFAULT_DATASET,
    chunks_doc_count,
    config_snapshot,
    dataset_sha256,
    eval_retrieval,
    git_info,
    load_dataset,
)
from mynotes.config import Settings, get_settings
from mynotes.index.opensearch import ensure_pipeline, get_client

logger = logging.getLogger(__name__)
console = Console()

RANK_CONSTANTS: tuple[int, ...] = (20, 40, 60, 80)
HYBRID_WEIGHTS: tuple[tuple[float, float], ...] = ((0.3, 0.7), (0.5, 0.5), (0.7, 0.3))
RESULTS_DIR = Path(__file__).resolve().parent / "results"

_GET_SETTINGS_MODULES = (
    mynotes.config,
    mynotes.embed.client,
    mynotes.index.opensearch,
    mynotes.llm.chat,
    mynotes.llm.limits,
    mynotes.retrieval.analyze,
    mynotes.retrieval.assemble,
    mynotes.retrieval.hybrid,
    mynotes.retrieval.pipeline,
    mynotes.retrieval.rerank_client,
)
"""Modules holding a ``from mynotes.config import get_settings`` reference."""


def settings_with_fusion(
    base: Settings, rank_constant: int, weights: tuple[float, float]
) -> Settings:
    """A copy of ``base`` with the fusion parameters replaced.

    Args:
        base: The original settings.
        rank_constant: The RRF rank constant.
        weights: The ``[knn, bm25]`` hybrid weights.

    Returns:
        The patched settings (validated values, no other change).
    """
    fusion = base.retrieval.fusion.model_copy(
        update={"rank_constant": rank_constant, "hybrid_weights": list(weights)}
    )
    retrieval = base.retrieval.model_copy(update={"fusion": fusion})
    return base.model_copy(update={"retrieval": retrieval})


@contextmanager
def patched_get_settings(settings: Settings) -> Iterator[None]:
    """Point every module-level ``get_settings`` reference at ``settings``.

       Args:
           settings: The settings the patched accessors return.

    Yields:
           Nothing; the original references are restored on exit.
    """
    originals = [(module, module.get_settings) for module in _GET_SETTINGS_MODULES]

    def _factory(patched: Settings = settings) -> Settings:
        return patched

    try:
        for module, _ in originals:
            module.get_settings = _factory
        yield
    finally:
        for module, original in originals:
            module.get_settings = original


def sweep_config(
    client: Any,
    questions: list[dict[str, Any]],
    base: Settings,
    rank_constant: int,
    weights: tuple[float, float],
) -> dict[str, Any]:
    """Run the retrieval-only eval for one (rank_constant, weights) pair.

    Args:
        client: The OpenSearch client (for the pipeline upsert).
        questions: The golden-set questions.
        base: The original settings (source of everything but the fusion).
        rank_constant: The RRF rank constant to apply.
        weights: The hybrid weights to apply.

    Returns:
        The config record with pre-rerank and post-rerank aggregates.
    """
    settings = settings_with_fusion(base, rank_constant, weights)
    with patched_get_settings(settings):
        ensure_pipeline(client, settings)
        rows = eval_retrieval(questions)
    fused_view = [{"question": row["question"], "metrics": row["fused_metrics"]} for row in rows]
    pre = m.aggregate(fused_view)["overall"]
    post = m.aggregate(rows)["overall"]
    return {
        "rank_constant": rank_constant,
        "hybrid_weights": list(weights),
        "pre_rerank": {key: pre.get(key) for key in ("mrr", "doc_hit@5", "doc_hit@10", "n")},
        "post_rerank": {key: post.get(key) for key in ("mrr", "doc_hit@5", "n")},
        "fused_latency_ms_mean": pre.get("fused_latency_ms"),
        "per_question": [
            {
                "id": row["question"]["id"],
                "fused_metrics": row["fused_metrics"],
                "metrics": row["metrics"],
            }
            for row in rows
        ],
    }


def rank_key(record: dict[str, Any]) -> tuple[float, float, float]:
    """Sort key: pre-rerank MRR, then pre-rerank doc_hit@5, then post-rerank MRR.

    Args:
        record: One sweep config record.

    Returns:
        The descending-sort key (missing metrics count as 0).
    """
    pre, post = record["pre_rerank"], record["post_rerank"]
    return (
        float(pre.get("mrr") or 0.0),
        float(pre.get("doc_hit@5") or 0.0),
        float(post.get("mrr") or 0.0),
    )


def print_table(records: list[dict[str, Any]]) -> None:
    """Print the ranked summary table.

    Args:
        records: Sweep records, already sorted best-first.
    """
    table = Table(title="RRF sweep (ranked by pre-rerank MRR, doc_hit@5, post MRR)")
    for column in (
        "rank",
        "rank_constant",
        "weights",
        "pre MRR",
        "pre dh@5",
        "post MRR",
        "post dh@5",
        "fused ms",
    ):
        table.add_column(column, justify="right" if column != "weights" else "left")
    for position, record in enumerate(records, start=1):
        pre, post = record["pre_rerank"], record["post_rerank"]
        table.add_row(
            str(position),
            str(record["rank_constant"]),
            str(record["hybrid_weights"]),
            f"{pre.get('mrr'):.4f}" if pre.get("mrr") is not None else "-",
            f"{pre.get('doc_hit@5'):.4f}" if pre.get("doc_hit@5") is not None else "-",
            f"{post.get('mrr'):.4f}" if post.get("mrr") is not None else "-",
            f"{post.get('doc_hit@5'):.4f}" if post.get("doc_hit@5") is not None else "-",
            str(record.get("fused_latency_ms_mean") or "-"),
        )
    console.print(table)


def main() -> None:
    """Run the sweep and write the results JSON."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("eval").setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="m2-sweep", help="label for the results file")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--split", choices=m.SPLITS, default="dev", help="golden-set part to tune on (dev)"
    )
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    questions = m.select_split(list(dataset["questions"]), args.split)
    if args.limit:
        questions = questions[: args.limit]

    base = get_settings()
    client = get_client()
    records: list[dict[str, Any]] = []
    try:
        for rank_constant in RANK_CONSTANTS:
            for weights in HYBRID_WEIGHTS:
                label = f"rank_constant={rank_constant} weights={list(weights)}"
                logger.info("sweep %s", label)
                record = sweep_config(client, questions, base, rank_constant, weights)
                records.append(record)
                pre = record["pre_rerank"]
                console.print(
                    f"[green]{label}[/] pre MRR={pre.get('mrr')} dh@5={pre.get('doc_hit@5')}"
                )
    finally:
        ensure_pipeline(client, base)
        logger.info("original RRF pipeline restored")

    records.sort(key=rank_key, reverse=True)
    print_table(records)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{stamp}_{args.tag}.json"
    out_path.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "timestamp": datetime.now(UTC).isoformat(),
                "dataset": str(args.dataset),
                "dataset_sha256": dataset_sha256(args.dataset),
                "n_questions": len(questions),
                "git": git_info(),
                "config": config_snapshot(),
                "chunks_doc_count": chunks_doc_count(),
                "ranked_configs": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    console.print(f"[green]Sweep results written to[/] {out_path}")
    best = records[0]
    console.print(
        f"[bold]best:[/] rank_constant={best['rank_constant']} "
        f"weights={best['hybrid_weights']} (pre MRR={best['pre_rerank'].get('mrr')})"
    )


if __name__ == "__main__":
    main()
