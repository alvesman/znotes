"""Eval harness runner.

Usage (from the repo root, against the real local stack):
    uv run python -m eval.run_eval --mode retrieval --tag baseline
    uv run python -m eval.run_eval --mode both --tag c1-context-budget
    uv run python -m eval.run_eval --mode answers --ids q001,q002 --tag debug

Modes:
    retrieval — ``retrieve()`` only (no answer LLM); primary metrics. Records
        both the reranked list and the pre-rerank fused list (RRF tuning).
    answers   — full ``answer()`` with citations and the LLM judge.
    both      — both of the above.

``--split dev`` (or ``holdout``) runs one part of the golden set: tune on
``dev``, confirm a finished change on ``holdout`` (``metrics.question_split``).

``--variants`` re-runs retrieval for every question with accents stripped
(case kept), all upper and all lower case, and records the parity of
doc_hit@5 / MRR against the original (target: no drop larger than 0.02).

Each run writes ``eval/results/<UTC timestamp>_<tag>.json`` with the git
commit + dirty flag, the golden-set SHA-256, a config snapshot (retrieval,
models without keys, versions), the chunk count, per-question rows and
aggregates. Compare two runs with ``eval/compare.py``. No auth bypass is
needed: local mode has no ACL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

from eval import judge as judge_mod
from eval import metrics as m
from mynotes.retrieval.types import SearchFilters

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("eval")
logger.setLevel(logging.INFO)

console = Console()

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_DATASET = Path(__file__).resolve().parent / "golden_set.yaml"

VARIANTS: dict[str, Any] = {
    "accents-stripped": m.strip_accents,
    "upper": str.upper,
    "lower": str.lower,
}
"""The case/accent robustness variants (PLAN.md §14.2)."""

_SECRET_KEYS = frozenset({"api_key", "password", "dsn"})
"""Keys dropped from the config snapshot (redacted_dict masks them anyway)."""


def load_dataset(path: Path) -> dict[str, Any]:
    """Load and validate the golden-set YAML.

    Args:
        path: Path to the golden-set file.

    Returns:
        The parsed dataset with at least a ``questions`` list.
    """
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not data or "questions" not in data:
        raise SystemExit(f"Dataset {path} has no questions")
    return data


def dataset_sha256(path: Path) -> str:
    """The golden-set file hash, recorded so compared runs are auditable."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_info() -> dict[str, Any]:
    """The HEAD commit and dirty flag of the repo (eval comparability)."""

    def _run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {"commit": _run("rev-parse", "HEAD"), "dirty": bool(_run("status", "--porcelain"))}


def _drop_secrets(node: Any) -> Any:
    """Recursively drop secret-named keys from a settings dump."""
    if isinstance(node, dict):
        return {key: _drop_secrets(value) for key, value in node.items() if key not in _SECRET_KEYS}
    if isinstance(node, list):
        return [_drop_secrets(item) for item in node]
    return node


def config_snapshot() -> dict[str, Any]:
    """Everything that can change a result, so runs are comparable."""
    from mynotes.config import get_settings

    redacted = get_settings().redacted_dict()
    return {
        "retrieval": redacted.get("retrieval"),
        "models": _drop_secrets(redacted.get("models")),
        "versions": redacted.get("versions"),
    }


def chunks_doc_count() -> int | str:
    """Document count of the chunks alias, or a reason when unavailable."""
    try:
        from mynotes.config import get_settings
        from mynotes.index.opensearch import get_client

        client = get_client()
        try:
            return int(client.count(index=get_settings().opensearch.chunks_alias)["count"])
        finally:
            client.close()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def filters_from_question(question: dict[str, Any]) -> SearchFilters:
    """Build the base :class:`SearchFilters` from a question's ``filters``."""
    explicit = question.get("filters") or {}
    return SearchFilters(
        tags=list(explicit.get("tags") or []),
        folders=list(explicit.get("folders") or []),
    )


def _hit_record(hit: Any) -> dict[str, Any]:
    """The ranked-record shape the metrics operate on."""
    return {
        "path": hit.path,
        "page": hit.page,
        "slide": hit.slide,
        "rank": hit.rank,
        "score": hit.score,
        "rerank_score": hit.rerank_score,
        "sources": dict(hit.sources),
        "hinted": bool(hit.query_hint),
    }


def _retrieve_question(
    question_text: str, question: dict[str, Any], *, rerank: bool = True
) -> tuple[Any, int]:
    """Run the pipeline once, returning the result and the latency in ms."""
    from mynotes.retrieval.pipeline import retrieve

    started = time.monotonic()
    result = retrieve(question_text, filters_from_question(question), rerank=rerank)
    return result, round((time.monotonic() - started) * 1000)


def eval_retrieval(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retrieval mode: metrics on the reranked list and the fused list.

    The fused metrics are taken on the **whole** fused candidate list, plus
    ``pool_hit``/``window_hit``: whether the expected document is among the
    candidates at all, and among the ones sent to the reranker.

    Args:
        questions: The selected golden-set questions.

    Returns:
        One row per question with the reranked records, the pre-rerank fused
        records and both metric dicts.
    """
    from mynotes.config import get_settings
    from mynotes.retrieval.rerank import rerank_window

    rows: list[dict[str, Any]] = []
    for question in questions:
        fused, fused_ms = _retrieve_question(question["question"], question, rerank=False)
        fused_ranked = [_hit_record(hit) for hit in fused.fused]
        for position, record in enumerate(fused_ranked, start=1):
            record["rank"] = position
        window = rerank_window(fused.fused, get_settings().retrieval.rerank_top)
        fused_metrics = m.question_retrieval_metrics(question, fused_ranked)
        fused_metrics.update(
            m.candidate_metrics(question, fused_ranked, [_hit_record(hit) for hit in window])
        )
        fused_metrics["fused_latency_ms"] = fused_ms

        final, latency_ms = _retrieve_question(question["question"], question)
        ranked = [_hit_record(hit) for hit in final.hits]
        row_metrics = m.question_retrieval_metrics(question, ranked)
        row_metrics["retrieval_latency_ms"] = latency_ms
        row_metrics["relevant"] = final.relevant

        rows.append(
            {
                "question": question,
                "ranked": ranked,
                "fused_ranked": fused_ranked,
                "metrics": row_metrics,
                "fused_metrics": fused_metrics,
            }
        )
        logger.info(
            "[retrieval] %s: mrr=%s doc_hit@5=%s (%dms)",
            question["id"],
            row_metrics["mrr"],
            row_metrics["doc_hit@5"],
            latency_ms,
        )
    return rows


def eval_variants(
    questions: list[dict[str, Any]], original_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Case/accent robustness variants (retrieval only) with parity blocks.

    Args:
        questions: The selected golden-set questions.
        original_rows: The original retrieval rows for the parity deltas.

    Returns:
        One block per variant with per-question rows, aggregate and parity.
    """

    blocks: dict[str, Any] = {}
    for name, transform in VARIANTS.items():
        rows = []
        for question in questions:
            variant = {**question, "question": transform(question["question"])}
            result, latency_ms = _retrieve_question(variant["question"], question)
            ranked = [_hit_record(hit) for hit in result.hits]
            row_metrics = m.question_retrieval_metrics(variant, ranked)
            row_metrics["retrieval_latency_ms"] = latency_ms
            rows.append({"question": variant, "ranked": ranked, "metrics": row_metrics})
            logger.info(
                "[variant %s] %s: doc_hit@5=%s", name, question["id"], row_metrics["doc_hit@5"]
            )
        blocks[name] = {
            "per_question": rows,
            "aggregate": m.aggregate(rows),
            "parity": m.parity_block(original_rows, rows),
        }
        logger.info("[variant %s] parity=%s", name, blocks[name]["parity"])
    return blocks


def eval_answers(questions: list[dict[str, Any]], with_judge: bool = True) -> list[dict[str, Any]]:
    """Answers mode: the full answer pipeline with version/stale metrics.

    Args:
        questions: The selected golden-set questions.
        with_judge: Whether to run the LLM judge (secondary evidence).

    Returns:
        One row per question with the answer, citations, judge verdict and
        metrics.
    """
    from mynotes.llm.chat import get_chat
    from mynotes.retrieval.answer import answer, citation_to_dict

    chat = get_chat("answer")
    rows: list[dict[str, Any]] = []
    for question in questions:
        started = time.monotonic()
        error: str | None = None
        try:
            result = answer(question["question"], filters_from_question(question))
            answer_text = result.answer
            citations = [citation_to_dict(citation) for citation in result.citations]
            relevant: bool | None = result.relevant
        except Exception as exc:
            logger.exception("answer failed for %s", question["id"])
            answer_text, citations, relevant, error = "", [], None, str(exc)
        latency_ms = round((time.monotonic() - started) * 1000)

        row_metrics: dict[str, Any] = {
            "answer_latency_ms": latency_ms,
            "source_attribution_correct": None,
            "current_version_precision": None,
            "stale_answer": None,
            "no_answer_correct": None,
        }
        if not error:
            row_metrics.update(
                {
                    "source_attribution_correct": m.source_attribution_correct(question, citations),
                    "current_version_precision": m.current_version_precision(question, citations),
                    "stale_answer": m.stale_answer(question, answer_text),
                    "no_answer_correct": m.no_answer_correct(
                        question, {"answer": answer_text, "relevant": relevant}
                    ),
                }
            )
        judge_result: dict[str, Any] = {}
        if with_judge and not error:
            reference = judge_mod.build_reference(question)
            judge_result = judge_mod.judge_answer(chat, question, answer_text, reference)
            if "judge_error" not in judge_result:
                for key in ("correctness", "groundedness", "completeness"):
                    row_metrics[f"judge_{key}"] = judge_result.get(key)
                row_metrics["judge_pass"] = judge_result.get("verdict") == "pass"

        rows.append(
            {
                "question": question,
                "answer": answer_text,
                "citations": citations,
                "relevant": relevant,
                "error": error,
                "judge": judge_result,
                "metrics": row_metrics,
            }
        )
        logger.info(
            "[answers] %s: verdict=%s (%dms)",
            question["id"],
            judge_result.get("verdict", "n/a"),
            latency_ms,
        )
    return rows


def apply_overrides(specs: list[str]) -> dict[str, Any]:
    """Apply ``dotted.key=value`` settings overrides in-process.

    Values are coerced to the current field's type (bool/int/float/str; YAML
    for an optional field that is unset, so ``--set retrieval.knn_k=200`` gives
    an int), so
    ``--set retrieval.graph_expansion=true`` and ``--set retrieval.graph_hops=2``
    work without YAML. The mutated :func:`~mynotes.config.get_settings` object
    is what the whole process then sees; a final
    ``Settings.model_validate`` rejects unknown or invalid values (``Literal``
    fields such as ``retrieval.graph_method``) before any question runs.

    Args:
        specs: The ``--set`` arguments, each ``dotted.key=value``.

    Returns:
        The applied overrides as ``key -> python value`` (recorded in the
        results JSON so runs stay comparable).
    """
    from mynotes.config import Settings, get_settings

    settings = get_settings()
    applied: dict[str, Any] = {}
    for spec in specs:
        key, sep, raw = spec.partition("=")
        if not sep:
            raise SystemExit(f"--set expects dotted.key=value, got {spec!r}")
        node: Any = settings
        *parents, leaf = key.split(".")
        for part in parents:
            node = getattr(node, part, None)
            if node is None:
                raise SystemExit(f"--set: unknown settings path {key!r}")
        if not hasattr(node, leaf):
            raise SystemExit(f"--set: unknown settings key {key!r}")
        current = getattr(node, leaf)
        if isinstance(current, bool):
            value = raw.strip().lower() in {"1", "true", "yes", "on"}
        elif isinstance(current, int):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        elif current is None:
            value = yaml.safe_load(raw)  # an unset optional field: "200" -> 200, "null" -> None
        else:
            value = raw
        setattr(node, leaf, value)
        applied[key] = value
    Settings.model_validate(settings.model_dump())
    return applied


def main() -> None:
    """Parse arguments, run the selected modes and write the results JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["retrieval", "answers", "both"], default="retrieval")
    parser.add_argument("--tag", required=True, help="label for this run")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--split",
        choices=m.SPLITS,
        default="all",
        help="golden-set part: dev (tune on it), holdout (confirm only) or all",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", type=str, default=None, help="comma-separated question ids")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    parser.add_argument("--variants", action="store_true", help="also run the case/accent variants")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a settings key in-process (repeatable), "
        "e.g. --set retrieval.graph_expansion=true",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    questions = m.select_split(list(dataset["questions"]), args.split)
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [question for question in questions if question["id"] in wanted]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        raise SystemExit("No questions selected")

    overrides = apply_overrides(args.set)

    result: dict[str, Any] = {
        "tag": args.tag,
        "timestamp": datetime.now(UTC).isoformat(),
        "mode": args.mode,
        "split": args.split,
        "n_questions": len(questions),
        "git": git_info(),
        "dataset_sha256": dataset_sha256(args.dataset),
        "overrides": overrides,
        "config": config_snapshot(),
        "chunks_doc_count": chunks_doc_count(),
    }

    if args.mode in ("retrieval", "both"):
        rows = eval_retrieval(questions)
        fused_view = [
            {"question": row["question"], "metrics": row["fused_metrics"]} for row in rows
        ]
        result["retrieval"] = {
            "per_question": rows,
            "aggregate": m.aggregate(rows),
            "aggregate_fused": m.aggregate(fused_view),
        }
        if args.variants:
            result["variants"] = eval_variants(questions, rows)
    if args.mode in ("answers", "both"):
        if args.variants:
            logger.warning("--variants only affects retrieval modes; ignoring it in answers mode")
        rows = eval_answers(questions, with_judge=not args.no_judge)
        result["answers"] = {"per_question": rows, "aggregate": m.aggregate(rows)}

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{stamp}_{args.tag}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    console.print(f"[green]Results written to[/] {out_path}")
    for section in ("retrieval", "answers"):
        if section in result:
            console.print(f"--- {section} aggregate (overall):")
            console.print_json(json.dumps(result[section]["aggregate"]["overall"]))
    retrieval = result.get("retrieval")
    if retrieval:
        stages = m.funnel(
            retrieval["aggregate"]["overall"], retrieval["aggregate_fused"]["overall"]
        )
        if stages:
            console.print("--- where the expected document ends up (share of questions):")
            console.print_json(json.dumps(stages))
    for name, block in (result.get("variants") or {}).items():
        console.print(f"--- variant {name} parity:")
        console.print_json(json.dumps(block["parity"]))


if __name__ == "__main__":
    main()
