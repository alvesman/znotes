"""Run the freshness/stale-docs evaluation on the synthetic version corpus.

Everything happens in an **isolated** environment, like ``scripts/perf/run_perf.py``:
a separate database (``mynotes_eval``), workspace (``version-eval``), root and
OpenSearch aliases (``eval_chunks`` / ``eval_entities``), configured through a
temporary YAML and ``MYNOTES_*`` overrides, so the user's own notes, database
and index are never touched. Steps:

1. generate the corpus (phase 1) and ``eval/golden_set_v2.yaml``;
2. create the eval database, run ``init``, scan, and drain the worker (extract,
   chunk, embed, version detection, retention) until the index is consistent;
3. apply the in-place edits (phase 2), scan again, drain again;
4. report what version detection did to every family (linked, proposed, none);
5. run ``eval.run_eval --mode both`` once per configuration (recency boost off
   and on, by default) against the v2 golden set, as subprocesses that inherit
   the isolated environment, and print a comparison.

    uv run python scripts/eval/run_version_eval.py --reset
    uv run python scripts/eval/run_version_eval.py --skip-ingest --runs base,recency

Result JSONs land in ``eval/results/`` under the tags ``v2-<run>`` / ``v3-<run>``
(``io-`` prefix with ``--index-old-versions``).

Smoke test (``make smoke``): ``--reset --smoke --mode retrieval --no-judge --runs auto``
for both kinds. It fails when a drain times out or leaves dead jobs (both
synthetic ingests have exposed concurrency bugs that unit tests never hit) or
when retrieval doc_hit@5 drops below the kind's floor. No judge calls; the
multi-hop kind still needs the LLM for extraction.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

EVAL_DB = "mynotes_eval"
REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "eval" / "results"

KINDS: dict[str, dict[str, Any]] = {
    "version": {
        "generator": REPO / "scripts" / "eval" / "generate_version_corpus.py",
        "corpus": Path("/tmp/mynotes-version-corpus"),
        "golden": REPO / "eval" / "golden_set_v2.yaml",
        "workspace": "version-eval",
        "root": "version-corpus",
        "chunks_alias": "eval_chunks",
        "entities_alias": "eval_entities",
        "phase2": True,  # in-place edits after the first ingest
        "family_report": True,
        "smoke_floor": 0.95,  # retrieval doc_hit@5 below this fails --smoke (measured: 1.00)
    },
    "multihop": {
        "generator": REPO / "scripts" / "eval" / "generate_multihop_corpus.py",
        "corpus": Path("/tmp/mynotes-multihop-corpus"),
        "golden": REPO / "eval" / "golden_set_v3.yaml",
        "workspace": "multihop-eval",
        "root": "multihop-corpus",
        "chunks_alias": "evalmh_chunks",
        "entities_alias": "evalmh_entities",
        "phase2": False,
        "family_report": False,
        "smoke_floor": 0.65,  # measured 0.72-0.80 across runs (LLM extraction varies)
    },
    "series": {
        "generator": REPO / "scripts" / "eval" / "generate_series_corpus.py",
        "corpus": Path("/tmp/mynotes-series-corpus"),
        "golden": REPO / "eval" / "golden_set_v4.yaml",
        "workspace": "series-eval",
        "root": "series-corpus",
        "chunks_alias": "evalsr_chunks",
        "entities_alias": "evalsr_entities",
        "phase2": False,
        "family_report": False,
        "smoke_floor": 0.9,  # measured 1.00 with dates and change questions (docs/eval/series.md)
    },
}
"""The synthetic corpora this runner knows; ``--kind`` picks one."""

# Set by apply_kind() from the chosen kind (module-level so helpers can read them).
WORKSPACE = KINDS["version"]["workspace"]
ROOT_NAME = KINDS["version"]["root"]
CHUNKS_ALIAS = KINDS["version"]["chunks_alias"]
ENTITIES_ALIAS = KINDS["version"]["entities_alias"]
GENERATOR = KINDS["version"]["generator"]

JOB_TYPES = (
    "extract",
    "chunk",
    "embed",
    "version_detect",
    "link_resolve",
    "reindex_metadata",
    "delete_document",
    "apply_retention",
)
"""Everything except the LLM-heavy graph and tagging jobs."""

GRAPH_JOB_TYPES = ("kg_extract", "resolve_entities", "adjudicate_entities", "kg_temporal_refresh")
"""Added with ``--with-graph`` so graph expansion has a knowledge graph to walk."""

RUNS: dict[str, list[str]] = {
    "auto": [],  # the defaults as configured (graph_expansion auto, recency on)
    "base": ["retrieval.recency_boost=false"],
    "recency": ["retrieval.recency_boost=true", "retrieval.recency_half_life_days=365"],
    "recency-180": ["retrieval.recency_boost=true", "retrieval.recency_half_life_days=180"],
    "graph": [
        "retrieval.graph_expansion=true",
        "retrieval.graph_method=one_hop",
        "retrieval.graph_hops=1",
    ],
    "graph-ppr": [
        "retrieval.graph_expansion=true",
        "retrieval.graph_method=ppr",
        "retrieval.graph_hops=2",
    ],
}
"""Named configurations, as ``--set`` overrides for ``eval.run_eval``."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--kind",
        choices=sorted(KINDS),
        default="version",
        help="which synthetic corpus (generator, paths, workspace)",
    )
    parser.add_argument("--corpus", type=Path, default=None, help="defaults per --kind")
    parser.add_argument("--golden", type=Path, default=None, help="defaults per --kind")
    parser.add_argument(
        "--with-graph",
        action="store_true",
        help="also run the knowledge-graph jobs (LLM calls) during ingest",
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--families-per-pattern", type=int, default=7)
    parser.add_argument("--distractors", type=int, default=60)
    parser.add_argument(
        "--reset", action="store_true", help="drop the eval database and indices first"
    )
    parser.add_argument(
        "--skip-generate", action="store_true", help="reuse the corpus and golden set"
    )
    parser.add_argument("--skip-ingest", action="store_true", help="the corpus is already indexed")
    parser.add_argument("--runs", default="base,recency", help="comma-separated names from RUNS")
    parser.add_argument("--mode", choices=["retrieval", "both"], default="both")
    parser.add_argument(
        "--index-old-versions",
        action="store_true",
        help="ingest with versions.index_old_versions=true (old snapshots searchable; "
        "result tags get an 'io-' prefix) so the as-of path over indexed history is measured",
    )
    parser.add_argument(
        "--write-golden",
        action="store_true",
        help="replace the repo golden set with the generated one even if a reviewed file exists",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="smoke test: fail (exit 1) when the drain times out or leaves dead jobs, or "
        "when retrieval doc_hit@5 falls below the kind's floor (version 0.95, multihop 0.65, "
        "series 0.9)",
    )
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args(argv)
    apply_kind(args)
    return args


def apply_kind(args: argparse.Namespace) -> None:
    """Resolve per-kind defaults into ``args`` and the module-level names."""
    global WORKSPACE, ROOT_NAME, CHUNKS_ALIAS, ENTITIES_ALIAS, GENERATOR
    kind = KINDS[args.kind]
    args.corpus = args.corpus or kind["corpus"]
    args.golden = args.golden or kind["golden"]
    WORKSPACE, ROOT_NAME = kind["workspace"], kind["root"]
    CHUNKS_ALIAS, ENTITIES_ALIAS = kind["chunks_alias"], kind["entities_alias"]
    GENERATOR = kind["generator"]


# ---------------------------------------------------------------- environment


def configure_environment(corpus: Path, *, index_old_versions: bool = False) -> None:
    """Point mynotes at the eval database, workspace, root and aliases (in-process + env)."""
    import yaml
    from sqlalchemy.engine import make_url

    from mynotes.config import get_settings, reset_settings_cache

    real = get_settings()
    if real.postgres.dsn is None:
        raise SystemExit("MYNOTES_PG_DSN is not configured; see docs/infrastructure.md")
    url = make_url(real.postgres.dsn.get_secret_value())
    eval_dsn = url.set(database=EVAL_DB).render_as_string(hide_password=False)

    corpus.mkdir(parents=True, exist_ok=True)
    yaml_path = corpus / "mynotes-eval.yaml"
    settings_dict = {
        "workspace": WORKSPACE,
        "roots": [{"name": ROOT_NAME, "path": str(corpus.resolve())}],
        "scan": {"settle_seconds": 0, "delete_grace_seconds": 0},
        "tags": {"llm_autotag": False},
        "versions": {"index_old_versions": index_old_versions},
        # retrieval settings are left at the code defaults; RUNS override them per run
        "opensearch": {"chunks_alias": CHUNKS_ALIAS, "entities_alias": ENTITIES_ALIAS},
    }
    yaml_path.write_text(yaml.safe_dump(settings_dict, sort_keys=False), encoding="utf-8")
    os.environ["MYNOTES_CONFIG"] = str(yaml_path)
    os.environ["MYNOTES_PG_DSN"] = eval_dsn
    os.environ["MYNOTES_OPENSEARCH_CHUNKS_ALIAS"] = CHUNKS_ALIAS
    os.environ.pop("MYNOTES_OPENSEARCH_ENTITIES_ALIAS", None)
    reset_settings_cache()
    assert get_settings().workspace == WORKSPACE


def _admin_dsn() -> str:
    from sqlalchemy.engine import make_url

    from mynotes.config import get_settings

    url = make_url(get_settings().postgres.dsn.get_secret_value())
    return url.set(drivername="postgresql", database="postgres").render_as_string(
        hide_password=False
    )


def create_database() -> None:
    import psycopg

    with psycopg.connect(_admin_dsn(), autocommit=True) as conn:
        if (
            conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (EVAL_DB,)).fetchone()
            is None
        ):
            conn.execute(f'CREATE DATABASE "{EVAL_DB}"')
            log(f"created database {EVAL_DB}")


def drop_state() -> None:
    import psycopg

    from mynotes.index.opensearch import get_client

    with psycopg.connect(_admin_dsn(), autocommit=True) as conn:
        conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname = %s AND pid <> pg_backend_pid()",
            (EVAL_DB,),
        )
        conn.execute(f'DROP DATABASE IF EXISTS "{EVAL_DB}"')
    client = get_client()
    for alias in (CHUNKS_ALIAS, ENTITIES_ALIAS):
        for index in list(client.indices.get(index=f"{alias}_v*", ignore_unavailable=True) or {}):
            client.indices.delete(index=index)
    log("dropped the eval database and indices")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- corpus and ingest


def generate(args: argparse.Namespace, phase: str) -> None:
    command = [sys.executable, str(GENERATOR), "--out", str(args.corpus)]
    if args.kind == "series":
        command += ["--seed", str(args.seed)]
    if args.kind == "version":
        command += [
            "--seed",
            str(args.seed),
            "--families-per-pattern",
            str(args.families_per_pattern),
            "--distractors",
            str(args.distractors),
            "--phase",
            phase,
        ]
    generated = args.corpus / "golden_generated.yaml"
    if phase == "1":
        command += ["--golden-out", str(generated)]
    log(f"generate phase {phase}")
    subprocess.run(command, check=True, cwd=REPO)
    if phase == "1":
        adopt_golden(generated, Path(args.golden), overwrite=args.write_golden)


def adopt_golden(generated: Path, golden: Path, *, overwrite: bool) -> None:
    """Install the generated golden set only when the repo file has no reviewed version.

    The generators are deterministic for a seed, but their header says "not
    human-reviewed" and would overwrite the review notes a person added to
    the repo file. The questions are compared; identical questions keep the
    reviewed file, different ones are refused unless ``--write-golden``.
    """
    import yaml

    new_questions = yaml.safe_load(generated.read_text(encoding="utf-8"))["questions"]
    if golden.exists() and not overwrite:
        old_questions = yaml.safe_load(golden.read_text(encoding="utf-8"))["questions"]
        if old_questions == new_questions:
            log(f"golden set unchanged; keeping {golden.name} (reviewed header preserved)")
            return
        raise SystemExit(
            f"generated questions differ from {golden}; the reviewed file was kept. "
            f"Inspect {generated} and pass --write-golden to replace it."
        )
    golden.write_text(generated.read_text(encoding="utf-8"), encoding="utf-8")
    log(f"golden set written to {golden}")


def workspace_id_of(session: Any) -> int:
    from sqlalchemy import select

    from mynotes.db.models import Workspace

    return int(
        session.execute(select(Workspace.id).where(Workspace.name == WORKSPACE)).scalar_one()
    )


def source_root_id() -> int:
    from sqlalchemy import select

    from mynotes.db.models import SourceRoot
    from mynotes.db.session import session_scope

    with session_scope() as session:
        return int(
            session.execute(select(SourceRoot).where(SourceRoot.name == ROOT_NAME)).scalar_one().id
        )


def scan() -> dict[str, int]:
    from mynotes.scan.scanner import scan_root

    summary = scan_root(source_root_id(), kind="incremental", trigger="manual")
    log(f"scan: {summary.counts}")
    return summary.counts


class DrainIncomplete(RuntimeError):
    """The drain timed out or left dead jobs (raised only under ``--smoke``)."""


def drain(
    concurrency: int, stop_after_s: float = 1800, *, with_graph: bool = False, strict: bool = False
) -> dict[str, int]:
    """Run the worker until no job of our types is queued/running and the outbox is empty.

    Returns ``{"pending": n, "dead": n, "retried": n, "seconds": s}``; with
    ``strict`` a timeout or any dead job of our types raises
    :class:`DrainIncomplete`. Retried jobs are logged as warnings.
    """
    from sqlalchemy import text as sa_text

    import mynotes.jobs.pipeline  # noqa: F401  (registers the handlers)
    from mynotes.db.session import session_scope
    from mynotes.index.sync import sync_once
    from mynotes.jobs.worker import Worker, unregister_background_thread

    unregister_background_thread("outbox-sync")
    unregister_background_thread("scheduler")
    types = JOB_TYPES + (GRAPH_JOB_TYPES if with_graph else ())
    worker = Worker(concurrency=concurrency, types=list(types), poll_interval_s=0.2)
    worker.start()
    typed = ",".join(f"'{item}'" for item in types)
    deadline = time.monotonic() + stop_after_s
    started = time.monotonic()
    pending_jobs = 0
    try:
        while True:
            with session_scope() as session:
                pending_jobs = int(
                    session.scalar(
                        sa_text(
                            "SELECT count(*) FROM job WHERE status IN ('queued','running') "
                            f"AND type IN ({typed})"
                        )
                    )
                    or 0
                )
                pending_outbox = int(
                    session.scalar(
                        sa_text("SELECT count(*) FROM index_outbox WHERE processed_at IS NULL")
                    )
                    or 0
                )
            if pending_outbox:
                sync_once()
                continue
            if pending_jobs == 0:
                break
            if time.monotonic() > deadline:
                log(f"drain timed out with {pending_jobs} job(s) pending")
                break
            time.sleep(0.5)
    finally:
        worker.stop()
    from mynotes.index.opensearch import get_client

    get_client().indices.refresh(index=CHUNKS_ALIAS)
    with session_scope() as session:
        dead = int(
            session.scalar(
                sa_text(f"SELECT count(*) FROM job WHERE status = 'dead' AND type IN ({typed})")
            )
            or 0
        )
    with session_scope() as session:
        retried = session.execute(
            sa_text(
                "SELECT type, attempts, left(last_error, 160) FROM job "
                f"WHERE attempts > 1 AND type IN ({typed}) ORDER BY id"
            )
        ).all()
    seconds = int(time.monotonic() - started)
    log(f"drained in {seconds}s" + (f" ({dead} dead job(s))" if dead else ""))
    for job_type, attempts, error in retried:
        # A retried job finished, but the first attempt failed: on the synthetic corpora that
        # has always been a concurrency defect (deadlocks, unique violations). Loud, not fatal.
        log(f"WARNING retried job {job_type} x{attempts}: {str(error or '').splitlines()[0]}")
    if strict and (pending_jobs or dead):
        raise DrainIncomplete(f"{pending_jobs} job(s) pending, {dead} dead after {seconds}s")
    return {"pending": pending_jobs, "dead": dead, "retried": len(retried), "seconds": seconds}


# ---------------------------------------------------------------- family report


def family_report(corpus: Path) -> dict[str, Any]:
    """What version detection did to every family: linked/proposed/none and statuses."""
    from sqlalchemy import select

    from mynotes.db.models import DocRelation, Document, DocumentVersion, File
    from mynotes.db.session import session_scope

    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    with session_scope() as session:
        path_to_doc = {
            rel_path: (doc_id, status)
            for rel_path, doc_id, status in session.execute(
                select(File.rel_path, Document.id, Document.status)
                .join(Document, Document.id == File.document_id)
                .where(File.status == "active")
            ).all()
        }
        for family in manifest["families"]:
            paths = [v["rel_path"] for v in family["versions"]]
            unique_paths = list(dict.fromkeys(paths))
            docs = {p: path_to_doc.get(p) for p in unique_paths}
            doc_ids = [d[0] for d in docs.values() if d]
            relations = (
                session.execute(
                    select(DocRelation).where(
                        DocRelation.type == "supersedes",
                        DocRelation.src_document_id.in_(doc_ids),
                        DocRelation.dst_document_id.in_(doc_ids),
                    )
                )
                .scalars()
                .all()
                if doc_ids
                else []
            )
            newest_id = docs[unique_paths[-1]][0] if docs.get(unique_paths[-1]) else None
            active = [r for r in relations if r.status in ("auto", "confirmed")]
            proposed = [r for r in relations if r.status == "proposed"]
            correct = all(
                r.src_document_id == newest_id or r.dst_document_id != newest_id for r in active
            )
            snapshots = None
            if family["pattern"] == "in-place" and newest_id is not None:
                snapshots = len(
                    session.execute(
                        select(DocumentVersion.id).where(DocumentVersion.document_id == newest_id)
                    ).all()
                )
            rows.append(
                {
                    "family": family["id"],
                    "pattern": family["pattern"],
                    "language": family["language"],
                    "versions": len(unique_paths),
                    "indexed": sum(1 for d in docs.values() if d),
                    "active_links": len(active),
                    "proposed_links": len(proposed),
                    "direction_ok": correct,
                    "older_superseded": sum(
                        1 for p in unique_paths[:-1] if docs.get(p) and docs[p][1] == "superseded"
                    ),
                    "snapshots": snapshots,
                }
            )
    by_pattern: dict[str, dict[str, int]] = {}
    for row in rows:
        agg = by_pattern.setdefault(
            row["pattern"],
            {
                "families": 0,
                "fully_linked": 0,
                "proposed_only": 0,
                "unlinked": 0,
                "wrong_direction": 0,
            },
        )
        agg["families"] += 1
        expected_links = row["versions"] - 1
        if row["pattern"] in ("lookalike", "in-place"):
            expected_links = 0
        if row["active_links"] >= expected_links and expected_links > 0 and row["direction_ok"]:
            agg["fully_linked"] += 1
        elif row["active_links"] and not row["direction_ok"]:
            agg["wrong_direction"] += 1
        elif row["proposed_links"] and not row["active_links"]:
            agg["proposed_only"] += 1
        elif not row["active_links"] and not row["proposed_links"]:
            agg["unlinked"] += 1
    return {"families": rows, "by_pattern": by_pattern}


# ---------------------------------------------------------------- eval runs


def run_eval(args: argparse.Namespace, name: str) -> Path:
    overrides = RUNS[name]
    prefix = "io-" if args.index_old_versions else ""
    tag = f"{prefix}{ {'version': 'v2', 'multihop': 'v3', 'series': 'v4'}[args.kind] }-{name}"
    command = [
        sys.executable,
        "-m",
        "eval.run_eval",
        "--mode",
        args.mode,
        "--tag",
        tag,
        "--dataset",
        str(args.golden),
    ]
    if args.no_judge:
        command.append("--no-judge")
    for spec in overrides:
        command += ["--set", f"{spec}"]
    log(f"eval run {tag}: {' '.join(overrides)}")
    subprocess.run(command, check=True, cwd=REPO)
    return Path(sorted(glob.glob(str(RESULTS_DIR / f"*_{tag}.json")))[-1])


def summarize(results: dict[str, Path]) -> None:
    def pick(d: dict[str, Any], keys: tuple[str, ...]) -> str:
        return "  ".join(
            f"{k}={d[k]:.3f}"
            if isinstance(d.get(k), (int, float)) and d.get(k) is not None
            else f"{k}=-"
            for k in keys
        )

    for name, path in results.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"\n=== {name}  ({path.name})")
        if "retrieval" in data:
            agg = data["retrieval"]["aggregate"]
            print("retrieval overall:", pick(agg["overall"], ("mrr", "doc_hit@1", "doc_hit@5")))
            for slice_name, values in sorted(agg["slices"].items()):
                if slice_name.startswith(("category:", "tag:pattern:")):
                    print(
                        f"  {slice_name:28}", pick(values, ("mrr", "doc_hit@1", "doc_hit@5", "n"))
                    )
        if "answers" in data:
            agg = data["answers"]["aggregate"]
            keys = (
                "stale_answer",
                "current_version_precision",
                "source_attribution_correct",
                "judge_pass",
            )
            print("answers overall:  ", pick(agg["overall"], keys))
            for slice_name, values in sorted(agg["slices"].items()):
                if slice_name.startswith(("category:", "tag:pattern:", "tag:hops:", "tag:style:")):
                    print(f"  {slice_name:28}", pick(values, keys))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.chdir(REPO)  # .env and eval/ are resolved from the repository root
    kind = KINDS[args.kind]
    if not args.skip_generate and not args.skip_ingest:
        generate(args, "1")
    configure_environment(args.corpus, index_old_versions=args.index_old_versions)
    if args.reset:
        drop_state()
    if not args.skip_ingest:
        create_database()
        from mynotes.core.setup import run_init

        failures = [r for r in run_init(apply_migrations=True) if r.level == "fail"]
        if failures:
            raise SystemExit(f"init failed: {[f.detail for f in failures]}")
        scan()
        drain(args.concurrency, with_graph=args.with_graph, strict=args.smoke)
        if kind["phase2"]:
            generate(args, "2")
            time.sleep(1.0)
            scan()
            drain(args.concurrency, with_graph=args.with_graph, strict=args.smoke)
    if args.with_graph and not args.skip_ingest:
        from mynotes.db.session import session_scope
        from mynotes.graph.store import merge_name_variants

        with session_scope() as session:
            pairs = merge_name_variants(session, workspace_id_of(session))
            workspace_id = workspace_id_of(session)
        log(f"merged {len(pairs)} entity name variant pair(s)")
        from mynotes.config import get_settings
        from mynotes.graph.lexical import LexicalParams, relink_workspace

        lexical = LexicalParams.from_settings(get_settings())
        if lexical is not None:
            report = relink_workspace(workspace_id, lexical)
            log(f"lexical mentions: {report}")
    if kind["family_report"]:
        report = family_report(args.corpus)
        print("\n=== version detection by pattern")
        for pattern, agg in report["by_pattern"].items():
            print(f"  {pattern:13}", "  ".join(f"{k}={v}" for k, v in agg.items()))
        report_path = args.corpus / "family_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"family report written to {report_path}")
    results = {name: run_eval(args, name) for name in args.runs.split(",") if name}
    summarize(results)
    if args.smoke:
        smoke_verdict(results, float(kind["smoke_floor"]))


def smoke_verdict(results: dict[str, Path], floor: float) -> None:
    """Exit 1 when any run's retrieval doc_hit@5 is below ``floor``."""
    failures: list[str] = []
    for name, path in results.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        hit = float(data["retrieval"]["aggregate"]["overall"]["doc_hit@5"])
        status = "ok" if hit >= floor else "FAIL"
        log(f"smoke {name}: doc_hit@5={hit:.3f} floor={floor:.2f} {status}")
        if hit < floor:
            failures.append(f"{name}: doc_hit@5 {hit:.3f} < {floor:.2f}")
    if failures:
        raise SystemExit("SMOKE FAILED: " + "; ".join(failures))
    log("smoke passed")


if __name__ == "__main__":
    main()
