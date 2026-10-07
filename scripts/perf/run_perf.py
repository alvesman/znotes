#!/usr/bin/env python3
"""M7 perf: 20k-file measurement harness (task 51).

Runs the real pipeline (scanner, worker, outbox sync, retrieval) against a
**separate** database (``mynotes_perf``) and index prefix (``perf_chunks``),
with the source root pointing at the synthetic corpus from
``generate_corpus.py``. Phases, each timed:

1. initial scan -> files/s;
2. initial processing (worker until no extract/chunk/embed jobs and the
   outbox is drained) -> docs/min, chunks/min, embedding calls, cache hits
   (KG extraction is excluded; measured separately by the ``kg`` phase);
3. no-change incremental scan -> wall time and SQL statement count (proves
   the scanner loads known files with one set-based query);
4. incremental after changes (50 modified, 20 renamed, 10 deleted, 20 new)
   -> time to fully consistent (scan + process + sync);
5. query latency: 50 queries x 3 runs -> p50/p95 for ``retrieve`` with and
   without rerank, and ``ask`` (real LLM) on 10 questions;
6. resources: OpenSearch index/heap, Postgres DB size, peak worker RSS.

Embeddings: the full 20k run needs about 300k real embedding calls, so
``--fake-embeddings`` swaps in a deterministic 4096-dim fake (pipeline
throughput only); run the real endpoint on a 2,000-file subset (a fresh
corpus + ``--reset``) to extrapolate. The mode is recorded in the results.

Usage (see docs/perf/m7-report.md for the recorded runs):

    uv run python scripts/perf/run_perf.py --corpus /tmp/mynotes-perf \
        --fake-embeddings --reset
    uv run python scripts/perf/run_perf.py --corpus /tmp/mynotes-perf \
        --phases 5 --json-out /tmp/mynotes-perf/query-latency.json

Results are printed as a JSON document and written to ``--json-out``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

PERF_DB = "mynotes_perf"
PERF_WORKSPACE = "perf"
PERF_ROOT_NAME = "perf"
FAKE_MODEL_PREFIX = "fake:"


@dataclass
class PhaseResult:
    """Timing and counters of one measured phase."""

    phase: str
    seconds: float
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerfResults:
    """The full harness outcome (serialized to JSON)."""

    hardware: dict[str, Any] = field(default_factory=dict)
    versions: dict[str, Any] = field(default_factory=dict)
    mode: str = "real"
    corpus: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    phases: list[PhaseResult] = field(default_factory=list)
    extrapolations: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def add(self, phase: str, started: float, **metrics: Any) -> PhaseResult:
        """Append one timed phase result and return it."""
        result = PhaseResult(
            phase=phase, seconds=round(time.perf_counter() - started, 3), metrics=metrics
        )
        self.phases.append(result)
        return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags."""
    parser = argparse.ArgumentParser(description="20k-file performance measurement (task 51)")
    parser.add_argument("--corpus", type=Path, required=True, help="Synthetic corpus root.")
    parser.add_argument("--files", type=int, default=0, help="Expected file count (check only).")
    parser.add_argument(
        "--fake-embeddings",
        action="store_true",
        help="Use the deterministic 4096-dim fake embedder instead of the real endpoint.",
    )
    parser.add_argument(
        "--phases",
        type=str,
        default="1,2,3,4,6",
        help="Comma-separated phases to run: 1,2,3,4,5,6,kg (default: 1,2,3,4,6).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=8, help="Worker job threads (default 8)."
    )
    parser.add_argument(
        "--kg-sample", type=int, default=200, help="Documents for the KG throughput phase."
    )
    parser.add_argument(
        "--kg-concurrency", type=int, default=8, help="Concurrent KG extraction calls."
    )
    parser.add_argument("--queries", type=int, default=50, help="Query-latency query count.")
    parser.add_argument("--query-runs", type=int, default=3, help="Runs per query.")
    parser.add_argument("--ask-questions", type=int, default=10, help="Questions for ask().")
    parser.add_argument("--ask-runs", type=int, default=1, help="Runs per ask() question.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the perf database and perf indices before starting.",
    )
    parser.add_argument(
        "--json-out", type=Path, default=None, help="Where to write the JSON results."
    )
    return parser.parse_args(argv)


def configure_environment(args: argparse.Namespace) -> None:
    """Point mynotes at the perf database, workspace, root and aliases.

    Reads the real settings once (secrets through the config module), derives
    the ``mynotes_perf`` DSN from the configured one, writes a temporary perf
    YAML and sets the ``MYNOTES_*`` overrides before mynotes caches anything.
    Secret values are never printed.
    """
    from sqlalchemy.engine import make_url

    from mynotes.config import get_settings, reset_settings_cache

    real = get_settings()
    if real.postgres.dsn is None:
        raise SystemExit("MYNOTES_PG_DSN is not configured; see docs/infrastructure.md")
    url = make_url(real.postgres.dsn.get_secret_value())
    perf_dsn = url.set(database=PERF_DB).render_as_string(hide_password=False)

    yaml_path = args.corpus / "mynotes-perf.yaml"
    settings_dict = {
        "workspace": PERF_WORKSPACE,
        "roots": [{"name": PERF_ROOT_NAME, "path": str(args.corpus.resolve())}],
        "tags": {"llm_autotag": False},
        "opensearch": {
            "chunks_alias": "perf_chunks",
            "entities_alias": "perf_entities",
        },
    }
    import yaml

    yaml_path.write_text(yaml.safe_dump(settings_dict, sort_keys=False), encoding="utf-8")

    os.environ["MYNOTES_CONFIG"] = str(yaml_path)
    os.environ["MYNOTES_PG_DSN"] = perf_dsn
    os.environ["MYNOTES_OPENSEARCH_CHUNKS_ALIAS"] = "perf_chunks"
    os.environ.pop("MYNOTES_OPENSEARCH_ENTITIES_ALIAS", None)
    reset_settings_cache()


def _psycopg_admin_dsn() -> str:
    """The admin (postgres-db) connection string for psycopg (driverless URL)."""
    from sqlalchemy.engine import make_url

    from mynotes.config import get_settings

    url = make_url(get_settings().postgres.dsn.get_secret_value())
    return url.set(drivername="postgresql", database="postgres").render_as_string(
        hide_password=False
    )


def create_perf_database() -> None:
    """Create ``mynotes_perf`` from the configured DSN when it is missing."""
    import psycopg

    with psycopg.connect(_psycopg_admin_dsn(), autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (PERF_DB,)).fetchone()
        if exists is None:
            conn.execute(f'CREATE DATABASE "{PERF_DB}"')
            print(f"created database {PERF_DB}", file=sys.stderr)


def drop_perf_state() -> None:
    """Drop the perf database and the perf indices (``--reset``)."""
    import psycopg

    from mynotes.index.opensearch import get_client

    with psycopg.connect(_psycopg_admin_dsn(), autocommit=True) as conn:
        conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname = %s AND pid <> pg_backend_pid()",
            (PERF_DB,),
        )
        conn.execute(f'DROP DATABASE IF EXISTS "{PERF_DB}"')
    client = get_client()
    for index in ("perf_chunks_v1", "perf_entities_v1"):
        if client.indices.exists(index=index):
            client.indices.delete(index=index)
    print("dropped perf database and indices", file=sys.stderr)


def prepare(args: argparse.Namespace, results: PerfResults) -> None:
    """Run migrations, create the workspace/root and ensure indices/pipeline."""
    from mynotes.core.setup import run_init

    if args.reset:
        drop_perf_state()
    create_perf_database()
    failures = [result for result in run_init(apply_migrations=True) if result.level == "fail"]
    if failures:
        raise SystemExit(f"init failed: {[f.detail for f in failures]}")
    _wait_for_settle(args.corpus)
    results.corpus = _corpus_stats(args.corpus)
    results.config = {
        "workspace": PERF_WORKSPACE,
        "database": PERF_DB,
        "chunks_alias": "perf_chunks",
        "worker_concurrency": args.concurrency,
        "worker_types": sorted(_PERF_JOB_TYPES),
        "llm_autotag": False,
        "kg_extraction": "excluded from phases 2/4 (measured by the kg phase)",
    }


def _wait_for_settle(corpus: Path) -> None:
    """Sleep until the newest corpus file is older than ``scan.settle_seconds``.

    A freshly generated corpus would otherwise be classified ``unsettled`` by
    the first scan (the scanner skips files younger than the settle window).
    """
    from mynotes.config import get_settings

    settle = get_settings().scan.settle_seconds
    newest = max((path.stat().st_mtime for path in corpus.rglob("*") if path.is_file()), default=0)
    remaining = newest + settle + 1.0 - time.time()
    if remaining > 0:
        print(f"waiting {remaining:.0f}s for the corpus to settle", file=sys.stderr)
        time.sleep(remaining)


def _corpus_stats(corpus: Path) -> dict[str, Any]:
    """Count corpus files per extension and total bytes."""
    counts: dict[str, int] = {}
    total_bytes = 0
    for path in corpus.rglob("*"):
        if path.is_file() and path.name != "mynotes-perf.yaml":
            extension = path.suffix.lstrip(".") or "none"
            counts[extension] = counts.get(extension, 0) + 1
            total_bytes += path.stat().st_size
    return {"files": sum(counts.values()), "bytes": total_bytes, "by_extension": counts}


_PERF_JOB_TYPES = (
    "extract",
    "chunk",
    "embed",
    "version_detect",
    "link_resolve",
    "reindex_metadata",
    "delete_document",
    "apply_retention",
)

# Phase 2's completion condition (task 51): no extract/chunk/embed jobs and
# the outbox drained. The lower-priority follow-ups (version_detect,
# link_resolve, apply_retention) keep running afterwards like in production.
_TIMED_JOB_TYPES = ("extract", "chunk", "embed")

# Phase 4's "fully consistent": the timed types plus the index-side effects of
# renames (reindex_metadata) and deletions (delete_document).
_CONSISTENCY_JOB_TYPES = ("extract", "chunk", "embed", "reindex_metadata", "delete_document")


class _EmbedCounter:
    """Counts embedding calls: requests, input texts and distinct texts.

    Real mode counts through a patched ``EmbeddingClient._embed_batch`` (the
    distinct-text set is fine at the 2,000-file subset size). Fake mode counts
    through the fake embedder's own counters; distinct inputs are then read
    from the ``embedding_cache`` row count (one row per distinct text).
    """

    def __init__(self, fake_embedder: _FakeEmbedder | None = None) -> None:
        self.requests = 0
        self.inputs = 0
        self.texts: set[str] = set()
        self.fake_embedder = fake_embedder

    @property
    def total_requests(self) -> int:
        """All embedder calls so far, real or fake."""
        fake = self.fake_embedder.requests if self.fake_embedder else 0
        return self.requests + fake

    @property
    def total_inputs(self) -> int:
        """All input texts sent so far, real or fake."""
        fake = self.fake_embedder.inputs if self.fake_embedder else 0
        return self.inputs + fake

    @property
    def unique_inputs(self) -> int:
        """Distinct texts embedded so far (cache rows in fake mode)."""
        if self.fake_embedder is not None:
            from sqlalchemy import text as sa_text

            from mynotes.db.session import session_scope

            with session_scope() as session:
                return int(session.scalar(sa_text("SELECT count(*) FROM embedding_cache")) or 0)
        return len(self.texts)


def _install_embed_counter(fake: bool) -> _EmbedCounter:
    """Patch the embedding clients so every request is counted.

    Real mode wraps ``EmbeddingClient._embed_batch``; fake mode also patches
    the worker's ``JobContext`` factory to hand handlers a fake 4096-dim
    embedder whose model name matches the settings (so the Postgres cache and
    the outbox document builder agree).
    """
    from mynotes.config import get_settings
    from mynotes.embed.client import EmbeddingClient

    counter = _EmbedCounter()
    original = EmbeddingClient._embed_batch

    def counted(self: EmbeddingClient, batch: list[str]) -> list[list[float]]:
        counter.requests += 1
        counter.inputs += len(batch)
        counter.texts.update(batch)
        return original(self, batch)

    EmbeddingClient._embed_batch = counted  # type: ignore[method-assign]
    fake_embedder = None
    if fake:
        import mynotes.jobs.worker as worker_module
        from mynotes.jobs.registry import JobContext as _JobContext

        settings = get_settings()
        fake_embedder = _FakeEmbedder(dim=settings.models.embedding.dim)
        fake_embedder.model = settings.models.embedding.model

        def context_factory(worker_id: str, settings: Any) -> Any:
            return _JobContext(worker_id=worker_id, settings=settings, embedder=fake_embedder)

        worker_module.JobContext = context_factory  # type: ignore[assignment]
    counter.fake_embedder = fake_embedder
    return counter


class _FakeEmbedder:
    """Deterministic 4096-dim hashing embedder (same trick as tests/fakes.py)."""

    def __init__(self, dim: int) -> None:
        import hashlib
        import math

        self._hashlib = hashlib
        self._math = math
        self.dim = dim
        self.model = "fake-embedder"
        self.requests = 0
        self.inputs = 0

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in text.split():
            digest = self._hashlib.blake2b(token.lower().encode("utf-8")).hexdigest()
            vector[int(digest, 16) % self.dim] += 1.0
        norm = self._math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents in input order (counted)."""
        self.requests += 1
        self.inputs += len(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str, task: str = "retrieval") -> list[float]:
        """Embed one query (counted)."""
        return self._vector(text)


class _RssMonitor:
    """Samples this process's RSS in a thread and records the peak."""

    def __init__(self, interval_s: float = 0.5) -> None:
        import psutil

        self._process = psutil.Process()
        self._interval_s = interval_s
        self._stop = threading.Event()
        self.peak_rss_bytes = 0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start sampling."""
        self._thread = threading.Thread(target=self._loop, name="rss-monitor", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self._process.memory_info().rss
                if rss > self.peak_rss_bytes:
                    self.peak_rss_bytes = rss
            except Exception:  # pragma: no cover - process gone
                return
            self._stop.wait(self._interval_s)

    def stop(self) -> int:
        """Stop sampling and return the peak RSS in bytes."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.peak_rss_bytes


def _start_worker(concurrency: int):
    """Start a worker restricted to the perf job types (no background threads)."""
    from mynotes.jobs.worker import Worker, unregister_background_thread

    unregister_background_thread("outbox-sync")
    unregister_background_thread("scheduler")
    worker = Worker(concurrency=concurrency, types=list(_PERF_JOB_TYPES), poll_interval_s=0.2)
    worker.start()
    return worker


def _drain_until_consistent(
    types: tuple[str, ...] = _TIMED_JOB_TYPES, stop_after_s: float | None = None
) -> dict[str, int]:
    """Loop until the given job types are quiet and the outbox is drained.

    Returns the sync totals observed: upserts, deletes and pending jobs left.
    """
    from sqlalchemy import text as sa_text

    from mynotes.db.session import session_scope
    from mynotes.index.sync import sync_once

    typed = ",".join(f"'{item}'" for item in types)
    totals = {"upserts": 0, "deletes": 0}
    deadline = time.monotonic() + stop_after_s if stop_after_s else None
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
            stats = sync_once()
            totals["upserts"] += stats.upserts
            totals["deletes"] += stats.deletes
            continue
        if pending_jobs == 0:
            break
        if deadline is not None and time.monotonic() > deadline:
            break
        time.sleep(0.5)
    return {**totals, "pending_jobs_left": 0}


def _queue_counts(types: tuple[str, ...] = _PERF_JOB_TYPES) -> dict[str, int]:
    """Jobs per status for the given job types."""
    from sqlalchemy import text as sa_text

    from mynotes.db.session import session_scope

    typed = ",".join(f"'{item}'" for item in types)
    with session_scope() as session:
        rows = session.execute(
            sa_text(f"SELECT status, count(*) FROM job WHERE type IN ({typed}) GROUP BY status")
        ).all()
    return {status: int(count) for status, count in rows}


def _source_root_id() -> int:
    """The perf source root id."""
    from sqlalchemy import select

    from mynotes.db.models import SourceRoot
    from mynotes.db.session import session_scope

    with session_scope() as session:
        row = session.execute(
            select(SourceRoot).where(SourceRoot.name == PERF_ROOT_NAME)
        ).scalar_one()
        return int(row.id)


def phase_1_initial_scan(results: PerfResults) -> None:
    """Phase 1: initial incremental scan of the empty database."""
    from mynotes.scan.scanner import scan_root

    root_id = _source_root_id()
    started = time.perf_counter()
    summary = scan_root(root_id, kind="incremental", trigger="manual")
    files = summary.counts.get("new", 0)
    results.add(
        "1-initial-scan",
        started,
        files=files,
        files_per_s=round(files / max(time.perf_counter() - started, 1e-6), 1),
        counts=summary.counts,
        errors=summary.errors,
    )


def phase_2_initial_processing(
    results: PerfResults, args: argparse.Namespace, embeds: _EmbedCounter
) -> None:
    """Phase 2: worker + outbox sync until fully consistent."""
    from sqlalchemy import text as sa_text

    from mynotes.db.session import session_scope

    rss = _RssMonitor()
    rss.start()
    worker = _start_worker(args.concurrency)
    started = time.perf_counter()
    try:
        drained = _drain_until_consistent()
    finally:
        worker.stop()
        peak_rss = rss.stop()
    elapsed = time.perf_counter() - started
    with session_scope() as session:
        documents = int(session.scalar(sa_text("SELECT count(*) FROM document")) or 0)
        versions = int(session.scalar(sa_text("SELECT count(*) FROM document_version")) or 0)
        chunks = int(session.scalar(sa_text("SELECT count(*) FROM chunk")) or 0)
    minutes = max(elapsed / 60.0, 1e-9)
    results.add(
        "2-initial-processing",
        started,
        documents=documents,
        versions=versions,
        chunks=chunks,
        docs_per_min=round(documents / minutes, 1),
        chunks_per_min=round(chunks / minutes, 1),
        embedding_requests=embeds.total_requests,
        embedding_inputs=embeds.total_inputs,
        embedding_unique_inputs=embeds.unique_inputs,
        cache_hits=max(chunks - embeds.total_inputs, 0),
        sync_upserts=drained["upserts"],
        sync_deletes=drained["deletes"],
        peak_worker_rss_mb=round(peak_rss / (1024 * 1024), 1),
        queue_counts=_queue_counts(),
        followup_lag=_queue_counts(("version_detect", "link_resolve", "apply_retention")),
    )


def _count_sql_statements(engine):
    """Return (counter dict, event listener removal) for SQL statements."""
    from sqlalchemy import event

    counter = {"statements": 0, "select": 0, "insert": 0, "update": 0, "delete": 0}

    def before_cursor_execute(conn, cursor, statement, params, context, executemany):
        verb = statement.lstrip().split(" ", 1)[0].lower()
        counter["statements"] += 1
        if verb in counter:
            counter[verb] += 1

    event.listen(engine, "before_cursor_execute", before_cursor_execute)

    def remove():
        event.remove(engine, "before_cursor_execute", before_cursor_execute)

    return counter, remove


def phase_3_no_change_scan(results: PerfResults) -> None:
    """Phase 3: no-change incremental scan; counts SQL statements."""
    from mynotes.db.session import get_engine
    from mynotes.scan.scanner import scan_root

    root_id = _source_root_id()
    counter, remove = _count_sql_statements(get_engine())
    before_jobs = _queue_counts().get("queued", 0) + _queue_counts().get("running", 0)
    started = time.perf_counter()
    try:
        summary = scan_root(root_id, kind="incremental", trigger="manual")
    finally:
        remove()
    elapsed = time.perf_counter() - started
    after_jobs = _queue_counts().get("queued", 0) + _queue_counts().get("running", 0)
    results.add(
        "3-no-change-scan",
        started,
        seconds=round(elapsed, 3),
        counts=summary.counts,
        sql_statements=counter["statements"],
        sql_selects=counter["select"],
        new_jobs=after_jobs - before_jobs,
        errors=summary.errors,
    )


def phase_4_incremental_changes(results: PerfResults, args: argparse.Namespace) -> None:
    """Phase 4: modify/rename/delete/add files, then measure scan+process+sync."""
    from sqlalchemy import text as sa_text

    from mynotes.db.session import session_scope
    from mynotes.scan.scanner import scan_root

    root = Path(get_settings_roots()[PERF_ROOT_NAME])
    md_files = sorted(root.rglob("*.md"))
    if len(md_files) < 120:
        raise SystemExit("corpus too small for phase 4")
    rng = __import__("random").Random(1234)
    modify = rng.sample(md_files, 90)
    to_modify, to_rename = modify[:50], modify[50:70]
    to_delete = modify[70:80]
    additions_dir = root / "projects" / "alfa" / "planos" / "2026"
    additions_dir.mkdir(parents=True, exist_ok=True)

    rss = _RssMonitor()
    rss.start()
    worker = _start_worker(args.concurrency)
    scan_seconds = 0.0
    scans = 0
    started = time.perf_counter()
    try:
        for path in to_modify:
            path.write_text(
                path.read_text(encoding="utf-8")
                + "\n## Alteração de desempenho\n\nO índice foi reconstruído e a "
                "latência de consulta melhorou depois da auditoria.\n",
                encoding="utf-8",
            )
        for path in to_rename:
            path.rename(path.with_name(path.stem + "-renomeado" + path.suffix))
        for path in to_delete:
            path.unlink()
        for index in range(20):
            (additions_dir / f"nota-nova-{index}.md").write_text(
                "# Nota nova\n\nConteúdo sintético para a fase incremental do teste de "
                "desempenho com o índice híbrido e o grafo de conhecimento.\n",
                encoding="utf-8",
            )
        time.sleep(12)  # scan.settle_seconds (10) + margin
        from mynotes.config import get_settings

        grace_deadline = time.monotonic() + get_settings().scan.delete_grace_seconds + 2.0
        consistent = False
        while not consistent:
            scan_started = time.perf_counter()
            scan_root(_source_root_id(), kind="incremental", trigger="manual")
            scan_seconds += time.perf_counter() - scan_started
            scans += 1
            _drain_until_consistent(types=_CONSISTENCY_JOB_TYPES)
            with session_scope() as session:
                missing = int(
                    session.scalar(sa_text("SELECT count(*) FROM file WHERE status = 'missing'"))
                    or 0
                )
            consistent = missing == 0
            if not consistent:
                remaining = grace_deadline - time.monotonic()
                time.sleep(max(remaining, 5.0))
    finally:
        worker.stop()
        peak_rss = rss.stop()
    elapsed = time.perf_counter() - started
    results.add(
        "4-incremental-changes",
        started,
        wall_seconds=round(elapsed, 3),
        processing_seconds=round(elapsed - 12.0, 3),
        scan_seconds=round(scan_seconds, 3),
        scans=scans,
        modified=50,
        renamed=20,
        deleted=10,
        added=20,
        peak_worker_rss_mb=round(peak_rss / (1024 * 1024), 1),
        counts=_queue_counts(),
    )


def get_settings_roots() -> dict[str, str]:
    """Configured root name -> absolute path."""
    from mynotes.config import get_settings

    return {root.name: str(Path(root.path).expanduser().resolve()) for root in get_settings().roots}


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of a sorted list."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = max(1, round(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def phase_5_query_latency(results: PerfResults, args: argparse.Namespace) -> None:
    """Phase 5: 50 queries x runs for retrieve (with/without rerank) + ask."""
    import generate_corpus as vocab

    from mynotes.retrieval.answer import answer
    from mynotes.retrieval.pipeline import retrieve
    from mynotes.retrieval.types import SearchFilters

    rng = __import__("random").Random(99)
    projects = vocab.PROJECTS
    pt_words = [word for word in vocab.PT_WORDS if " " not in word]
    en_words = [word for word in vocab.EN_WORDS if " " not in word]
    tags = [f"project:{name}" for name in projects] + [f"area:{name}" for name in vocab.AREAS[:3]]
    queries: list[tuple[str, str]] = []
    for _ in range(args.queries // 2):
        queries.append(
            (
                "keyword",
                " ".join(rng.sample(pt_words + en_words, rng.randint(1, 3))),
            )
        )
    for _ in range(args.queries // 3):
        queries.append(
            (
                "semantic",
                rng.choice(
                    [
                        "Como está planeado o {w} do projeto {p}?",
                        "What are the risks for the {w} work?",
                        "Quais os resultados do experimento sobre {w}?",
                        "What did the team decide about the {w}?",
                    ]
                ).format(w=rng.choice(pt_words + en_words), p=rng.choice(projects)),
            )
        )
    for _ in range(args.queries // 6):
        queries.append(("filtered", f"#{rng.choice(tags)} {rng.choice(pt_words + en_words)}"))
    for _ in range(args.queries - len(queries)):
        queries.append(
            (
                "history",
                rng.choice(
                    [
                        "O que mudou no {w} desde março?",
                        "What changed in the {w} plan?",
                        "Qual é a versão mais recente do {w}?",
                    ]
                ).format(w=rng.choice(pt_words + en_words)),
            )
        )
    rng.shuffle(queries)

    no_rerank_ms: list[float] = []
    rerank_ms: list[float] = []
    stage_sums: dict[str, list[int]] = {}
    for _ in range(args.query_runs):
        for _kind, query in queries:
            result = retrieve(query, SearchFilters(), rerank=False)
            no_rerank_ms.append(sum(result.timings_ms.values()))
            for stage, value in result.timings_ms.items():
                stage_sums.setdefault(f"no_rerank_{stage}", []).append(value)
            result = retrieve(query, SearchFilters(), rerank=True)
            rerank_ms.append(sum(result.timings_ms.values()))
            for stage, value in result.timings_ms.items():
                stage_sums.setdefault(f"rerank_{stage}", []).append(value)

    ask_results: list[dict[str, Any]] = []
    ask_questions = [query for kind, query in queries if kind in ("semantic", "history")][
        : args.ask_questions
    ]
    while len(ask_questions) < args.ask_questions:
        ask_questions.append(queries[len(ask_questions)][1])
    for question in ask_questions:
        for _ in range(args.ask_runs):
            started = time.perf_counter()
            try:
                outcome = answer(question)
                ask_results.append(
                    {
                        "question": question,
                        "seconds": round(time.perf_counter() - started, 3),
                        "relevant": outcome.relevant,
                        "citations": len(outcome.citations),
                    }
                )
            except Exception as exc:
                ask_results.append(
                    {
                        "question": question,
                        "seconds": round(time.perf_counter() - started, 3),
                        "error": f"{type(exc).__name__}: {exc}"[:200],
                    }
                )

    ask_seconds = [item["seconds"] for item in ask_results if "error" not in item]
    metrics = {
        "queries": len(queries),
        "runs_per_query": args.query_runs,
        "retrieve_no_rerank_p50_ms": round(_percentile(no_rerank_ms, 0.50), 1),
        "retrieve_no_rerank_p95_ms": round(_percentile(no_rerank_ms, 0.95), 1),
        "retrieve_rerank_p50_ms": round(_percentile(rerank_ms, 0.50), 1),
        "retrieve_rerank_p95_ms": round(_percentile(rerank_ms, 0.95), 1),
        "ask_p50_s": round(_percentile(ask_seconds, 0.50), 2) if ask_seconds else None,
        "ask_p95_s": round(_percentile(ask_seconds, 0.95), 2) if ask_seconds else None,
        "ask_errors": sum(1 for item in ask_results if "error" in item),
        "ask_results": ask_results,
        "stage_p50_ms": {
            name: round(_percentile([float(v) for v in values], 0.50), 1)
            for name, values in stage_sums.items()
        },
    }
    started = time.perf_counter()
    results.add("5-query-latency", started, **metrics)


def phase_6_resources(results: PerfResults) -> None:
    """Phase 6: OpenSearch index/heap, Postgres DB size, worker RSS."""
    from sqlalchemy import text as sa_text

    from mynotes.db.session import session_scope
    from mynotes.index.opensearch import get_client

    client = get_client()
    indices = {
        row["index"]: {"docs": row.get("docs.count"), "store": row.get("store.size")}
        for row in client.cat.indices(format="json", s="index")
        if row["index"].startswith("perf_")
    }
    heap = {}
    for _node_id, node in client.nodes.stats(metric="jvm")["nodes"].items():
        mem = node["jvm"]["mem"]
        heap = {
            "heap_used_gb": round(mem["heap_used_in_bytes"] / 1e9, 2),
            "heap_max_gb": round(mem["heap_max_in_bytes"] / 1e9, 2),
        }
    with session_scope() as session:
        db_size = int(session.scalar(sa_text("SELECT pg_database_size(current_database())")) or 0)
        tables = {
            name: int(total)
            for name, total in session.execute(
                sa_text(
                    "SELECT relname, pg_total_relation_size(relid) AS total "
                    "FROM pg_catalog.pg_statio_user_tables ORDER BY total DESC LIMIT 8"
                )
            ).all()
        }
    import psutil

    rss_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    results.add(
        "6-resources",
        time.perf_counter(),
        opensearch_indices=indices,
        opensearch_heap=heap,
        postgres_db_bytes=db_size,
        postgres_top_tables=tables,
        harness_rss_mb=rss_mb,
    )


def phase_kg_sample(results: PerfResults, args: argparse.Namespace) -> None:
    """KG phase: extract a document sample through the real LLM, extrapolate."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from sqlalchemy import select

    from mynotes.config import get_settings
    from mynotes.db.models import DocumentVersion
    from mynotes.db.session import session_scope
    from mynotes.jobs.pipeline import handle_kg_extract
    from mynotes.jobs.registry import JobContext

    with session_scope() as session:
        version_ids = list(
            session.execute(
                select(DocumentVersion.id)
                .where(DocumentVersion.is_latest.is_(True), DocumentVersion.status == "embedded")
                .order_by(DocumentVersion.id)
                .limit(args.kg_sample)
            ).scalars()
        )
    if not version_ids:
        raise SystemExit("no embedded versions to sample; run phases 1 and 2 first")

    context = JobContext(worker_id="perf-kg", settings=get_settings())
    started = time.perf_counter()
    failures = 0
    with ThreadPoolExecutor(max_workers=args.kg_concurrency) as pool:
        futures = {
            pool.submit(handle_kg_extract, {"version_id": version_id}, context): version_id
            for version_id in version_ids
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                failures += 1
    elapsed = time.perf_counter() - started

    from sqlalchemy import text as sa_text

    from mynotes.db.session import session_scope as scope

    with scope() as session:
        cache_rows = int(session.scalar(sa_text("SELECT count(*) FROM kg_extraction_cache")) or 0)
        entities = int(session.scalar(sa_text("SELECT count(*) FROM kg_entity")) or 0)
        relations = int(session.scalar(sa_text("SELECT count(*) FROM kg_relation")) or 0)
    documents_total = _document_count()
    share = documents_total / max(len(version_ids), 1)
    results.add(
        "kg-sample",
        started,
        documents_sampled=len(version_ids),
        kg_concurrency=args.kg_concurrency,
        seconds=round(elapsed, 3),
        docs_per_min=round(len(version_ids) / max(elapsed / 60.0, 1e-9), 1),
        failures=failures,
        extraction_cache_rows=cache_rows,
        entities=entities,
        relations=relations,
        extrapolated_full_backfill_days=round(elapsed * share / 86400.0, 2),
    )


def _document_count() -> int:
    from sqlalchemy import text as sa_text

    from mynotes.db.session import session_scope

    with session_scope() as session:
        return int(session.scalar(sa_text("SELECT count(*) FROM document")) or 0)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    if not args.corpus.is_dir():
        raise SystemExit(f"corpus directory not found: {args.corpus}")
    configure_environment(args)

    import mynotes.logging_setup as logging_setup

    logging_setup.configure_logging()

    results = PerfResults(
        hardware=_hardware(),
        versions=_versions(),
        mode="fake-embeddings" if args.fake_embeddings else "real-embeddings",
    )
    prepare(args, results)
    if args.files and results.corpus.get("files") not in (args.files, args.files + 1):
        results.notes.append(
            f"corpus has {results.corpus.get('files')} files, expected {args.files}"
        )

    phases = {item.strip() for item in args.phases.split(",") if item.strip()}
    embeds = _install_embed_counter(args.fake_embeddings)
    if "1" in phases:
        phase_1_initial_scan(results)
    if "2" in phases:
        phase_2_initial_processing(results, args, embeds)
    if "3" in phases:
        phase_3_no_change_scan(results)
    if "4" in phases:
        phase_4_incremental_changes(results, args)
    if "5" in phases:
        phase_5_query_latency(results, args)
    if "kg" in phases:
        phase_kg_sample(results, args)
    if "6" in phases:
        phase_6_resources(results)

    _add_extrapolations(results, args)
    payload = asdict(results)
    text = json.dumps(payload, indent=2, default=str)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    return 0


def _add_extrapolations(results: PerfResults, args: argparse.Namespace) -> None:
    """Derive full-scale projections from the measured phases."""
    phase_by_name = {phase.phase: phase for phase in results.phases}
    processing = phase_by_name.get("2-initial-processing")
    if processing:
        chunks = processing.metrics.get("chunks", 0)
        docs = processing.metrics.get("documents", 0)
        results.extrapolations["fake_full_scale"] = {
            "documents": docs,
            "chunks": chunks,
            "processing_seconds": processing.seconds,
        }
    kg = phase_by_name.get("kg-sample")
    if kg:
        results.extrapolations["kg_backfill"] = {
            "sampled_documents": kg.metrics.get("documents_sampled"),
            "docs_per_min": kg.metrics.get("docs_per_min"),
            "full_scale_days": kg.metrics.get("extrapolated_full_backfill_days"),
        }


def _hardware() -> dict[str, Any]:
    """CPU, memory, platform and disk of the measurement machine."""
    import platform

    import psutil

    vm = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cpu_model": _cpu_model(),
        "ram_total_gb": round(vm.total / 1e9, 1),
        "disk_free_gb": round(disk.free / 1e9, 1),
    }


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _versions() -> dict[str, Any]:
    """Software versions recorded for the report."""
    from importlib.metadata import version

    result = {}
    for package in ("mynotes", "sqlalchemy", "psycopg", "opensearch-py", "pymupdf"):
        with contextlib.suppress(Exception):
            result[package] = version(package)
    try:
        from mynotes.index.opensearch import get_client

        result["opensearch"] = get_client().info()["version"]["number"]
        result["postgres"] = _postgres_version()
    except Exception:
        pass
    return result


def _postgres_version() -> str | None:
    from sqlalchemy import text as sa_text

    from mynotes.db.session import session_scope

    try:
        with session_scope() as session:
            return str(session.scalar(sa_text("SHOW server_version")))
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
