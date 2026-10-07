"""Command-line interface for mynotes."""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mynotes import __version__
from mynotes.core.setup import CheckResult, check_endpoints, collect_status, run_init
from mynotes.logging_setup import configure_logging

if TYPE_CHECKING:
    from mynotes.retrieval.assemble import Citation
    from mynotes.retrieval.types import SearchFilters, SearchHit

console = Console()

logger = logging.getLogger(__name__)

_LEVEL_STYLES = {"ok": "green", "warn": "yellow", "fail": "bold red"}


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"mynotes {__version__}")
        raise typer.Exit()


def _setup_logging(verbose: bool) -> None:
    """Configure logging; fall back to ``basicConfig`` when settings are broken."""
    try:
        configure_logging(level="DEBUG" if verbose else "INFO")
    except Exception:
        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, stream=sys.stderr)


app = typer.Typer(
    help="mynotes: RAG + knowledge graph over your notes folder.", no_args_is_help=True
)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Log at DEBUG level."),
) -> None:
    """mynotes command-line entry point."""
    _setup_logging(verbose)


def _results_table(results: Sequence[CheckResult], title: str) -> Table:
    """Render check results as a Rich table (check, result, detail)."""
    table = Table(title=title)
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Result")
    table.add_column("Detail", overflow="fold")
    for result in results:
        style = _LEVEL_STYLES[result.level]
        table.add_row(result.name, f"[{style}]{result.level}[/]", result.detail)
    return table


def _kv_table(rows: Sequence[tuple[str, Any]]) -> Table:
    """Render key/value pairs as a borderless Rich table."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(overflow="fold")
    for key, value in rows:
        table.add_row(key, str(value))
    return table


def _counts_text(counts: dict[str, Any] | None) -> str:
    """Render a ``{status: count}`` mapping as ``status=count`` pairs."""
    if not counts:
        return "(none)"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _human_size(num_bytes: int | float) -> str:
    """Render a byte count in the largest sensible binary unit."""
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def _sizes_text(sizes: dict[str, Any] | None) -> str:
    """Render the status ``{table: bytes}`` mapping as ``name=size`` pairs."""
    if not sizes:
        return "(none)"
    return ", ".join(f"{name}={_human_size(size)}" for name, size in sizes.items())


def _config_rows(data: dict[str, Any]) -> list[tuple[str, Any]]:
    """Rows for the configuration panel of ``status``."""
    rows: list[tuple[str, Any]] = [
        ("config file", data.get("config_file") or "(defaults)"),
        ("workspace", data.get("workspace")),
    ]
    roots = data.get("roots") or []
    if not roots:
        rows.append(("roots", "(none)"))
        return rows
    for root in roots:
        rows.append(
            (
                f"root: {root.get('name')}",
                f"{root.get('path')} (id {root.get('source_root_id')})",
            )
        )
    return rows


def _postgres_rows(data: dict[str, Any]) -> list[tuple[str, Any]]:
    """Rows for the PostgreSQL panel of ``status``."""
    if "error" in data:
        return [("state", data["error"])]
    jobs = data.get("jobs", {})
    return [
        ("files by status", _counts_text(data.get("files_by_status"))),
        ("documents by status", _counts_text(data.get("documents_by_status"))),
        ("versions", data.get("versions")),
        ("chunks", data.get("chunks")),
        (
            "jobs (queued/running/failed)",
            f"{jobs.get('queued', 0)} / {jobs.get('running', 0)} / {jobs.get('failed', 0)}",
        ),
        ("outbox unprocessed", data.get("outbox_unprocessed")),
        ("table sizes", _sizes_text(data.get("table_sizes"))),
    ]


def _unsupported_rows(data: dict[str, Any]) -> list[tuple[str, Any]]:
    """Rows for the unsupported-files panel of ``status``."""
    if "error" in data:
        return [("state", data["error"])]
    count = data.get("count") or 0
    if not count:
        return [("files", "(none)")]
    examples = ", ".join(data.get("examples") or [])
    return [
        ("files", f"{count}: {examples}"),
        ("note", "legacy formats are out of scope; convert to .docx/.pptx"),
    ]


def _opensearch_rows(data: dict[str, Any]) -> list[tuple[str, Any]]:
    """Rows for the OpenSearch panel of ``status``."""
    if "error" in data:
        return [("state", data["error"])]
    rows: list[tuple[str, Any]] = []
    for alias, info in (data.get("aliases") or {}).items():
        if isinstance(info, dict):
            rows.append((alias, f"{info.get('physical')} ({info.get('docs')} docs)"))
        else:
            rows.append((alias, str(info)))
    problems = data.get("problems") or []
    rows.append(("problems", "; ".join(problems) if problems else "(none)"))
    return rows


def _endpoints_table(snapshots: Sequence[dict[str, Any]]) -> Table:
    """Render endpoint guard snapshots as a Rich table."""
    table = Table()
    columns = (
        "process",
        "endpoint",
        "breaker",
        "calls",
        "p50 ms",
        "p95 ms",
        "errors",
        "last error",
    )
    for column in columns:
        table.add_column(column, overflow="fold")
    if not snapshots:
        table.add_row("", "(no endpoint used yet)", "", "", "", "", "", "")
        return table
    for snapshot in snapshots:
        if "error" in snapshot:
            table.add_row("", "snapshots", "unavailable", str(snapshot["error"]), "", "", "", "")
            continue
        p50 = snapshot.get("p50_ms")
        p95 = snapshot.get("p95_ms")
        age = snapshot.get("age_s")
        process = str(snapshot.get("process") or "")
        if age:
            process += f" ({age / 60:.0f} min ago)" if age >= 90 else f" ({age:.0f} s ago)"
        table.add_row(
            process,
            str(snapshot.get("name")),
            str(snapshot.get("state")),
            str(snapshot.get("calls")),
            "-" if p50 is None else f"{p50:.0f}",
            "-" if p95 is None else f"{p95:.0f}",
            f"{snapshot.get('error_rate', 0):.1%}",
            str(snapshot.get("last_error") or "-"),
        )
    return table


def _reconcile_rows() -> list[tuple[str, Any]]:
    """Rows for the last-drift-report panel of ``status``."""
    from mynotes.index.reconcile import load_last_report

    report = load_last_report()
    if report is None:
        return [("last reconcile", "(no report yet)")]
    return [
        ("last reconcile", str(report.get("saved_at") or "?")),
        ("expected / indexed", f"{report.get('expected', '?')} / {report.get('indexed', '?')}"),
        (
            "missing / extra / stale",
            f"{len(report.get('missing') or [])} / {len(report.get('extra') or [])} / "
            f"{len(report.get('stale') or [])}",
        ),
    ]


def _print_status_panels(data: dict[str, Any]) -> None:
    """Print ``collect_status()`` output as Rich panels."""
    console.print(Panel(_kv_table(_config_rows(data)), title="Configuration", expand=False))
    console.print(
        Panel(
            _kv_table(_postgres_rows(data.get("postgres") or {})), title="PostgreSQL", expand=False
        )
    )
    console.print(
        Panel(
            _kv_table(_opensearch_rows(data.get("opensearch") or {})),
            title="OpenSearch",
            expand=False,
        )
    )
    console.print(Panel(_kv_table(_reconcile_rows()), title="Index drift", expand=False))
    console.print(
        Panel(
            _kv_table(_unsupported_rows(data.get("unsupported") or {})),
            title="Unsupported files",
            expand=False,
        )
    )
    console.print(
        Panel(_endpoints_table(data.get("endpoints") or []), title="Model endpoints", expand=False)
    )


@app.command()
def init(
    no_migrate: bool = typer.Option(False, "--no-migrate", help="Skip database migrations."),
) -> None:
    """Prepare the database, workspace, source roots and OpenSearch indices."""
    results = run_init(apply_migrations=not no_migrate)
    console.print(_results_table(results, "mynotes init"))
    if any(result.level == "fail" for result in results):
        raise typer.Exit(1)


@app.command()
def status(
    as_json: bool = typer.Option(False, "--json", help="Print the status as JSON."),
) -> None:
    """Show the state of the database, search indices and endpoints."""
    data = collect_status()
    if as_json:
        typer.echo(json.dumps(data, default=str, indent=2))
        return
    _print_status_panels(data)


@app.command("check-endpoints")
def check_endpoints_cmd(
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip the LLM chat check."),
) -> None:
    """Verify the embedding, reranker and chat endpoints."""
    results = check_endpoints(include_llm=not no_llm)
    console.print(_results_table(results, "mynotes check-endpoints"))
    if any(result.level == "fail" for result in results):
        raise typer.Exit(1)


def _parse_since(value: str, option: str = "--since") -> datetime:
    """Parse a date option (``YYYY-MM`` or ``YYYY-MM-DD``) as UTC midnight.

    Args:
        value: The raw option value.
        option: The option name used in the error message.

    Returns:
        The UTC datetime at the start of the given day (or month).

    Raises:
        typer.BadParameter: If the value does not match the accepted formats.
    """
    parts = value.split("-")
    try:
        if len(parts) == 2:
            return datetime(int(parts[0]), int(parts[1]), 1, tzinfo=UTC)
        if len(parts) == 3:
            return datetime(int(parts[0]), int(parts[1]), int(parts[2]), tzinfo=UTC)
    except ValueError:
        pass
    raise typer.BadParameter(f"{option} expects YYYY-MM or YYYY-MM-DD") from None


def _hit_path(hit: SearchHit) -> str:
    """Render a hit's path with the PDF page or PPTX slide when present."""
    if hit.page is not None:
        return f"{hit.path}:p{hit.page}"
    if hit.slide is not None:
        return f"{hit.path}:s{hit.slide}"
    return hit.path


def _snippet(text: str, width: int = 160) -> str:
    """Collapse whitespace and truncate a hit's text to ``width`` characters."""
    return " ".join(text.split())[:width]


def _json_default(value: Any) -> str:
    """Serialize JSON-unencodable values; datetimes as ISO 8601."""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _hit_sources(hit: SearchHit) -> str:
    """Render a hit's fusion sources as ``name:rank`` pairs, best first."""
    if not hit.sources:
        return "-"
    ordered = sorted(hit.sources.items(), key=lambda item: item[1])
    return ",".join(f"{name}:{rank}" for name, rank in ordered)


def _hits_table(hits: Sequence[SearchHit], title: str, *, rerank_columns: bool) -> Table:
    """Render hits as a Rich table; ``rerank_columns`` adds Rerank and Sources."""
    table = Table(title=f"mynotes search: {title}")
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("Title", no_wrap=True)
    table.add_column("Path", style="cyan", no_wrap=True)
    table.add_column("Section", no_wrap=True)
    if rerank_columns:
        table.add_column("Rerank", justify="right", no_wrap=True)
        table.add_column("Sources", no_wrap=True)
    table.add_column("Snippet", overflow="fold")
    for hit in hits:
        row: list[str] = [str(hit.rank), hit.title or "-", _hit_path(hit), hit.section or "-"]
        if rerank_columns:
            rerank = "-" if hit.rerank_score is None else f"{hit.rerank_score:.3f}"
            row.append(rerank)
            row.append(_hit_sources(hit))
        row.append(_snippet(hit.text))
        table.add_row(*row)
    return table


@app.command()
def search(
    query: str = typer.Argument(
        ...,
        help=(
            "The search query text. Inline filters work: #tag, in:path, "
            "format:pdf, since:YYYY-MM[-DD], until:..., history:, as-of:YYYY-MM[-DD]."
        ),
    ),
    tag: list[str] | None = typer.Option(  # noqa: B008
        None, "--tag", help="Only documents with this tag (repeatable)."
    ),
    in_folder: list[str] | None = typer.Option(  # noqa: B008
        None, "--in", help="Only documents under this folder prefix (repeatable)."
    ),
    format_: list[str] | None = typer.Option(  # noqa: B008
        None, "--format", help="Only documents of this file format (repeatable)."
    ),
    since: str | None = typer.Option(
        None, "--since", help="Only documents dated YYYY-MM or YYYY-MM-DD or later."
    ),
    history: bool = typer.Option(
        False, "--history", help="Include superseded documents (history intent)."
    ),
    limit: int = typer.Option(10, "-n", "--limit", min=1, help="Maximum number of hits to show."),
    raw: bool = typer.Option(
        False,
        "--raw",
        help="Plain hybrid search: no query analysis, fusion or reranking.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the hits as JSON."),
) -> None:
    """Search the notes: query analysis, hybrid RRF search, fusion, rerank."""
    from mynotes.config import get_settings
    from mynotes.retrieval.types import SearchFilters

    filters = SearchFilters(
        tags=list(tag or []),
        folders=list(in_folder or []),
        formats=list(format_ or []),
        since=_parse_since(since) if since is not None else None,
        include_superseded=history,
        include_old_versions=history and get_settings().versions.index_old_versions,
    )
    if raw:
        from mynotes.retrieval.hybrid import hybrid_search

        hits = hybrid_search(query, filters)[:limit]
        if as_json:
            typer.echo(
                json.dumps(
                    [asdict(hit) for hit in hits],
                    ensure_ascii=False,
                    indent=2,
                    default=_json_default,
                )
            )
            return
        if not hits:
            console.print("[yellow]No results.[/]")
            return
        console.print(_hits_table(hits, query, rerank_columns=False))
        return

    from mynotes.retrieval.pipeline import retrieve

    result = retrieve(query, filters)
    hits = result.hits[:limit]
    if as_json:
        typer.echo(
            json.dumps(
                [asdict(hit) for hit in hits],
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )
        )
        return
    if not hits:
        console.print("[yellow]No results.[/]")
        return
    console.print(_hits_table(hits, query, rerank_columns=True))


def _citation_line(citation: Citation) -> str:
    """One ``Sources:`` line: ``[n] title — path (p. X)``, historical marked."""
    title = citation.title or citation.path
    line = f"[{citation.n}] {title} — {citation.path}"
    if citation.page is not None:
        line += f" (p. {citation.page})"
    elif citation.slide is not None:
        line += f" (slide {citation.slide})"
    if not citation.is_current:
        line += " (historical)"
    return line


def _print_sources(citations: Sequence[Citation]) -> None:
    """Print the ``Sources:`` block of an answer."""
    typer.echo("Sources:")
    if not citations:
        typer.echo("(none)")
        return
    for citation in citations:
        typer.echo(_citation_line(citation))


def _last_citations_path() -> Path:
    """The state file with the citations of the last ``mynotes ask`` (task 45)."""
    import platformdirs

    return Path(platformdirs.user_state_dir("mynotes")) / "last_citations.json"


def _save_last_citations(citations: Sequence[Citation]) -> None:
    """Persist the ask citations so ``mynotes open [n]`` can resolve them."""
    from mynotes.retrieval.answer import citation_to_dict

    try:
        path = _last_citations_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": datetime.now(UTC).isoformat(),
            "citations": [citation_to_dict(citation) for citation in citations],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        logger.warning("could not save last citations for 'mynotes open'", exc_info=True)


def _ask_json(query: str, filters: SearchFilters) -> None:
    """Run one non-streaming ask and print the result as JSON."""
    from mynotes.retrieval.answer import answer, citation_to_dict

    try:
        result = answer(query, filters)
    except Exception as exc:
        logger.exception("ask failed")
        typer.echo(json.dumps({"error": f"answer failed: {type(exc).__name__}"}))
        raise typer.Exit(1) from exc
    payload = {
        "question": result.question,
        "answer": result.answer,
        "citations": [citation_to_dict(citation) for citation in result.citations],
        "context_citations": [citation_to_dict(citation) for citation in result.context_citations],
        "relevant": result.relevant,
        "retrieval": {
            "hit_count": len(result.retrieval.hits),
            "timings_ms": dict(result.retrieval.timings_ms),
        },
        "timings_ms": dict(result.timings_ms),
    }
    _save_last_citations(result.citations)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


def _ask_complete(query: str, filters: SearchFilters) -> None:
    """Run one non-streaming ask and print the answer and its sources."""
    from mynotes.retrieval.answer import answer

    try:
        result = answer(query, filters)
    except Exception as exc:
        logger.exception("ask failed")
        console.print(f"[red]Answer failed: {type(exc).__name__}[/]")
        raise typer.Exit(1) from exc
    typer.echo(result.answer)
    _print_sources(result.citations)
    _save_last_citations(result.citations)


def _ask_stream(query: str, filters: SearchFilters) -> None:
    """Run one streaming ask, printing tokens and then the sources."""
    from mynotes.retrieval.answer import answer_stream

    ends_with_newline = True
    try:
        for event in answer_stream(query, filters):
            if event.type == "token":
                typer.echo(event.data, nl=False)
                ends_with_newline = False
            elif event.type == "error":
                if not ends_with_newline:
                    typer.echo("")
                console.print(f"[red]{event.data}[/]")
                ends_with_newline = True
            elif event.type == "done":
                if not ends_with_newline:
                    typer.echo("")
                _print_sources(event.data.citations)
                _save_last_citations(event.data.citations)
    except Exception as exc:
        logger.exception("ask failed")
        console.print(f"[red]Answer failed: {type(exc).__name__}[/]")
        raise typer.Exit(1) from exc


@app.command()
def ask(
    question: str = typer.Argument(
        ...,
        help=(
            "The question text. Inline filters work: #tag, in:path, format:pdf, "
            "since:YYYY-MM[-DD], history:, as-of:YYYY-MM[-DD]."
        ),
    ),
    tag: list[str] | None = typer.Option(  # noqa: B008
        None, "--tag", help="Only documents with this tag (repeatable)."
    ),
    in_folder: list[str] | None = typer.Option(  # noqa: B008
        None, "--in", help="Only documents under this folder prefix (repeatable)."
    ),
    format_: list[str] | None = typer.Option(  # noqa: B008
        None, "--format", help="Only documents of this file format (repeatable)."
    ),
    since: str | None = typer.Option(
        None, "--since", help="Only documents dated YYYY-MM or YYYY-MM-DD or later."
    ),
    history: bool = typer.Option(
        False, "--history", help="Include superseded documents (history intent)."
    ),
    as_of: str | None = typer.Option(
        None,
        "--as-of",
        help="Answer as of this date, YYYY-MM or YYYY-MM-DD (history intent).",
    ),
    no_stream: bool = typer.Option(
        False, "--no-stream", help="Wait for the complete answer instead of streaming tokens."
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Print the full answer result as JSON (non-streaming)."
    ),
) -> None:
    """Ask the notes a question; get a grounded, cited answer."""
    from mynotes.config import get_settings
    from mynotes.retrieval.types import SearchFilters

    query = question
    history_intent = history
    if as_of is not None:
        _parse_since(as_of, option="--as-of")
        query = f"as-of:{as_of} {query}"
        history_intent = True

    filters = SearchFilters(
        tags=list(tag or []),
        folders=list(in_folder or []),
        formats=list(format_ or []),
        since=_parse_since(since) if since is not None else None,
        include_superseded=history_intent,
        include_old_versions=history_intent and get_settings().versions.index_old_versions,
    )
    if as_json:
        _ask_json(query, filters)
    elif no_stream:
        _ask_complete(query, filters)
    else:
        _ask_stream(query, filters)


_CITATION_REF = re.compile(r"\[(\d+)\]")
"""A ``[n]`` citation reference as printed in the ask ``Sources:`` block."""


def _citation_document_id(number: int) -> int:
    """The document id of citation ``[number]`` from the last ``mynotes ask``."""
    path = _last_citations_path()
    if not path.exists():
        console.print("[yellow]No saved citations yet; run 'mynotes ask' first.[/]")
        raise typer.Exit(1)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        console.print("[red]The saved citations are unreadable; run 'mynotes ask' again.[/]")
        raise typer.Exit(1) from None
    citations = data.get("citations", []) if isinstance(data, dict) else data
    for citation in citations:
        if citation.get("n") == number:
            return int(citation["document_id"])
    console.print(f"[yellow]The last ask output has no citation [{number}].[/]")
    raise typer.Exit(1)


def _document_id_for_open(ref: str) -> int:
    """Resolve an ``open`` reference (citation, id, path or title) to a document id."""
    from mynotes.core import service

    match = _CITATION_REF.fullmatch(ref.strip())
    if match is not None:
        return _citation_document_id(int(match.group(1)))
    return service.get_document(ref).document_id


def _print_open_candidates(candidates: Sequence[dict[str, Any]]) -> None:
    """List ambiguous document candidates for ``mynotes open``."""
    for candidate in candidates:
        title = candidate.get("title") or candidate.get("name") or "?"
        document_id = candidate.get("document_id") or candidate.get("id")
        path = candidate.get("path", "")
        console.print(f"  [{document_id}] {title} ({path})")


@app.command("open")
def open_(
    ref: str = typer.Argument(
        ...,
        help=(
            "A document id, path or title; or a citation like [1] from the last "
            "'mynotes ask' output."
        ),
    ),
) -> None:
    """Open a document's original file in the OS default app (local mode)."""
    from mynotes.core import service

    try:
        document_id = _document_id_for_open(ref)
        service.open_original(document_id)
    except service.Ambiguous as exc:
        console.print(f"[yellow]{exc}[/]")
        _print_open_candidates(exc.candidates)
        raise typer.Exit(1) from exc
    except (service.NotFound, service.Forbidden, RuntimeError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc


@app.command()
def worker(
    concurrency: int = typer.Option(4, "--concurrency", min=1, help="Number of job threads."),
    types: str | None = typer.Option(
        None,
        "--types",
        help="Comma-separated job types to claim (default: every type with a handler).",
    ),
    no_schedule: bool = typer.Option(
        False,
        "--no-schedule",
        help="Disable the scheduler thread (scheduled scans and the nightly reconcile).",
    ),
) -> None:
    """Consume background jobs from the Postgres queue until interrupted."""
    from mynotes.db.session import session_scope
    from mynotes.jobs.queue import counts
    from mynotes.jobs.worker import Worker, unregister_background_thread

    if no_schedule:
        unregister_background_thread("scheduler")
    from mynotes.scan.watch import register_watchers

    try:
        register_watchers()
    except Exception:
        logger.warning("could not register file watchers", exc_info=True)
    type_list = [item.strip() for item in types.split(",") if item.strip()] if types else None
    from mynotes.jobs.endpoint_stats import set_process_role

    set_process_role("worker")
    job_worker = Worker(concurrency=concurrency, types=type_list)
    job_worker.start()

    stop_requested = threading.Event()

    def _request_stop(signum: int, frame: Any) -> None:
        logger.info("signal %s received, stopping worker", signum)
        stop_requested.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _request_stop)
        except (OSError, ValueError):
            logger.debug("cannot install handler for %s", sig)

    logger.info(
        "worker running (concurrency=%d, types=%s); press Ctrl+C to stop",
        concurrency,
        ",".join(type_list) if type_list else "(all with handler)",
    )
    try:
        while not stop_requested.wait(60.0):
            try:
                with session_scope() as session:
                    job_counts = counts(session)
            except Exception:
                logger.warning("could not read job counts", exc_info=True)
                continue
            logger.info("jobs: %s", _counts_text(job_counts))
    except KeyboardInterrupt:
        pass
    finally:
        job_worker.stop()
        logger.info("worker stopped")


@app.command()
def scan(
    root: str | None = typer.Option(
        None, "--root", help="Root name to scan (default: all enabled roots)."
    ),
    paths: list[str] | None = typer.Option(  # noqa: B008
        None, "--paths", help="Relative paths to scan; implies a targeted scan."
    ),
    full: bool = typer.Option(False, "--full", help="Re-hash every file (full verification scan)."),
) -> None:
    """Scan source roots for changes and persist documents, versions and jobs."""
    from sqlalchemy import select

    from mynotes.config import get_settings
    from mynotes.db.models import SourceRoot, Workspace
    from mynotes.db.session import session_scope
    from mynotes.scan.scanner import scan_root

    kind = "targeted" if paths else ("full" if full else "incremental")
    settings = get_settings()
    stmt = (
        select(SourceRoot)
        .join(Workspace, Workspace.id == SourceRoot.workspace_id)
        .where(Workspace.name == settings.workspace)
    )
    stmt = stmt.where(SourceRoot.enabled.is_(True))
    if root is not None:
        stmt = stmt.where(SourceRoot.name == root)
    with session_scope() as session:
        roots = [
            (source_root.id, source_root.name) for source_root in session.execute(stmt).scalars()
        ]
    if not roots:
        console.print(
            "[yellow]No matching source root; check the roots in mynotes.yaml "
            "and run 'mynotes init'.[/]"
        )
        raise typer.Exit(1)

    table = Table(title=f"mynotes scan ({kind})")
    table.add_column("Root", style="cyan", no_wrap=True)
    table.add_column("Counts", overflow="fold")
    table.add_column("Errors", overflow="fold")
    failed = False
    for source_root_id, source_root_name in roots:
        try:
            summary = scan_root(
                source_root_id,
                kind=kind,  # type: ignore[arg-type]
                trigger="manual",
                rel_paths=list(paths) if paths else None,
            )
            counts_text = _counts_text(summary.counts)
            errors_text = "; ".join(summary.errors)
            if summary.errors:
                failed = True
        except Exception as exc:
            logger.exception("scan of root '%s' failed", source_root_name)
            counts_text = "failed"
            errors_text = f"{type(exc).__name__}: {exc}"
            failed = True
        table.add_row(source_root_name, counts_text, errors_text or "-")
    console.print(table)
    if failed:
        raise typer.Exit(1)


watch_app = typer.Typer(help="File watcher commands.")
app.add_typer(watch_app, name="watch")


@watch_app.command("status")
def watch_status() -> None:
    """Show the per-root watcher state last reported by the worker."""
    from mynotes.scan.watch import watcher_status_rows

    rows = watcher_status_rows()
    if not rows:
        console.print(
            "[yellow]No watcher state found; the worker is not running or watching is disabled.[/]"
        )
        return
    table = Table(title="mynotes watch status")
    table.add_column("Root", style="cyan", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("Backend", no_wrap=True)
    table.add_column("Since", overflow="fold")
    for row in rows:
        state = str(row["state"])
        if row["stale"]:
            state += " [stale (worker not running?)]"
        table.add_row(str(row["root"]), state, str(row["backend"]), str(row["since"]))
    console.print(table)


service_app = typer.Typer(help="Worker autostart service (systemd, launchd, Task Scheduler).")
app.add_typer(service_app, name="service")


_COMPONENT_OPTION = typer.Option(
    "worker",
    "--component",
    "-c",
    help="worker (scan, index, graph), web (the web interface, localhost:8765) or all.",
)


def _service_parts(component: str = "worker") -> tuple[Any, Any]:
    """Build the spec for one component and the platform backend, or exit with an error."""
    from mynotes.platform.service import ServiceError, backend_for_platform, default_spec

    try:
        return default_spec(component), backend_for_platform()
    except ServiceError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc


def _components(selected: str) -> list[str]:
    """Expand the ``--component`` value (``all`` = worker then web)."""
    from mynotes.platform.service import COMPONENTS

    if selected == "all":
        return list(COMPONENTS)
    if selected not in COMPONENTS:
        console.print(
            f"[red]Unknown component {selected!r}; use {', '.join(COMPONENTS)} or all.[/]"
        )
        raise typer.Exit(2)
    return [selected]


@service_app.command("install")
def service_install(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the rendered files/commands without touching the system.",
    ),
    component: str = _COMPONENT_OPTION,
) -> None:
    """Install the autostart definition of the worker and/or the web interface."""
    for name in _components(component):
        spec, backend = _service_parts(name)
        rendered = backend.render(spec)
        if dry_run:
            for target, content in rendered.items():
                console.print(Panel(Text(content), title=f"dry-run: {target}", expand=False))
            console.print(f"[green]Dry run only; '{spec.name}' was not installed.[/]")
            continue
        try:
            backend.install(spec)
        except Exception as exc:
            logger.exception("service install failed")
            console.print(f"[red]Service install failed: {type(exc).__name__}: {exc}[/]")
            raise typer.Exit(1) from exc
        console.print(f"[green]Service '{spec.name}' installed.[/]")
        for hint in backend.hints(spec):
            console.print(hint, markup=False)


@service_app.command("uninstall")
def service_uninstall(component: str = _COMPONENT_OPTION) -> None:
    """Disable the service(s) and remove the definition(s)."""
    for name in _components(component):
        spec, backend = _service_parts(name)
        try:
            backend.uninstall(spec)
        except Exception as exc:
            logger.exception("service uninstall failed")
            console.print(f"[red]Service uninstall failed: {type(exc).__name__}: {exc}[/]")
            raise typer.Exit(1) from exc
        console.print(f"[green]Service '{spec.name}' uninstalled.[/]")


@service_app.command("status")
def service_status(component: str = _COMPONENT_OPTION) -> None:
    """Show whether the service(s) are installed and running."""
    for name in _components(component):
        spec, backend = _service_parts(name)
        try:
            console.print(f"{spec.name}: {backend.status(spec)}")
        except Exception as exc:
            logger.exception("service status failed")
            console.print(f"[red]Service status failed: {type(exc).__name__}: {exc}[/]")
            raise typer.Exit(1) from exc


def _tail_follow(path: Path) -> None:
    """Print the last lines of ``path`` and follow its growth until Ctrl+C."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle.readlines()[-50:]:
            typer.echo(line.rstrip("\n"))
        handle.seek(0, os.SEEK_END)
        try:
            while True:
                line = handle.readline()
                if line:
                    typer.echo(line.rstrip("\n"))
                else:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            return


@service_app.command("logs")
def service_logs(component: str = _COMPONENT_OPTION) -> None:
    """Follow one service's logs (prints the platform hint on Linux)."""
    if component == "all":
        console.print("[red]--component all is not supported for logs; pick worker or web.[/]")
        raise typer.Exit(2)
    spec, backend = _service_parts(component)
    log_file = spec.log_dir / spec.log_name
    if sys.platform in ("darwin", "win32") and log_file.is_file():
        _tail_follow(log_file)
        return
    console.print(backend.logs_hint(spec))


index_app = typer.Typer(help="Index maintenance commands.")
app.add_typer(index_app, name="index")


@index_app.command("sync")
def index_sync() -> None:
    """Drain the index outbox into OpenSearch until a pass processes nothing."""
    from mynotes.index.sync import SyncStats, sync_once

    totals = SyncStats()
    while True:
        stats = sync_once()
        totals.processed += stats.processed
        totals.upserts += stats.upserts
        totals.deletes += stats.deletes
        totals.failures += stats.failures
        if stats.processed == 0 or stats.failures:
            break
    table = Table(title="mynotes index sync")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Count", justify="right")
    table.add_row("outbox rows processed", str(totals.processed))
    table.add_row("upserts", str(totals.upserts))
    table.add_row("deletes", str(totals.deletes))
    table.add_row("failures", str(totals.failures))
    console.print(table)
    if totals.failures:
        raise typer.Exit(1)


def _examples(ids: Sequence[str], limit: int = 10) -> str:
    """Render up to ``limit`` example ids for a drift list."""
    if not ids:
        return "-"
    shown = ", ".join(ids[:limit])
    more = len(ids) - limit
    return f"{shown} (+{more} more)" if more > 0 else shown


@index_app.command("reconcile")
def index_reconcile(
    fix: bool = typer.Option(
        True, "--fix/--no-fix", help="Queue outbox repairs for the drift found (default)."
    ),
) -> None:
    """Compare the chunks index with Postgres and report (or fix) the drift."""
    from mynotes.index.reconcile import reconcile

    report = reconcile(fix=fix)
    table = Table(title="mynotes index reconcile")
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Count", justify="right")
    table.add_column("Examples (up to 10)", overflow="fold")
    table.add_row("expected (Postgres)", str(report.expected), "-")
    table.add_row("indexed (OpenSearch)", str(report.indexed), "-")
    table.add_row("missing", str(len(report.missing)), _examples(report.missing))
    table.add_row("extra", str(len(report.extra)), _examples(report.extra))
    table.add_row("stale", str(len(report.stale)), _examples(report.stale))
    console.print(table)


@index_app.command("rebuild")
def index_rebuild(
    reembed: bool = typer.Option(
        False, "--reembed", help="Drop the embedding cache of the current model and re-embed."
    ),
    entities: bool = typer.Option(
        False, "--entities", help="Rebuild the entities index instead of the chunks index."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Rebuild an index into a new physical index and swap the alias."""
    if not entities:
        from mynotes.index.rebuild import rebuild_chunks_index

        if not yes:
            typer.confirm(
                "Rebuild the chunks index from Postgres"
                + (" with full re-embedding" if reembed else "")
                + "?",
                abort=True,
            )
        result = rebuild_chunks_index(reembed=reembed)
    else:
        from mynotes.index.rebuild import rebuild_entities_index

        if not yes:
            typer.confirm("Rebuild the entities index from Postgres?", abort=True)
        result = rebuild_entities_index()
    table = Table(title="mynotes index rebuild")
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    for key, value in result.items():
        table.add_row(key, str(value))
    console.print(table)


versions_app = typer.Typer(help="Version (supersedes) review and history commands.")
app.add_typer(versions_app, name="versions")


def _doc_label(ref: Any) -> str:
    """Render one relation endpoint as ``title — path (date)``."""
    parts: list[str] = []
    if ref.title:
        parts.append(ref.title)
    parts.append(ref.path or "(no file)")
    if ref.doc_date is not None:
        parts.append(f"{ref.doc_date:%Y-%m-%d}")
    return " — ".join(parts) + f" [{ref.status}]"


def _evidence_text(evidence: dict[str, Any], width: int = 120) -> str:
    """Render a relation's evidence dict as one short JSON line."""
    if not evidence:
        return "-"
    text = json.dumps(evidence, ensure_ascii=False, sort_keys=True, default=str)
    return text if len(text) <= width else text[: width - 1] + "…"


def _relations_table(views: Sequence[Any]) -> Table:
    """Render supersedes relations as a Rich table."""
    table = Table(title="mynotes versions")
    table.add_column("ID", justify="right", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Conf", justify="right", no_wrap=True)
    table.add_column("Method", no_wrap=True)
    table.add_column("Newer", overflow="fold")
    table.add_column("Older", overflow="fold")
    table.add_column("Evidence", overflow="fold")
    for view in views:
        table.add_row(
            str(view.id),
            view.status,
            f"{view.confidence:.2f}",
            view.method,
            _doc_label(view.newer),
            _doc_label(view.older),
            _evidence_text(view.evidence),
        )
    return table


@versions_app.command("list")
def versions_list(
    status: str | None = typer.Option(
        None,
        "--status",
        help="Only relations with this status: proposed, auto, confirmed or rejected.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the relations as JSON."),
) -> None:
    """List supersedes relations (default: all statuses)."""
    from mynotes.db.session import session_scope
    from mynotes.versioning.review import list_relations

    statuses = [status.strip()] if status and status.strip() else None
    with session_scope() as session:
        views = list_relations(session, status=statuses)
    if as_json:
        typer.echo(
            json.dumps(
                [asdict(view) for view in views],
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )
        )
        return
    if not views:
        console.print("[yellow]No version relations found.[/]")
        return
    console.print(_relations_table(views))


def _print_relation(view: Any, newer_preview: str, older_preview: str) -> None:
    """Show one relation with both document previews, for the review loop."""
    header = (
        f"Relation {view.id} · {view.status} · confidence {view.confidence:.2f}"
        f" · method {view.method}"
    )
    console.rule(header)
    console.print(f"[bold]Newer:[/] {_doc_label(view.newer)}")
    if newer_preview:
        console.print(newer_preview, markup=False)
    console.print(f"[bold]Older:[/] {_doc_label(view.older)}")
    if older_preview:
        console.print(older_preview, markup=False)
    console.print(f"[bold]Evidence:[/] {_evidence_text(view.evidence, width=400)}")


@versions_app.command("review")
def versions_review() -> None:
    """Interactively review the proposed supersedes relations."""
    from mynotes.db.session import session_scope
    from mynotes.versioning.review import (
        confirm,
        document_preview,
        list_relations,
        reject,
        swap,
    )

    with session_scope() as session:
        views = list_relations(session, status=["proposed"])
        previews = {
            view.id: (
                document_preview(session, view.newer.document_id),
                document_preview(session, view.older.document_id),
            )
            for view in views
        }
    if not views:
        console.print("[green]No proposed version relations to review.[/]")
        return
    console.print(f"{len(views)} proposed relation(s).")
    for view in views:
        newer_preview, older_preview = previews[view.id]
        _print_relation(view, newer_preview, older_preview)
        choice = ""
        while choice not in ("c", "r", "s", "k", "q"):
            choice = (
                typer.prompt(
                    "[c]onfirm / [r]eject / [s]wap / s[k]ip / [q]uit",
                    default="k",
                    show_default=False,
                )
                .strip()
                .lower()
            )
        if choice == "q":
            return
        if choice == "k":
            continue
        labels = {"c": "confirmed", "r": "rejected", "s": "swapped"}
        actions = {"c": confirm, "r": reject, "s": swap}
        try:
            with session_scope() as session:
                actions[choice](session, view.id)
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1) from exc
        console.print(f"[green]Relation {view.id} {labels[choice]}.[/]")


def _versions_action(action_name: str, relation_id: int) -> None:
    """Run one review action (confirm/reject/swap) for a relation id."""
    from mynotes.db.session import session_scope
    from mynotes.versioning import review

    action = getattr(review, action_name)
    try:
        with session_scope() as session:
            action(session, relation_id)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Relation {relation_id}: {action_name}ed.[/]")


@versions_app.command("confirm")
def versions_confirm(
    relation_id: int = typer.Argument(..., help="The supersedes relation id."),
) -> None:
    """Confirm a relation: the older document becomes superseded."""
    _versions_action("confirm", relation_id)


@versions_app.command("reject")
def versions_reject(
    relation_id: int = typer.Argument(..., help="The supersedes relation id."),
) -> None:
    """Reject a relation: the older document returns to current."""
    _versions_action("reject", relation_id)


@versions_app.command("swap")
def versions_swap(
    relation_id: int = typer.Argument(..., help="The supersedes relation id."),
) -> None:
    """Reverse a relation's direction (the order is wrong) and confirm it."""
    _versions_action("swap", relation_id)


@versions_app.command("add")
def versions_add(
    newer_path: str = typer.Argument(..., help="Path of the successor document."),
    older_path: str = typer.Argument(..., help="Path of the superseded document."),
) -> None:
    """Record a manual, confirmed supersedes relation between two documents."""
    from mynotes.db.session import session_scope
    from mynotes.versioning.review import add_manual

    try:
        with session_scope() as session:
            relation_id = add_manual(session, newer_path, older_path)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Relation {relation_id} added (manual, confirmed).[/]")


@versions_app.command("history")
def versions_history(
    path: str = typer.Argument(..., help="Path of the document (as shown by search)."),
) -> None:
    """Print a document's retained version timeline as a table."""
    from mynotes.db.session import session_scope
    from mynotes.versioning.history import document_history
    from mynotes.versioning.review import find_document_by_path

    try:
        with session_scope() as session:
            document_id, title = find_document_by_path(session, path)
            infos = document_history(session, document_id)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    table = Table(title=f"mynotes versions history: {title or path}")
    table.add_column("Version", justify="right", no_wrap=True)
    table.add_column("Date", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Chunks", justify="right", no_wrap=True)
    table.add_column("Changed sections", overflow="fold")
    for info in infos:
        date = info.doc_date if info.doc_date is not None else info.created_at
        row = [
            f"v{info.version_no}" + (" (latest)" if info.is_latest else ""),
            f"{date:%Y-%m-%d}",
            info.status,
            str(info.chunk_count),
            ", ".join(info.changed_sections) or "-",
        ]
        table.add_row(*row)
    console.print(table)


tags_app = typer.Typer(help="Tag vocabulary and document tags.")
app.add_typer(tags_app, name="tags")


def _workspace_id(session: Any) -> int:
    """The default workspace's id, or exit with an error."""
    from mynotes.config import get_settings
    from mynotes.tags.store import workspace_id_by_name

    workspace_id = workspace_id_by_name(session, get_settings().workspace)
    if workspace_id is None:
        console.print(
            f"[red]Workspace {get_settings().workspace!r} not found; run 'mynotes init' first.[/]"
        )
        raise typer.Exit(1)
    return workspace_id


@tags_app.command("list")
def tags_list(
    prefix: str | None = typer.Option(
        None, "--prefix", help="Only tags whose name starts with this prefix."
    ),
    source: str | None = typer.Option(
        None,
        "--source",
        help="Only tags applied with this source: folder, frontmatter, hashtag, llm or manual.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the tags as JSON."),
) -> None:
    """List the tag vocabulary with document counts and sources."""
    from mynotes.db.session import session_scope
    from mynotes.tags.store import tag_summaries

    with session_scope() as session:
        summaries = tag_summaries(session, _workspace_id(session), prefix=prefix, source=source)
    if as_json:
        typer.echo(
            json.dumps(
                [asdict(summary) for summary in summaries],
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not summaries:
        console.print("[yellow]No tags found.[/]")
        return
    table = Table(title="mynotes tags")
    table.add_column("Tag", style="cyan", no_wrap=True)
    table.add_column("Kind", no_wrap=True)
    table.add_column("Docs", justify="right", no_wrap=True)
    table.add_column("Sources", no_wrap=True)
    for summary in summaries:
        table.add_row(
            summary.name,
            summary.kind,
            str(summary.document_count),
            ", ".join(summary.sources) or "-",
        )
    console.print(table)


@tags_app.command("merge")
def tags_merge(
    from_name: str = typer.Argument(..., help="The tag to merge away."),
    into_name: str = typer.Argument(..., help="The surviving tag."),
) -> None:
    """Merge one tag into another (moves documents, creates an alias)."""
    from mynotes.db.session import session_scope
    from mynotes.jobs.queue import enqueue
    from mynotes.jobs.registry import Priority
    from mynotes.tags.store import document_ids_with_tag, merge_tags

    try:
        with session_scope() as session:
            workspace_id = _workspace_id(session)
            affected = document_ids_with_tag(session, workspace_id, from_name)
            moved = merge_tags(session, workspace_id, from_name, into_name)
            for document_id in affected:
                enqueue(
                    session,
                    "reindex_metadata",
                    {"document_id": document_id},
                    dedupe_key=f"reindex_metadata:{document_id}",
                    priority=Priority.EMBED,
                )
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]Merged '{from_name}' into '{into_name}': {moved} document tag row(s) moved, "
        f"{len(affected)} document(s) queued for reindexing.[/]"
    )


@tags_app.command("show")
def tags_show(
    path: str = typer.Argument(..., help="Path of the document (as shown by search)."),
) -> None:
    """Show the tags of one document, with source and status."""
    from mynotes.db.session import session_scope
    from mynotes.tags.store import document_tag_details
    from mynotes.versioning.review import find_document_by_path

    try:
        with session_scope() as session:
            document_id, title = find_document_by_path(session, path)
            details = document_tag_details(session, document_id)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    table = Table(title=f"mynotes tags: {title or path}")
    table.add_column("Tag", style="cyan", no_wrap=True)
    table.add_column("Kind", no_wrap=True)
    table.add_column("Source", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Confidence", justify="right", no_wrap=True)
    for detail in details:
        table.add_row(
            detail.name,
            detail.kind,
            detail.source,
            detail.status,
            f"{detail.confidence:.2f}",
        )
    console.print(table)


def _enqueue_reindex(document_id: int, session: Any) -> None:
    """Queue a metadata reindex for a document after a tag decision."""
    from mynotes.jobs.queue import enqueue
    from mynotes.jobs.registry import Priority

    enqueue(
        session,
        "reindex_metadata",
        {"document_id": document_id},
        dedupe_key=f"reindex_metadata:{document_id}",
        priority=Priority.EMBED,
    )


@tags_app.command("review")
def tags_review() -> None:
    """Interactively review the proposed LLM tags, oldest document first."""
    from mynotes.db.session import session_scope
    from mynotes.tags.autotag import decide_proposal, merge_proposal, pending_proposals

    with session_scope() as session:
        pending = pending_proposals(session)
    total = sum(len(doc.proposals) for doc in pending)
    if not pending:
        console.print("[green]No proposed LLM tags to review.[/]")
        return
    console.print(f"{total} proposed tag(s) on {len(pending)} document(s).")
    for doc in pending:
        console.rule(f"{doc.title or '(untitled)'} — {doc.rel_path}")
        for proposal in doc.proposals:
            console.print(
                f"  [cyan]{proposal.name}[/] ({proposal.confidence:.2f}): {proposal.reason or '-'}",
                markup=False,
            )
            while True:
                choice = (
                    typer.prompt(
                        "[a]ccept / [r]eject / [m]erge into… / [s]kip / [q]uit",
                        default="s",
                        show_default=False,
                    )
                    .strip()
                    .lower()
                )
                if choice == "s":
                    break
                if choice == "q":
                    return
                target: str | None = None
                try:
                    if choice == "m":
                        target = typer.prompt("Merge into tag").strip()
                    with session_scope() as session:
                        if choice == "m":
                            assert target is not None
                            merge_proposal(
                                session,
                                _workspace_id(session),
                                doc.document_id,
                                proposal.name,
                                target,
                            )
                        elif choice in ("a", "r"):
                            decide_proposal(
                                session,
                                doc.document_id,
                                proposal.tag_id,
                                "accepted" if choice == "a" else "rejected",
                            )
                        else:
                            console.print("[red]Choose a, r, m, s or q.[/]")
                            continue
                        _enqueue_reindex(doc.document_id, session)
                except ValueError as exc:
                    console.print(f"[red]{exc}[/]")
                    continue
                label = {
                    "a": "accepted",
                    "r": "rejected",
                    "m": f"merged into '{target}'",
                }[choice]
                console.print(f"[green]'{proposal.name}' {label}.[/]")
                break


def _decide_command(path: str, tag: str, decision: str) -> None:
    """Apply one non-interactive accept/reject decision for a document tag."""
    from mynotes.db.session import session_scope
    from mynotes.tags.autotag import decide_proposal, tag_id_by_name
    from mynotes.versioning.review import find_document_by_path

    verb = "Accepted" if decision == "accepted" else "Rejected"
    try:
        with session_scope() as session:
            document_id, title = find_document_by_path(session, path)
            tag_id = tag_id_by_name(session, _workspace_id(session), tag)
            if tag_id is None:
                raise ValueError(f"tag {tag!r} is not in the vocabulary")
            if not decide_proposal(session, document_id, tag_id, decision):
                raise ValueError(
                    f"no proposed LLM tag {tag!r} for {path!r} (see 'mynotes tags show')"
                )
            _enqueue_reindex(document_id, session)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    console.print(f"[green]{verb} '{tag}' for {title or path}; reindex queued.[/]")


@tags_app.command("accept")
def tags_accept(
    path: str = typer.Argument(..., help="Path of the document (as shown by search)."),
    tag: str = typer.Argument(..., help="The proposed tag to accept."),
) -> None:
    """Accept one proposed LLM tag of a document (non-interactive)."""
    _decide_command(path, tag, "accepted")


@tags_app.command("reject")
def tags_reject(
    path: str = typer.Argument(..., help="Path of the document (as shown by search)."),
    tag: str = typer.Argument(..., help="The proposed tag to reject."),
) -> None:
    """Reject one proposed LLM tag of a document (it will not be proposed again)."""
    _decide_command(path, tag, "rejected")


graph_app = typer.Typer(help="Structural and knowledge graph commands.")
app.add_typer(graph_app, name="graph")


def _print_neighbours(data: Any) -> None:
    """Render the :class:`StructuralNeighbours` sections of one document."""
    console.rule(f"mynotes graph · {_doc_label(data.document)}")
    sections: list[tuple[str, Sequence[Any]]] = [
        ("Links out", data.links_out),
        ("Links in", data.links_in),
        ("Supersedes", data.supersedes),
        ("Superseded by", data.superseded_by),
        ("Duplicates", data.duplicates),
        ("Same folder", data.same_folder),
        ("Same project", data.same_project),
    ]
    for title, nodes in sections:
        console.print(f"[bold]{title}[/] ({len(nodes)})")
        for node in nodes:
            console.print(f"  - {_doc_label(node)}")
    if data.tags:
        console.print(f"[bold]Tags[/]: {', '.join(data.tags)}")


@graph_app.command("doc")
def graph_doc(
    ref: str = typer.Argument(..., help="Document path or title."),
    as_json: bool = typer.Option(False, "--json", help="Print the neighbours as JSON."),
) -> None:
    """Show the structural neighbours of one document (task 39)."""
    from mynotes.db.session import session_scope
    from mynotes.graph.structural import find_document, neighbours

    with session_scope() as session:
        candidates = find_document(session, ref)
        data = neighbours(session, candidates[0].document_id) if len(candidates) == 1 else None
    if data is None:
        if not candidates:
            console.print(f"[red]No document matches {ref!r}.[/]")
        else:
            console.print(f"[red]{len(candidates)} documents match {ref!r}:[/]")
            for node in candidates:
                console.print(f"  - {_doc_label(node)}")
        raise typer.Exit(1)
    if as_json:
        typer.echo(json.dumps(asdict(data), ensure_ascii=False, indent=2, default=_json_default))
        return
    _print_neighbours(data)


@graph_app.command("backfill")
def graph_backfill(
    limit: int | None = typer.Option(
        None, "--limit", min=1, help="Maximum number of versions to enqueue."
    ),
) -> None:
    """Enqueue kg_extract for latest embedded versions with uncached groups (task 40).

    Versions whose section groups are all in ``kg_extraction_cache`` are
    skipped, so re-running backfill only tops up what is missing. Run
    'mynotes worker --concurrency 10' to drain the queue: the LLM background
    pool (8 slots) caps the actual model concurrency.
    """
    from mynotes.db.session import session_scope
    from mynotes.graph.extract import versions_missing_extraction
    from mynotes.jobs.queue import enqueue
    from mynotes.jobs.registry import Priority

    enqueued = 0
    with session_scope() as session:
        for version_id in versions_missing_extraction(session):
            if limit is not None and enqueued >= limit:
                break
            if (
                enqueue(
                    session,
                    "kg_extract",
                    {"version_id": version_id},
                    dedupe_key=f"kg_extract:{version_id}",
                    priority=Priority.KG_EXTRACT,
                )
                is not None
            ):
                enqueued += 1
    console.print(
        f"Enqueued {enqueued} kg_extract job(s); "
        "run 'mynotes worker --concurrency 10' to drain them."
    )


def _format_duration(seconds: float) -> str:
    """Render a duration as a compact ``~1h 23m`` style estimate."""
    total = round(seconds)
    if total < 60:
        return f"~{total}s"
    days, rest = divmod(total, 86_400)
    hours, rest = divmod(rest, 3_600)
    minutes = rest // 60
    if days:
        return f"~{days}d {hours}h"
    if hours:
        return f"~{hours}h {minutes}m"
    return f"~{minutes}m"


@graph_app.command("extract-status")
def graph_extract_status(
    as_json: bool = typer.Option(False, "--json", help="Print the status as JSON."),
) -> None:
    """Show KG extraction progress: groups, cache state, queue depth and ETA (task 40).

    The ETA multiplies the pending-group estimate by the observed LLM p50
    latency and divides by the background concurrency; it is only available
    once this process has observed LLM calls.
    """
    from sqlalchemy import func, select

    from mynotes.config import get_settings
    from mynotes.db.models import Job
    from mynotes.db.session import session_scope
    from mynotes.graph.extract import groups_overview
    from mynotes.llm.limits import all_snapshots

    with session_scope() as session:
        data = groups_overview(session)
        queued = int(
            session.execute(
                select(func.count())
                .select_from(Job)
                .where(Job.type == "kg_extract", Job.status == "queued")
            ).scalar_one()
        )
    p50_ms = next((snapshot.p50_ms for snapshot in all_snapshots() if snapshot.name == "llm"), None)
    concurrency = get_settings().models.llm.limits.background_concurrency
    eta_s = data["pending"] * (p50_ms / 1000.0) / concurrency if p50_ms is not None else None
    if as_json:
        typer.echo(
            json.dumps(
                {
                    **data,
                    "queued_kg_extract_jobs": queued,
                    "llm_p50_ms": p50_ms,
                    "background_concurrency": concurrency,
                    "eta_seconds": eta_s,
                },
                indent=2,
            )
        )
        return
    console.print(
        Panel(
            _kv_table(
                [
                    ("versions (latest, embedded)", data["versions"]),
                    ("groups total", data["total"]),
                    ("groups cached", data["cached"]),
                    ("groups pending (estimate)", data["pending"]),
                    ("queued kg_extract jobs", queued),
                    (
                        "llm p50 latency",
                        "(no llm calls observed)" if p50_ms is None else f"{p50_ms / 1000.0:.1f} s",
                    ),
                    ("background concurrency", concurrency),
                    ("eta", _format_duration(eta_s) if eta_s is not None else "(unknown)"),
                ]
            ),
            title="KG extraction",
            expand=False,
        )
    )


def _entity_ref_id(session: Any, ref: str) -> int | None:
    """Resolve one entity reference to a single id, printing candidates otherwise."""
    from mynotes.graph.store import find_entities

    matches = find_entities(session, ref)
    if len(matches) == 1:
        return int(matches[0].id)
    if not matches:
        console.print(f"[red]No entity matches {ref!r}.[/]")
    else:
        console.print(f"[red]{len(matches)} entities match {ref!r}:[/]")
        for entity in matches:
            status = f" [dim]({entity.status})[/]" if entity.status != "active" else ""
            console.print(f"  - {entity.canonical_name} ({entity.type})#{entity.id}{status}")
    return None


@graph_app.command("entities")
def graph_entities(
    type: str | None = typer.Option(None, "--type", help="Only entities of this type."),
    limit: int = typer.Option(50, "--limit", min=1, help="Maximum number of entities."),
) -> None:
    """List the active entities with their current mention counts (task 41)."""
    from mynotes.db.session import session_scope
    from mynotes.graph.store import list_entities

    with session_scope() as session:
        rows = [
            (entity.canonical_name, entity.type, entity.mention_count_current, entity.id)
            for entity in list_entities(session, _workspace_id(session), type=type, limit=limit)
        ]
    if not rows:
        console.print("[yellow]No entities found.[/]")
        return
    table = Table(title="mynotes graph entities")
    table.add_column("Name", overflow="fold")
    table.add_column("Type", style="cyan", no_wrap=True)
    table.add_column("Mentions", justify="right")
    table.add_column("Id", justify="right")
    for name, entity_type, mentions, entity_id in rows:
        table.add_row(name, entity_type, str(mentions), str(entity_id))
    console.print(table)


@graph_app.command("entity")
def graph_entity(
    ref: str = typer.Argument(..., help="Entity name (exact, normalized or fuzzy)."),
    type: str | None = typer.Option(None, "--type", help="Only entities of this type."),
    as_json: bool = typer.Option(False, "--json", help="Print the entity card as JSON."),
) -> None:
    """Show one entity card, or the candidates when the reference is ambiguous (task 41)."""
    from mynotes.db.session import session_scope
    from mynotes.graph.store import entity_card, find_entities

    with session_scope() as session:
        matches = find_entities(session, ref, type=type)
        candidates = [(entity.canonical_name, entity.type, entity.id) for entity in matches]
        card = entity_card(session, matches[0].id) if len(matches) == 1 else None
    if card is None:
        if not candidates:
            console.print(f"[red]No entity matches {ref!r}.[/]")
        else:
            console.print(f"[red]{len(candidates)} entities match {ref!r}:[/]")
            for name, entity_type, entity_id in candidates:
                console.print(f"  - {name} ({entity_type}) #{entity_id}")
        raise typer.Exit(1)
    if as_json:
        typer.echo(json.dumps(card, ensure_ascii=False, indent=2))
        return
    console.rule(f"mynotes graph · {card['name']} ({card['type']})")
    console.print(
        _kv_table(
            [
                ("id", card["id"]),
                (
                    "status",
                    card["status"]
                    + (
                        f" (merged into {card['merged_into_id']})" if card["merged_into_id"] else ""
                    ),
                ),
                ("description", card["description"] or "-"),
                ("summary stale", card["summary_stale"]),
                ("aliases", ", ".join(card["aliases"]) or "-"),
                ("mention count (current)", card["mention_count_current"]),
                ("relations out / in", f"{card['relations_out']} / {card['relations_in']}"),
            ]
        )
    )
    if card["top_documents"]:
        table = Table(title="Top documents")
        table.add_column("Title", overflow="fold")
        table.add_column("Path", overflow="fold")
        table.add_column("Mentions", justify="right")
        for document in card["top_documents"]:
            table.add_row(
                document["title"] or "-",
                document["rel_path"] or "-",
                str(document["mentions"]),
            )
        console.print(table)


@graph_app.command("neighbors")
def graph_neighbors(
    ref: str = typer.Argument(..., help="Entity name (exact, normalized or fuzzy)."),
    hops: int = typer.Option(2, "--hops", min=1, help="Maximum relation hops."),
    as_json: bool = typer.Option(False, "--json", help="Print the neighbours as JSON."),
) -> None:
    """Show the top 20 one-hop-scored neighbours of an entity (task 43)."""
    from mynotes.db.session import session_scope
    from mynotes.graph.expand import fetch_subgraph, score_entities
    from mynotes.graph.store import find_entities

    with session_scope() as session:
        matches = find_entities(session, ref)
        if len(matches) != 1:
            if not matches:
                console.print(f"[red]No entity matches {ref!r}.[/]")
            else:
                console.print(f"[red]{len(matches)} entities match {ref!r}:[/]")
                for entity in matches:
                    status = f" [dim]({entity.status})[/]" if entity.status != "active" else ""
                    console.print(
                        f"  - {entity.canonical_name} ({entity.type})#{entity.id}{status}"
                    )
            raise typer.Exit(1)
        entity_id = int(matches[0].id)
        sub = fetch_subgraph(session, [entity_id], hops)
        scores = score_entities(sub, {entity_id: 1.0}, method="one_hop")
        relation_types = _neighbour_relation_types(session, sub)
    rows = []
    for neighbour_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])):
        if neighbour_id == entity_id:
            continue
        rows.append(
            {
                "id": neighbour_id,
                "name": sub.nodes.get(neighbour_id, f"entity {neighbour_id}"),
                "score": round(score, 4),
                "types": sorted(relation_types.get(neighbour_id, [])),
            }
        )
        if len(rows) >= 20:
            break
    if as_json:
        typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    if not rows:
        console.print("[yellow]No neighbours found.[/]")
        return
    table = Table(title="mynotes graph neighbors")
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Name", overflow="fold")
    table.add_column("Relation types", style="cyan", overflow="fold")
    table.add_column("Id", justify="right")
    for row in rows:
        table.add_row(
            f"{row['score']:.4f}",
            row["name"],
            ", ".join(row["types"]) or "-",
            str(row["id"]),
        )
    console.print(table)


def _neighbour_relation_types(session: Any, sub: Any) -> dict[int, set[str]]:
    """Relation types of the subgraph edges, keyed by every endpoint."""
    from sqlalchemy import select

    from mynotes.db.models import KgRelation

    if not sub.edges:
        return {}
    relations = {
        int(relation.id): relation
        for relation in session.execute(
            select(KgRelation).where(KgRelation.id.in_([edge[2] for edge in sub.edges]))
        ).scalars()
    }
    types: dict[int, set[str]] = {}
    for src, dst, relation_id, _weight in sub.edges:
        relation = relations.get(relation_id)
        if relation is None:
            continue
        types.setdefault(src, set()).add(relation.type)
        types.setdefault(dst, set()).add(relation.type)
    return types


@graph_app.command("merge")
def graph_merge(
    keep: str = typer.Argument(..., help="The entity reference that survives."),
    drop: str = typer.Argument(..., help="The entity reference folded into KEEP."),
) -> None:
    """Merge one KG entity into another; reversible with 'graph unmerge' (task 41)."""
    from mynotes.db.session import session_scope
    from mynotes.graph.store import merge_entities

    with session_scope() as session:
        keep_id = _entity_ref_id(session, keep)
        drop_id = _entity_ref_id(session, drop) if keep_id is not None else None
        if keep_id is None or drop_id is None:
            raise typer.Exit(1)
        merge_entities(session, keep_id=keep_id, drop_id=drop_id)
    console.print(f"Merged entity {drop_id} into {keep_id} (reversible: 'graph unmerge').")


@graph_app.command("merge-variants")
def graph_merge_variants(
    dry_run: bool = typer.Option(False, "--dry-run", help="List the pairs without merging."),
) -> None:
    """Merge entities that differ only by articles or type words ("X plugin" into "X")."""
    from sqlalchemy import select

    from mynotes.db.models import KgEntity
    from mynotes.db.session import session_scope
    from mynotes.graph.store import merge_name_variants

    with session_scope() as session:
        pairs = merge_name_variants(session, _workspace_id(session), dry_run=dry_run)
        ids = {entity_id for pair in pairs for entity_id in pair}
        names = (
            dict(
                session.execute(
                    select(KgEntity.id, KgEntity.canonical_name).where(KgEntity.id.in_(ids))
                ).all()
            )
            if ids
            else {}
        )
    for keep, drop in pairs:
        console.print(f"  {names.get(drop, drop)!r} -> {names.get(keep, keep)!r}")
    verb = "would merge" if dry_run else "merged"
    suffix = " (reversible: 'graph unmerge')" if pairs and not dry_run else ""
    console.print(f"{verb} {len(pairs)} entity pair(s){suffix}")


@graph_app.command("fold-dates")
def graph_fold_dates(
    dry_run: bool = typer.Option(False, "--dry-run", help="Count without writing."),
) -> None:
    """Turn bare-date entities ("2023-05-20") into date attributes of what they date."""
    from mynotes.db.session import session_scope
    from mynotes.graph.dates import fold_date_entities

    with session_scope() as session:
        report = fold_date_entities(session, _workspace_id(session), dry_run=dry_run)
    verb = "would fold" if dry_run else "folded"
    console.print(
        f"{verb} {report.entities} date entit{'y' if report.entities == 1 else 'ies'}: "
        f"{report.relations_folded} relation(s) became attribute values, "
        f"{report.relations_dropped} dropped, {report.mentions_removed} mention(s) removed"
    )


@graph_app.command("hubs")
def graph_hubs(
    entity_type: str | None = typer.Option(None, "--type", help="Only this entity type."),
    limit: int = typer.Option(20, "--limit", help="Rows to show."),
    json_out: bool = typer.Option(False, "--json", help="Print JSON."),
) -> None:
    """List the most connected entities (review candidates for merging or ignoring)."""
    import json as _json

    from rich.table import Table

    from mynotes.db.session import session_scope
    from mynotes.graph.dates import hub_entities

    with session_scope() as session:
        rows = hub_entities(session, _workspace_id(session), entity_type=entity_type, limit=limit)
    if json_out:
        console.print_json(_json.dumps(rows))
        return
    table = Table("degree", "mentions", "type", "name")
    for row in rows:
        table.add_row(str(row["degree"]), str(row["mentions"]), row["type"], row["name"])
    console.print(table)


@graph_app.command("link-mentions")
def graph_link_mentions(
    dry_run: bool = typer.Option(False, "--dry-run", help="Count without writing."),
) -> None:
    """Link known entities named verbatim in notes that the extractor missed."""
    from mynotes.config import get_settings
    from mynotes.db.session import session_scope
    from mynotes.graph.lexical import LexicalParams, relink_workspace

    params = LexicalParams.from_settings(get_settings())
    if params is None:
        console.print("graph.lexical_mentions is off; nothing to do")
        return
    with session_scope() as session:
        workspace_id = _workspace_id(session)
    report = relink_workspace(workspace_id, params, dry_run=dry_run)
    verb = "would add" if dry_run else "added"
    console.print(
        f"{report['versions']} document(s): removed {report['mentions_removed']} and "
        f"{verb} {report['mentions_added']} lexical mention(s)"
    )


@graph_app.command("unmerge")
def graph_unmerge(
    name: str = typer.Argument(..., help="The merged entity reference to restore."),
) -> None:
    """Undo a merge: restore the entity and record the pair as distinct (task 41)."""
    from mynotes.db.session import session_scope
    from mynotes.graph.store import unmerge_entity

    with session_scope() as session:
        drop_id = _entity_ref_id(session, name)
        if drop_id is None:
            raise typer.Exit(1)
        keep_id = unmerge_entity(session, drop_id)
    if keep_id is None:
        console.print(f"[red]Entity {drop_id} is not merged; nothing to unmerge.[/]")
        raise typer.Exit(1)
    console.print(
        f"Restored entity {drop_id} (was merged into {keep_id}); re-resolution was re-enqueued."
    )


@graph_app.command("timeline")
def graph_timeline(
    ref: str = typer.Argument(..., help="Entity name, or 'project:NAME' for a project tag."),
    as_json: bool = typer.Option(False, "--json", help="Print the timeline as JSON."),
) -> None:
    """Show the temporal timeline of one entity, or of a project tag's documents (task 42)."""
    from mynotes.db.session import session_scope
    from mynotes.graph.temporal import project_entity_id, tag_timeline, timeline

    with session_scope() as session:
        workspace_id = _workspace_id(session)
        cleaned = ref.strip()
        items = []
        if cleaned.lower().startswith("project:"):
            name = cleaned[len("project:") :].strip()
            entity_id = project_entity_id(session, workspace_id, name) if name else None
            items = (
                timeline(session, entity_id)
                if entity_id is not None
                else tag_timeline(session, workspace_id, cleaned)
            )
        else:
            entity_id = _entity_ref_id(session, cleaned)
            if entity_id is None:
                raise typer.Exit(1)
            items = timeline(session, entity_id)
    rows = [
        {
            "date": item.date.date().isoformat() if item.date else None,
            "kind": item.kind,
            "text": item.text,
            "status": item.status,
            "document_path": item.document_path,
        }
        for item in items
    ]
    if as_json:
        typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    if not rows:
        console.print("[yellow]No timeline entries found.[/]")
        return
    table = Table(title="mynotes graph timeline")
    table.add_column("Date", no_wrap=True)
    table.add_column("Kind", style="cyan", no_wrap=True)
    table.add_column("Event", overflow="fold")
    table.add_column("Status", no_wrap=True)
    table.add_column("Path", overflow="fold")
    for row in rows:
        table.add_row(
            row["date"] or "-",
            row["kind"],
            row["text"],
            row["status"],
            row["document_path"] or "-",
        )
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address (default: localhost)."),
    port: int = typer.Option(8765, "--port", min=1, help="Port to listen on."),
    reload: bool = typer.Option(False, "--reload", help="Restart on code changes (development)."),
) -> None:
    """Run the HTTP API (answers, search, reviews, graph) with uvicorn."""
    import uvicorn

    from mynotes.config import get_settings

    if host not in ("127.0.0.1", "localhost", "::1") and get_settings().auth.mode == "none":
        console.print(
            "[yellow]Warning: binding to a non-localhost address with auth.mode='none' "
            "exposes your notes to everyone who can reach this machine.[/]"
        )
    uvicorn.run("mynotes.api.app:create_app", host=host, port=port, reload=reload, factory=True)


@app.command()
def mcp() -> None:
    """Run the MCP server over stdio (Claude Code, Claude Desktop)."""
    from mynotes.mcp.server import mcp as mcp_server

    _setup_logging(False)
    mcp_server.run()


maintenance_app = typer.Typer(help="Database housekeeping (tombstones, caches, queue).")
app.add_typer(maintenance_app, name="maintenance")


@maintenance_app.command("run")
def maintenance_run(
    dry_run: bool = typer.Option(False, "--dry-run", help="Report counts without deleting."),
) -> None:
    """Purge tombstones, clean caches and prune finished jobs/outbox rows."""
    from mynotes.config import get_settings
    from mynotes.jobs.maintenance import run_maintenance

    try:
        report = run_maintenance(get_settings(), dry_run=dry_run)
    except Exception as exc:
        console.print(f"[red]maintenance failed: {exc}[/]")
        raise typer.Exit(1) from exc
    table = Table(title="mynotes maintenance run" + (" (dry run)" if dry_run else ""))
    table.add_column("Operation", style="cyan", no_wrap=True)
    table.add_column("Would act on" if dry_run else "Rows", overflow="fold")
    for operation, operation_counts in report.items():
        table.add_row(operation, _counts_text(operation_counts))
    console.print(table)


acl_app = typer.Typer(help="Team-mode path ACL grants (read).")
app.add_typer(acl_app, name="acl")


@acl_app.command("grant")
def acl_grant(
    principal: str = typer.Argument(..., help="A username or 'group:<name>'."),
    prefix: str = typer.Argument(..., help="Exact folder prefix relative to a root ('' = all)."),
) -> None:
    """Grant a principal read access under one path prefix."""
    from mynotes.core import acl

    try:
        acl.grant(principal, prefix)
    except (ValueError, acl.NotFound) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    shown = acl.normalize_prefix(prefix)
    console.print(f"[green]Granted {principal} read access under '{shown or '/'}'.[/]")


@acl_app.command("revoke")
def acl_revoke(
    principal: str = typer.Argument(..., help="A username or 'group:<name>'."),
    prefix: str = typer.Argument(..., help="Exact folder prefix relative to a root ('' = all)."),
) -> None:
    """Revoke a principal's read access under one path prefix."""
    from mynotes.core import acl

    try:
        acl.revoke(principal, prefix)
    except (ValueError, acl.NotFound) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    shown = acl.normalize_prefix(prefix)
    console.print(f"[green]Revoked {principal.strip()}'s access under '{shown or '/'}'.[/]")


@acl_app.command("list")
def acl_list() -> None:
    """List every ACL grant of the workspace."""
    from mynotes.core import acl

    try:
        grants = acl.list_grants()
    except acl.NotFound as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    if not grants:
        console.print("[yellow]No ACL grants; local mode sees the whole workspace.[/]")
        return
    table = Table(title="mynotes acl grants")
    table.add_column("Principal", style="cyan", no_wrap=True)
    table.add_column("Prefix", no_wrap=True)
    table.add_column("Granted", overflow="fold")
    for item in grants:
        table.add_row(item.principal, item.path_prefix or "(whole workspace)", str(item.created_at))
    console.print(table)


@app.command()
def backup(
    out: Path | None = typer.Option(  # noqa: B008
        None,
        "--out",
        help="Target directory (default: '<user data dir>/mynotes/backups').",
    ),
) -> None:
    """Dump the Postgres database (pg_dump -Fc) and copy mynotes.yaml (never .env)."""
    import platformdirs

    from mynotes.config import get_settings
    from mynotes.jobs.maintenance import create_backup

    directory = out if out is not None else Path(platformdirs.user_data_dir("mynotes")) / "backups"
    try:
        target = create_backup(get_settings(), directory)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    console.print(f"Backup written: [cyan]{target}[/] ({_human_size(target.stat().st_size)})")
    config_copy = directory / "mynotes.yaml"
    if config_copy.is_file():
        console.print(f"Config copied: {config_copy}")


@app.command()
def restore(
    file: Path = typer.Argument(  # noqa: B008
        ..., exists=True, dir_okay=False, readable=True, help="A .dump file from 'mynotes backup'."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Restore a backup, then upgrade the schema and rebuild both indices."""
    from mynotes.config import get_settings
    from mynotes.jobs.maintenance import restore_backup

    if not yes:
        typer.confirm(
            f"Restore {file} into the configured database? This OVERWRITES all current "
            "mynotes data (pg_restore --clean).",
            abort=True,
        )
    try:
        result = restore_backup(get_settings(), file)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    table = Table(title="mynotes restore")
    table.add_column("Step", style="cyan", no_wrap=True)
    table.add_column("Result", overflow="fold")
    table.add_row("restored", str(result["restored"]))
    table.add_row("chunks index", str(result["chunks_index"]))
    table.add_row("entities index", str(result["entities_index"]))
    console.print(table)
