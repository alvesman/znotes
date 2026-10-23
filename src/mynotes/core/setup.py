"""Environment preparation, status collection and endpoint checks.

:func:`run_init` prepares everything ``mynotes`` needs locally (migrations,
workspace, source roots, OpenSearch indices and the RRF pipeline) and reports
each step as a :class:`CheckResult`; :func:`collect_status` gathers the current
state for ``mynotes status``; :func:`check_endpoints` ports the checks of
``scripts/check_endpoints.py`` onto the real task 06/07 clients. Check details
are human-readable and never contain secrets.
"""

from __future__ import annotations

import logging
import math
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from opensearchpy import OpenSearch
from sqlalchemy.orm import Session

from mynotes.config import (
    RootConfig,
    Settings,
    _resolve_yaml_path,
    get_settings,
    missing_model_settings,
)
from mynotes.db.models import (
    Chunk,
    Document,
    DocumentVersion,
    File,
    IndexOutbox,
    Job,
    SourceRoot,
    Workspace,
)
from mynotes.db.session import session_scope
from mynotes.embed.client import get_embedder
from mynotes.index.opensearch import (
    current_physical_index,
    ensure_indices,
    ensure_pipeline,
    get_client,
    inspect_live,
)
from mynotes.index.outbox import enqueue_chunk_deletes, enqueue_chunk_upserts
from mynotes.index.reconcile import expected_chunk_ids, root_chunk_ids
from mynotes.llm.chat import ChatMessage, get_chat
from mynotes.retrieval.rerank_client import get_reranker

logger = logging.getLogger(__name__)

_MAX_MAP_COUNT = 262144
_MAX_MAP_COUNT_PATH = Path("/proc/sys/vm/max_map_count")
_SYNC_FOLDER_MARKERS = ("iCloud Drive", "Mobile Documents", "OneDrive", "Dropbox")
_EXCERPT_CHARS = 300
_NORM_TOLERANCE = 1e-2
_MAX_UNSUPPORTED_EXAMPLES = 10
_EMBED_QUERY = "Como se limpa o filtro do exaustor?"
_EMBED_DOCS = (
    "Para limpar o filtro, retire-o do exaustor e lave-o com água morna e detergente.",
    "O orçamento do projeto foi aprovado pela direção em março.",
)
_RERANK_DOCS = (
    "O orçamento do projeto foi aprovado pela direção em março.",
    "Para limpar o filtro, retire-o do exaustor e lave-o com água morna e detergente.",
    "The filter should be cleaned every month with warm soapy water.",
)


@dataclass
class CheckResult:
    """One setup or endpoint check, safe to show to the user.

    Attributes:
        name: Short check name shown in the first table column.
        ok: False only for ``level == "fail"``; warnings keep the run going.
        detail: Human-readable outcome; never contains secrets.
        level: Severity of the result.
    """

    name: str
    ok: bool
    detail: str
    level: Literal["ok", "warn", "fail"] = "ok"


def _excerpt(text: str) -> str:
    """Return at most 300 characters of ``text`` for a check detail."""
    return text[:_EXCERPT_CHARS]


def _exc_detail(exc: BaseException) -> str:
    """Render an exception as a compact check detail."""
    return _excerpt(f"{type(exc).__name__}: {exc}")


def _ok(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, ok=True, detail=detail, level="ok")


def _warn(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, ok=True, detail=detail, level="warn")


def _fail(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, ok=False, detail=detail, level="fail")


def _check_config() -> tuple[Settings | None, CheckResult]:
    """Load settings and report which YAML file was used (step 1)."""
    try:
        settings = get_settings()
        yaml_path = _resolve_yaml_path()
    except Exception as exc:
        return None, _fail("config", f"could not load settings: {_exc_detail(exc)}")
    source = str(yaml_path) if yaml_path is not None else "defaults (no mynotes.yaml found)"
    if not settings.roots:
        return settings, _warn("config", f"{source}; no roots configured")
    return settings, _ok("config", f"{source}; {len(settings.roots)} root(s)")


def _check_postgres() -> CheckResult:
    """Run ``SELECT 1`` against the configured database (step 2)."""
    try:
        with session_scope() as session:
            session.execute(sa.text("SELECT 1"))
    except Exception as exc:
        return _fail("postgres reachable", _exc_detail(exc))
    return _ok("postgres reachable", "SELECT 1 succeeded")


def _alembic_upgrade_head(dsn: str) -> None:
    """Run ``alembic upgrade head`` against ``dsn``.

    The Alembic config is built in memory (no ``alembic.ini``), so Alembic's
    ``fileConfig`` never replaces the logging set up by ``configure_logging``.
    The DSN travels through the ``-x dsn=...`` override (``alembic/env.py``).
    """
    repo_root = Path(__file__).resolve().parents[3]
    cfg = Config()
    cfg.set_main_option("script_location", str(repo_root / "alembic"))
    cfg.cmd_opts = SimpleNamespace(x=[f"dsn={dsn}"])
    command.upgrade(cfg, "head")


def _check_migrations(settings: Settings, apply_migrations: bool) -> CheckResult:
    """Bring the database schema up to date (step 2)."""
    if not apply_migrations:
        return _ok("migrations", "skipped (--no-migrate)")
    dsn_setting = settings.postgres.dsn
    dsn = dsn_setting.get_secret_value() if dsn_setting is not None else ""
    if not dsn:
        return _fail("migrations", "no Postgres DSN configured")
    try:
        _alembic_upgrade_head(dsn)
    except Exception as exc:
        return _fail("migrations", f"alembic upgrade head failed: {_exc_detail(exc)}")
    return _ok("migrations", "alembic upgrade head: schema is up to date")


def _resolved_root_path(root: RootConfig) -> Path:
    """Return the resolved absolute path of a configured root."""
    return Path(root.path).expanduser().resolve()


def _set_root_enabled(session: Session, row: SourceRoot, enabled: bool, settings: Settings) -> int:
    """Enable or disable a source root and queue the matching index changes.

    A disabled root (removed from the configuration) is no longer scanned and its
    documents leave the search index, but its rows stay in the database, so
    re-adding the root restores them without re-reading or re-embedding files.

    Returns:
        The number of chunk upserts (enable) or deletes (disable) queued.
    """
    row.enabled = enabled
    session.flush()
    if enabled:
        return enqueue_chunk_upserts(
            session, expected_chunk_ids(session, settings, source_root_id=row.id)
        )
    return enqueue_chunk_deletes(session, root_chunk_ids(session, row.id))


def _ensure_workspace_and_roots(settings: Settings) -> list[CheckResult]:
    """Ensure the workspace exists and upsert ``source_root`` rows (step 3).

    Roots listed in the configuration are enabled; roots no longer listed are
    disabled (see :func:`_set_root_enabled`).
    """
    try:
        summaries: list[str] = []
        changes: list[CheckResult] = []
        with session_scope() as session:
            workspace = session.execute(
                sa.select(Workspace).where(Workspace.name == settings.workspace)
            ).scalar_one_or_none()
            if workspace is None:
                workspace = Workspace(name=settings.workspace)
                session.add(workspace)
                session.flush()
            for root in settings.roots:
                abs_path = str(_resolved_root_path(root))
                row = session.execute(
                    sa.select(SourceRoot).where(
                        SourceRoot.workspace_id == workspace.id, SourceRoot.name == root.name
                    )
                ).scalar_one_or_none()
                if row is None:
                    row = SourceRoot(workspace_id=workspace.id, name=root.name, abs_path=abs_path)
                    session.add(row)
                    session.flush()
                elif row.kind == "remote":
                    return [
                        _fail(
                            f"root '{root.name}'",
                            "this name belongs to a remote root fed by an agent; "
                            "choose another name for the local folder",
                        )
                    ]
                else:
                    row.abs_path = abs_path
                    if not row.enabled:
                        queued = _set_root_enabled(session, row, True, settings)
                        changes.append(
                            _ok(
                                f"root '{root.name}'",
                                f"re-enabled; {queued} chunk(s) queued for the search index",
                            )
                        )
                summaries.append(f"{root.name} (id {row.id})")
            configured = {root.name for root in settings.roots}
            removed = session.execute(
                sa.select(SourceRoot).where(
                    SourceRoot.workspace_id == workspace.id,
                    SourceRoot.enabled.is_(True),
                    SourceRoot.kind == "local",
                    SourceRoot.name.not_in(configured),
                )
            ).scalars()
            for row in list(removed):
                queued = _set_root_enabled(session, row, False, settings)
                changes.append(
                    _warn(
                        f"root '{row.name}'",
                        "disabled: no longer in the configuration; no longer scanned, "
                        f"{queued} chunk(s) queued for removal from the search index "
                        "(add the root back and re-run init to restore it)",
                    )
                )
    except Exception as exc:
        return [_fail("workspace & roots", f"could not ensure workspace/roots: {_exc_detail(exc)}")]
    joined = ", ".join(summaries) if summaries else "none"
    results = [_ok("workspace & roots", f"'{settings.workspace}': {joined}"), *changes]
    for root in settings.roots:
        if not Path(root.path).expanduser().exists():
            results.append(_warn(f"root '{root.name}'", f"path does not exist: {root.path}"))
    return results


def _check_storage(settings: Settings) -> list[CheckResult]:
    """The blob store of remote roots: created and writable when set; a warning when needed."""
    directory = settings.storage.dir
    if directory is None:
        try:
            with session_scope() as session:
                remote = session.execute(
                    sa.select(sa.func.count())
                    .select_from(SourceRoot)
                    .where(SourceRoot.kind == "remote", SourceRoot.enabled.is_(True))
                ).scalar_one()
        except Exception:
            remote = 0
        if remote:
            return [
                _warn(
                    "storage",
                    f"{remote} remote root(s) but storage.dir is not set: uploads will fail "
                    "(set storage.dir or MYNOTES_STORAGE_DIR)",
                )
            ]
        return [_ok("storage", "not configured (no remote roots)")]
    path = Path(directory).expanduser()
    try:
        (path / "tmp").mkdir(parents=True, exist_ok=True)
        probe = path / "tmp" / ".write-test"
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as exc:
        return [_fail("storage", f"{path} is not writable: {_exc_detail(exc)}")]
    return [_ok("storage", f"{path} (writable)")]


def _status_storage(settings: Settings | None) -> dict[str, Any]:
    """The blob store's directory and size for ``mynotes status``."""
    if settings is None or settings.storage.dir is None:
        return {"configured": False}
    path = Path(settings.storage.dir).expanduser()
    entry: dict[str, Any] = {"configured": True, "dir": str(path), "exists": path.is_dir()}
    if path.is_dir():
        try:
            from mynotes.storage.blobs import BlobStore

            entry["bytes"] = BlobStore(path).total_size()
        except OSError as exc:
            entry["bytes"] = f"unavailable: {_exc_detail(exc)}"
    return entry


def _check_opensearch(settings: Settings) -> list[CheckResult]:
    """Check OpenSearch reachability, indices, pipeline and mappings (step 4)."""
    try:
        client = get_client()
        info = client.info()
    except Exception as exc:
        return [_fail("opensearch reachable", _exc_detail(exc))]
    version = (info.get("version") or {}).get("number", "unknown")
    results = [_ok("opensearch reachable", f"version {version}")]
    try:
        physical = ensure_indices(client, settings)
        ensure_pipeline(client, settings)
    except Exception as exc:
        results.append(
            _fail("opensearch indices", f"could not ensure indices/pipeline: {_exc_detail(exc)}")
        )
        return results
    joined = ", ".join(f"{alias} -> {index}" for alias, index in sorted(physical.items()))
    results.append(_ok("opensearch indices", joined or "none"))
    try:
        reports = inspect_live(client, settings)
    except Exception as exc:
        results.append(_warn("opensearch inspect", f"live inspection failed: {_exc_detail(exc)}"))
        return results
    for report in reports:
        for problem in report.problems:
            results.append(_warn(f"opensearch: {report.alias}", problem))
        if not report.problems:
            results.append(
                _ok(f"opensearch: {report.alias}", f"'{report.physical}' matches the configuration")
            )
    return results


def _check_max_map_count() -> CheckResult | None:
    """Warn when ``vm.max_map_count`` is too low for OpenSearch (step 5, Linux)."""
    if sys.platform != "linux":
        return None
    try:
        value = int(_MAX_MAP_COUNT_PATH.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        return _warn(
            "vm.max_map_count", f"could not read {_MAX_MAP_COUNT_PATH}: {_exc_detail(exc)}"
        )
    if value >= _MAX_MAP_COUNT:
        return _ok("vm.max_map_count", f"{value} (>= {_MAX_MAP_COUNT})")
    return _warn(
        "vm.max_map_count",
        f"{value} < {_MAX_MAP_COUNT}; run: sudo sysctl -w vm.max_map_count={_MAX_MAP_COUNT}",
    )


def _check_optional_tools() -> list[CheckResult]:
    """Locate the optional OCR tools (step 6)."""
    results: list[CheckResult] = []
    for tool in ("tesseract", "ocrmypdf"):
        found = shutil.which(tool)
        if found:
            results.append(_ok(f"tool: {tool}", found))
        else:
            results.append(_warn(f"tool: {tool}", "not found on PATH; only needed for OCR"))
    return results


def _sync_folder_problem(root: RootConfig) -> str | None:
    """Return a warning text when ``root`` sits inside a sync-managed folder."""
    try:
        resolved = _resolved_root_path(root)
    except OSError as exc:
        return f"could not resolve path {root.path}: {_exc_detail(exc)}"
    marker = next((item for item in _SYNC_FOLDER_MARKERS if item in str(resolved)), None)
    if marker is not None:
        return f"path is inside a {marker} sync folder; sync clients may interfere with scanning"
    if sys.platform == "darwin":
        home = Path.home()
        for protected in ("Documents", "Desktop"):
            if resolved.is_relative_to(home / protected):
                return (
                    f"path is under ~/{protected}; iCloud 'Desktop & Documents' may sync it; "
                    "check System Settings -> Apple ID -> iCloud"
                )
    return None


def _check_sync_folders(settings: Settings) -> list[CheckResult]:
    """Warn about roots inside cloud-sync or macOS protected folders (step 7)."""
    results: list[CheckResult] = []
    for root in settings.roots:
        problem = _sync_folder_problem(root)
        if problem is not None:
            results.append(_warn(f"root '{root.name}' sync folder", problem))
    return results


def run_init(apply_migrations: bool = True) -> list[CheckResult]:
    """Prepare the local mynotes environment and report every step.

    Args:
        apply_migrations: Run ``alembic upgrade head`` (``--no-migrate`` skips it).

    Returns:
        One :class:`CheckResult` per step, in execution order. Warnings do not
        stop the run; a failure skips only the steps that depend on it.
    """
    results: list[CheckResult] = []
    settings, config_result = _check_config()
    results.append(config_result)
    if settings is None:
        return results
    missing = missing_model_settings(settings)
    if missing:
        # The search indices are created with the embedding dimension: stop before them.
        results.append(
            _fail(
                "models",
                "not configured: "
                + ", ".join(missing)
                + ". Set them in mynotes.yaml or MYNOTES_* variables (see mynotes.example.yaml)",
            )
        )
        return results
    postgres_result = _check_postgres()
    results.append(postgres_result)
    if postgres_result.level == "fail":
        results.append(_warn("migrations", "skipped: Postgres is unreachable"))
        results.append(_warn("workspace & roots", "skipped: Postgres is unreachable"))
    else:
        migration_result = _check_migrations(settings, apply_migrations)
        results.append(migration_result)
        if migration_result.level == "fail":
            results.append(_warn("workspace & roots", "skipped: migrations failed"))
        else:
            results.extend(_ensure_workspace_and_roots(settings))
    results.extend(_check_opensearch(settings))
    results.extend(_check_storage(settings))
    map_result = _check_max_map_count()
    if map_result is not None:
        results.append(map_result)
    results.extend(_check_optional_tools())
    results.extend(_check_sync_folders(settings))
    return results


def _status_roots(settings: Settings | None) -> list[dict[str, Any]]:
    """Configured roots with their ``source_root`` ids from Postgres."""
    if settings is None:
        return []
    roots: list[dict[str, Any]] = []
    for root in settings.roots:
        entry: dict[str, Any] = {"name": root.name, "path": str(root.path)}
        try:
            with session_scope() as session:
                row_id = session.execute(
                    sa.select(SourceRoot.id)
                    .join(Workspace, Workspace.id == SourceRoot.workspace_id)
                    .where(Workspace.name == settings.workspace, SourceRoot.name == root.name)
                ).scalar_one_or_none()
            entry["source_root_id"] = int(row_id) if row_id is not None else None
        except Exception as exc:
            entry["source_root_id"] = f"unavailable: {_exc_detail(exc)}"
        roots.append(entry)
    try:
        with session_scope() as session:
            remote = session.execute(
                sa.select(SourceRoot)
                .join(Workspace, Workspace.id == SourceRoot.workspace_id)
                .where(
                    Workspace.name == settings.workspace,
                    SourceRoot.kind == "remote",
                    SourceRoot.enabled.is_(True),
                )
                .order_by(SourceRoot.id)
            ).scalars()
            for row in remote:
                roots.append(
                    {
                        "name": row.name,
                        "path": f"remote (agent of {row.owner or 'unknown owner'})",
                        "source_root_id": int(row.id),
                        "remote": True,
                        "owner": row.owner,
                        "agent_last_seen_at": (
                            row.agent_last_seen_at.isoformat() if row.agent_last_seen_at else None
                        ),
                    }
                )
    except Exception as exc:
        roots.append({"name": "(remote roots)", "path": f"unavailable: {_exc_detail(exc)}"})
    return roots


def _status_table_sizes(session: Any) -> dict[str, int]:
    """Total on-disk size (indexes and toast included) of the largest tables.

    Purely informational: ``mynotes status`` shows it so the user can decide on
    a manual ``VACUUM``; nothing is ever vacuumed automatically (task 50).
    """
    rows = session.execute(
        sa.text(
            "SELECT relname, pg_total_relation_size(relid) AS total "
            "FROM pg_catalog.pg_statio_user_tables ORDER BY total DESC LIMIT 10"
        )
    ).all()
    return {relname: int(total) for relname, total in rows}


def _status_postgres() -> dict[str, Any]:
    """Corpus, job queue and outbox counts from Postgres."""
    try:
        with session_scope() as session:
            files_by_status = dict(
                session.execute(sa.select(File.status, sa.func.count()).group_by(File.status)).all()
            )
            documents_by_status = dict(
                session.execute(
                    sa.select(Document.status, sa.func.count()).group_by(Document.status)
                ).all()
            )
            versions = int(
                session.scalar(sa.select(sa.func.count()).select_from(DocumentVersion)) or 0
            )
            chunks = int(session.scalar(sa.select(sa.func.count()).select_from(Chunk)) or 0)
            job_counts = dict(
                session.execute(sa.select(Job.status, sa.func.count()).group_by(Job.status)).all()
            )
            outbox = int(
                session.scalar(
                    sa.select(sa.func.count())
                    .select_from(IndexOutbox)
                    .where(IndexOutbox.processed_at.is_(None))
                )
                or 0
            )
            table_sizes = _status_table_sizes(session)
    except Exception as exc:
        return {"error": f"unavailable: {_exc_detail(exc)}"}
    return {
        "files_by_status": files_by_status,
        "documents_by_status": documents_by_status,
        "versions": versions,
        "chunks": chunks,
        "jobs": {
            status: int(job_counts.get(status, 0)) for status in ("queued", "running", "failed")
        },
        "outbox_unprocessed": outbox,
        "table_sizes": table_sizes,
    }


def _status_opensearch(settings: Settings | None) -> dict[str, Any]:
    """Alias-to-index resolution, doc counts and live mapping problems."""
    if settings is None:
        return {"error": "unavailable: settings could not be loaded"}
    try:
        client = get_client()
    except Exception as exc:
        return {"error": f"unavailable: {_exc_detail(exc)}"}
    aliases: dict[str, Any] = {}
    for alias in (settings.opensearch.chunks_alias, settings.opensearch.entities_alias):
        try:
            physical = current_physical_index(client, alias)
            docs = int(client.count(index=physical)["count"]) if physical is not None else 0
            aliases[alias] = {"physical": physical, "docs": docs}
        except Exception as exc:
            aliases[alias] = f"unavailable: {_exc_detail(exc)}"
    try:
        problems = [
            problem for report in inspect_live(client, settings) for problem in report.problems
        ]
    except Exception as exc:
        problems = [f"unavailable: {_exc_detail(exc)}"]
    try:
        mismatch = _old_versions_mismatch(client, settings)
    except Exception as exc:
        mismatch = f"unavailable: {_exc_detail(exc)}"
    if mismatch is not None:
        problems.append(mismatch)
    return {"aliases": aliases, "problems": problems}


def _old_versions_mismatch(client: OpenSearch, settings: Settings) -> str | None:
    """Detect an index built under a different ``versions.index_old_versions``.

    The regular embed/sync flow only keeps the chunks it writes consistent, so
    flipping the setting leaves stale old-version chunks behind (setting off)
    or old snapshots missing from the index (setting on); both require
    ``mynotes index rebuild``.

    Args:
        client: An OpenSearch client.
        settings: Application settings.

    Returns:
        A human-readable problem description, or None when index and setting
        agree (or there is nothing to compare against yet).
    """
    physical = current_physical_index(client, settings.opensearch.chunks_alias)
    if physical is None:
        return None
    old_indexed = int(
        client.count(index=physical, body={"query": {"term": {"is_latest": False}}})["count"]
    )
    if not settings.versions.index_old_versions:
        if old_indexed > 0:
            return (
                f"{old_indexed} old-version chunk(s) are indexed while "
                "versions.index_old_versions is off; run 'mynotes index rebuild' "
                "to apply the setting"
            )
        return None
    with session_scope() as session:
        multi_version_documents = (
            session.scalar(
                sa.select(sa.func.count()).select_from(
                    sa.select(DocumentVersion.document_id)
                    .join(Document, Document.id == DocumentVersion.document_id)
                    .where(Document.status != "deleted")
                    .group_by(DocumentVersion.document_id)
                    .having(sa.func.count() > 1)
                    .subquery()
                )
            )
            or 0
        )
    if multi_version_documents and old_indexed == 0:
        return (
            "versions.index_old_versions is on but no old-version chunks are "
            "indexed; run 'mynotes index rebuild' to apply the setting"
        )
    return None


def _status_endpoints() -> list[dict[str, Any]]:
    """Guard snapshots (calls, latency, breaker state) for every endpoint.

    This process's snapshots first (``process`` = its role), then the rows the
    other processes published (the worker, every 30 s), each with ``age_s``.
    """
    from mynotes.jobs.endpoint_stats import endpoint_views

    try:
        try:
            with session_scope() as session:
                return endpoint_views(session)
        except Exception as exc:
            logger.warning("published endpoint stats unavailable: %s", _exc_detail(exc))
            return endpoint_views(None)
    except Exception as exc:
        return [{"error": f"unavailable: {_exc_detail(exc)}"}]


def _status_unsupported() -> dict[str, Any]:
    """Count and example paths of files the scanner recorded as unsupported."""
    try:
        with session_scope() as session:
            count = int(
                session.scalar(
                    sa.select(sa.func.count()).select_from(File).where(File.status == "unsupported")
                )
                or 0
            )
            examples = [
                rel_path
                for (rel_path,) in session.execute(
                    sa.select(File.rel_path)
                    .where(File.status == "unsupported")
                    .order_by(File.rel_path)
                    .limit(_MAX_UNSUPPORTED_EXAMPLES)
                ).all()
            ]
    except Exception as exc:
        return {"error": f"unavailable: {_exc_detail(exc)}"}
    return {"count": count, "examples": examples}


def collect_status() -> dict[str, Any]:
    """Collect the current state of config, database, indices and endpoints.

    Returns:
        A JSON-serializable dict (rendered by ``mynotes status``). Every part
        catches its own errors and reports ``"unavailable: <error>"`` instead
        of crashing.
    """
    status: dict[str, Any] = {}
    settings: Settings | None = None
    try:
        settings = get_settings()
        yaml_path = _resolve_yaml_path()
        status["config_file"] = str(yaml_path) if yaml_path is not None else None
        status["workspace"] = settings.workspace
    except Exception as exc:
        status["config_file"] = f"unavailable: {_exc_detail(exc)}"
        status["workspace"] = "unavailable"
    status["roots"] = _status_roots(settings)
    status["postgres"] = _status_postgres()
    status["unsupported"] = _status_unsupported()
    status["opensearch"] = _status_opensearch(settings)
    status["storage"] = _status_storage(settings)
    status["endpoints"] = _status_endpoints()
    return status


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine of the angle between ``a`` and ``b`` (0.0 for zero vectors)."""
    numerator = sum(x * y for x, y in zip(a, b, strict=False))
    denominator = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return numerator / denominator if denominator else 0.0


def _check_embeddings() -> list[CheckResult]:
    """Dimension, normalization and relevance sanity of the embedding endpoint."""
    try:
        embedder = get_embedder()
        query = embedder.embed_query(_EMBED_QUERY)
        docs = embedder.embed_documents(_EMBED_DOCS)
    except Exception as exc:
        return [_fail("embedding", f"call failed: {_exc_detail(exc)}")]
    vectors = [query, *docs]
    dims = sorted({len(vector) for vector in vectors})
    if dims != [embedder.dim]:
        return [_fail("embedding: dimension", f"got {dims}, expected [{embedder.dim}]")]
    results = [_ok("embedding: dimension", f"{dims[0]} as configured")]
    norms = [math.sqrt(sum(value * value for value in vector)) for vector in vectors]
    shown = ", ".join(f"{norm:.3f}" for norm in norms)
    if all(abs(norm - 1.0) < _NORM_TOLERANCE for norm in norms):
        results.append(_ok("embedding: norm", f"L2-normalized: [{shown}]"))
    else:
        results.append(
            _warn(
                "embedding: norm",
                f"not normalized server-side: [{shown}]; the client normalizes (normalize: auto)",
            )
        )
    relevant, irrelevant = _cosine(query, docs[0]), _cosine(query, docs[1])
    detail = f"relevant={relevant:.3f} irrelevant={irrelevant:.3f}"
    if relevant > irrelevant:
        results.append(_ok("embedding: relevance sanity", detail))
    else:
        results.append(_fail("embedding: relevance sanity", detail))
    return results


def _check_reranker() -> list[CheckResult]:
    """Ordering sanity and a 40-document request against the reranker."""
    if not get_settings().models.reranker.url:
        return [_warn("reranker", "not configured: results keep their search order (optional)")]
    try:
        reranker = get_reranker()
        top_results = reranker.rerank(_EMBED_QUERY, _RERANK_DOCS, top_n=len(_RERANK_DOCS))
    except Exception as exc:
        return [_fail("reranker", f"call failed: {_exc_detail(exc)}")]
    top = top_results[0].index if top_results else None
    detail = f"top document index = {top} (expected 1 or 2)"
    if top in (1, 2):
        results = [_ok("reranker: ordering", detail)]
    else:
        return [_fail("reranker: ordering", detail)]
    many_docs = [
        f"Documento {index}: notas sobre o projeto e o filtro do exaustor." for index in range(40)
    ]
    try:
        many = reranker.rerank(_EMBED_QUERY, many_docs, top_n=len(many_docs))
    except Exception as exc:
        results.append(_fail("reranker: 40 documents", f"call failed: {_exc_detail(exc)}"))
        return results
    detail = f"{len(many)} results"
    if len(many) == len(many_docs):
        results.append(_ok("reranker: 40 documents", detail))
    else:
        results.append(_fail("reranker: 40 documents", f"{detail}, expected {len(many_docs)}"))
    return results


def _check_chat() -> list[CheckResult]:
    """One chat completion with thinking disabled."""
    try:
        chat = get_chat()
        answer = chat.complete(
            [ChatMessage(role="user", content="Responde apenas: OK")], max_tokens=20
        )
    except Exception as exc:
        return [_fail("llm: chat", f"call failed: {_exc_detail(exc)}")]
    if "</think>" in answer.text:
        return [
            _fail("llm: chat", "response still contains a think block; thinking is not disabled")
        ]
    return [_ok("llm: chat", f"thinking disabled: {answer.text.strip()[:80]!r}")]


def check_endpoints(include_llm: bool = True) -> list[CheckResult]:
    """Verify the embedding, reranker (and chat) model endpoints.

    Ports the sanity checks of ``scripts/check_endpoints.py`` onto the real
    clients; concurrency probing is not included.

    Args:
        include_llm: Also run the chat check (``--no-llm`` skips it).

    Returns:
        One :class:`CheckResult` per check; a failing endpoint produces a
        single ``fail`` for that endpoint.
    """
    results = _check_embeddings()
    results.extend(_check_reranker())
    if include_llm:
        results.extend(_check_chat())
    return results
