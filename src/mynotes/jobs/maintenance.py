"""Idempotent maintenance handlers, backup and restore (task 50).

The three handlers keep a long-running install tidy: ``purge_tombstones``
removes documents (and their files) whose deletion is older than
``retention.deleted_days``, ``clean_caches`` drops orphaned cache rows and
``prune_ops`` removes finished jobs and processed outbox rows. Every handler
is idempotent (the registry requires it: crash recovery can re-run a job) and
reports the number of rows it acted on; with ``dry_run=True`` it only counts.

The backup/restore helpers shell out to ``pg_dump``/``pg_restore``. They prefer
the Docker Compose Postgres service when the configured DSN points at it and
fall back to a local ``pg_dump`` from ``PATH``. OpenSearch is never backed up:
both indices are always rebuildable from Postgres (``mynotes index rebuild``).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from sqlalchemy import (
    Text,
    column,
    delete,
    func,
    select,
    table,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from mynotes.config import Settings
from mynotes.db.models import (
    Chunk,
    Document,
    EmbeddingCache,
    File,
    IndexOutbox,
    Job,
    KgEntity,
    KgExtractionCache,
    KgMention,
)
from mynotes.db.session import session_scope
from mynotes.embed.cache import embed_key
from mynotes.embed.client import Embedder
from mynotes.graph.ontology import EXTRACTOR_VERSION
from mynotes.index.outbox import enqueue_chunk_deletes, enqueue_entity_deletes
from mynotes.jobs.registry import JobContext, Priority, handler

logger = logging.getLogger(__name__)

MAINTENANCE_JOB_TYPES = ("purge_tombstones", "clean_caches", "prune_ops")
"""The job types the scheduler's nightly ``maintenance`` task enqueues."""

_PURGE_BATCH = 200
"""Documents deleted per loop iteration (one transaction; bounds memory only)."""

_LIVE_KEYS_TABLE = "maintenance_live_embed_keys"
"""Name of the per-transaction temp table holding the live embed keys."""

_STREAM_BATCH = 1000
"""Chunks streamed per temp-table insert batch."""


def daily_dedupe_key(job_type: str) -> str:
    """Return the scheduler's daily dedupe key for a maintenance job type."""
    return f"{job_type}:daily"


def _cutoff(days: int) -> datetime:
    """Return ``now - days`` as an aware UTC datetime."""
    return datetime.now(UTC) - timedelta(days=days)


def _live_keys_table():
    """A SQLAlchemy core table mirroring the live-keys temp table."""
    return table(_LIVE_KEYS_TABLE, column("key", Text))


def _zero_mention_clause(eligible_ids: list[int] | None) -> Any:
    """Clause: the entity has no mention outside ``eligible_ids``.

    With ``None`` (or an empty purge list) the clause is "no mentions at all",
    which equals the post-purge state when nothing is purged.
    """
    mention_exists = select(KgMention.id).where(KgMention.entity_id == KgEntity.id)
    if not eligible_ids:
        return ~mention_exists.exists()
    return (
        ~mention_exists.join(Chunk, Chunk.id == KgMention.chunk_id)
        .where(Chunk.document_id.not_in(eligible_ids))
        .exists()
    )


def purge_tombstones(
    session: Session, settings: Settings, *, dry_run: bool = False
) -> dict[str, int]:
    """Delete tombstoned documents older than ``retention.deleted_days``.

    For every document with ``status='deleted'`` and ``updated_at`` before the
    cutoff, the remaining chunk ids are first queued as outbox deletes (a
    safety net for chunks the ``delete_document`` job never saw), then the
    file rows with ``status='deleted'`` and the document row are deleted; the
    schema cascades remove its versions, chunks, tags, relations, mentions and
    evidence. After the deletions, entities with zero remaining mentions are
    marked ``stale`` and their index documents are queued for removal (an
    index rebuild only places active entities).

    Args:
        session: Open session/transaction; the caller commits.
        settings: Application settings (the ``retention`` section).
        dry_run: Count the would-be purges without deleting anything.

    Returns:
        ``{"documents": ..., "files": ..., "entities_stale": ...}``.
    """
    cutoff = _cutoff(settings.retention.deleted_days)
    eligible: list[int] = list(
        session.execute(
            select(Document.id).where(Document.status == "deleted", Document.updated_at < cutoff)
        ).scalars()
    )
    counts = {"documents": 0, "files": 0, "entities_stale": 0}
    if dry_run:
        counts["documents"] = len(eligible)
        if eligible:
            counts["files"] = int(
                session.scalar(
                    select(func.count())
                    .select_from(File)
                    .where(File.document_id.in_(eligible), File.status == "deleted")
                )
                or 0
            )
        counts["entities_stale"] = int(
            session.scalar(
                select(func.count())
                .select_from(KgEntity)
                .where(KgEntity.status == "active", _zero_mention_clause(eligible))
            )
            or 0
        )
        return counts
    for start in range(0, len(eligible), _PURGE_BATCH):
        batch = eligible[start : start + _PURGE_BATCH]
        chunk_ids = list(
            session.execute(select(Chunk.id).where(Chunk.document_id.in_(batch))).scalars()
        )
        if chunk_ids:
            enqueue_chunk_deletes(session, chunk_ids)
        counts["files"] += session.execute(
            delete(File).where(File.document_id.in_(batch), File.status == "deleted")
        ).rowcount
        # ``superseded_by_document_id`` is a plain self-FK: point the survivors
        # at nothing before the referenced tombstones disappear.
        session.execute(
            update(Document)
            .where(Document.superseded_by_document_id.in_(batch))
            .values(superseded_by_document_id=None)
        )
        counts["documents"] += session.execute(
            delete(Document).where(Document.id.in_(batch))
        ).rowcount
    stale_ids = list(
        session.execute(
            select(KgEntity.id).where(KgEntity.status == "active", _zero_mention_clause(None))
        ).scalars()
    )
    if stale_ids:
        session.execute(
            update(KgEntity)
            .where(KgEntity.id.in_(stale_ids))
            .values(status="stale", updated_at=datetime.now(UTC))
        )
        enqueue_entity_deletes(session, stale_ids)
    counts["entities_stale"] = len(stale_ids)
    if eligible:
        logger.info("purged %d tombstoned document(s)", len(eligible))
    return counts


def _fill_live_embed_keys(session: Session, embedder: Embedder) -> None:
    """Hash every chunk's ``embed_text`` into a temp table of live cache keys.

    The chunks are streamed in batches with a server-side cursor, so a large
    corpus never holds all keys in Python memory at once. The temp table lives
    for the current transaction only (``ON COMMIT DROP``).
    """
    session.execute(
        text(f"CREATE TEMP TABLE {_LIVE_KEYS_TABLE} (key text PRIMARY KEY) ON COMMIT DROP")
    )
    live = _live_keys_table()
    result = session.execute(select(Chunk.embed_text).execution_options(yield_per=_STREAM_BATCH))
    batch: list[str] = []
    total = 0
    for (embed_text,) in result:
        batch.append(embed_key(embedder, embed_text))
        if len(batch) >= _STREAM_BATCH:
            total += _insert_live_keys(session, live, batch)
            batch.clear()
    total += _insert_live_keys(session, live, batch)
    logger.debug("collected %d live embed key(s) into '%s'", total, _LIVE_KEYS_TABLE)


def _insert_live_keys(session: Session, live: Any, keys: list[str]) -> int:
    """Insert one batch of keys, ignoring duplicates within and across batches."""
    unique = list(dict.fromkeys(keys))
    if not unique:
        return 0
    session.execute(
        pg_insert(live)
        .values([{"key": key} for key in unique])
        .on_conflict_do_nothing(index_elements=[live.c.key])
    )
    return len(unique)


def clean_caches(
    session: Session, settings: Settings, embedder: Embedder, *, dry_run: bool = False
) -> dict[str, int]:
    """Delete unused cache rows older than ``retention.cache_orphan_days``.

    ``embedding_cache`` rows are removed when their key is not the
    :func:`~mynotes.embed.cache.embed_key` of any existing chunk; the live keys
    are streamed into a transaction-scoped temp table first. ``kg_extraction_cache``
    rows cannot be matched against current group hashes cheaply, so every row
    with ``extractor_version != EXTRACTOR_VERSION`` past the threshold is
    removed instead.

    Args:
        session: Open session/transaction; the caller commits.
        settings: Application settings (the ``retention`` section).
        embedder: The embedder whose ``passage_prefix``/model define the keys.
        dry_run: Count the would-be deletions without deleting anything.

    Returns:
        ``{"embedding_cache": ..., "kg_extraction_cache": ...}``.
    """
    cutoff = _cutoff(settings.retention.cache_orphan_days)
    _fill_live_embed_keys(session, embedder)
    live = _live_keys_table()
    orphan = ~select(live.c.key).where(live.c.key == EmbeddingCache.content_hash).exists()
    embedding_criteria = (EmbeddingCache.created_at < cutoff, orphan)
    kg_criteria = (
        KgExtractionCache.extractor_version != EXTRACTOR_VERSION,
        KgExtractionCache.created_at < cutoff,
    )
    if dry_run:
        return {
            "embedding_cache": int(
                session.scalar(
                    select(func.count()).select_from(EmbeddingCache).where(*embedding_criteria)
                )
                or 0
            ),
            "kg_extraction_cache": int(
                session.scalar(
                    select(func.count()).select_from(KgExtractionCache).where(*kg_criteria)
                )
                or 0
            ),
        }
    embedding_count = session.execute(delete(EmbeddingCache).where(*embedding_criteria)).rowcount
    kg_count = session.execute(delete(KgExtractionCache).where(*kg_criteria)).rowcount
    if embedding_count or kg_count:
        logger.info(
            "cleaned %d embedding and %d kg extraction cache row(s)", embedding_count, kg_count
        )
    return {"embedding_cache": int(embedding_count), "kg_extraction_cache": int(kg_count)}


def prune_ops(session: Session, settings: Settings, *, dry_run: bool = False) -> dict[str, int]:
    """Delete finished jobs and processed outbox rows past their retention.

    ``job`` rows with status ``done``/``dead`` whose ``finished_at`` (falling
    back to ``created_at``) is older than ``retention.jobs_done_days`` are
    removed; dead jobs' last errors are logged first so the failure history
    survives the pruning. ``index_outbox`` rows processed longer than
    ``retention.outbox_processed_days`` ago are removed as well.

    Args:
        session: Open session/transaction; the caller commits.
        settings: Application settings (the ``retention`` section).
        dry_run: Count the would-be prunes without deleting anything.

    Returns:
        ``{"jobs": ..., "index_outbox": ...}``.
    """
    jobs_cutoff = _cutoff(settings.retention.jobs_done_days)
    outbox_cutoff = _cutoff(settings.retention.outbox_processed_days)
    finished_before = func.coalesce(Job.finished_at, Job.created_at) < jobs_cutoff
    job_criteria = (Job.status.in_(("done", "dead")), finished_before)
    outbox_criteria = (
        IndexOutbox.processed_at.is_not(None),
        IndexOutbox.processed_at < outbox_cutoff,
    )
    if dry_run:
        return {
            "jobs": int(
                session.scalar(select(func.count()).select_from(Job).where(*job_criteria)) or 0
            ),
            "index_outbox": int(
                session.scalar(
                    select(func.count()).select_from(IndexOutbox).where(*outbox_criteria)
                )
                or 0
            ),
        }
    dead_rows = session.execute(
        select(Job.id, Job.type, Job.last_error).where(Job.status == "dead", finished_before)
    ).all()
    for job_id, job_type, last_error in dead_rows:
        logger.warning(
            "pruning dead job %d (type=%s), last error: %s",
            job_id,
            job_type,
            (last_error or "(no error)")[:500],
        )
    jobs = session.execute(delete(Job).where(*job_criteria)).rowcount
    outbox = session.execute(delete(IndexOutbox).where(*outbox_criteria)).rowcount
    if jobs or outbox:
        logger.info("pruned %d job(s) and %d processed outbox row(s)", jobs, outbox)
    return {"jobs": int(jobs), "index_outbox": int(outbox)}


@handler("purge_tombstones")
def handle_purge_tombstones(payload: dict[str, Any], context: JobContext) -> None:
    """Delete tombstoned documents past ``retention.deleted_days``."""
    with session_scope() as session:
        counts = purge_tombstones(session, context.settings)
    logger.info("purge_tombstones done: %s", counts)


@handler("clean_caches")
def handle_clean_caches(payload: dict[str, Any], context: JobContext) -> None:
    """Drop orphaned embedding and outdated kg extraction cache rows."""
    with session_scope() as session:
        counts = clean_caches(session, context.settings, context.embedder)
    logger.info("clean_caches done: %s", counts)
    if context.settings.storage.dir is not None:
        from mynotes.storage.blobs import get_blob_store
        from mynotes.storage.gc import collect_garbage

        try:
            with session_scope() as session:
                collect_garbage(session, get_blob_store())
        except Exception:
            logger.warning("blob store gc failed", exc_info=True)


@handler("prune_ops")
def handle_prune_ops(payload: dict[str, Any], context: JobContext) -> None:
    """Prune done/dead jobs and processed outbox rows."""
    with session_scope() as session:
        counts = prune_ops(session, context.settings)
    logger.info("prune_ops done: %s", counts)


def _libpq_dsn(settings: Settings) -> str:
    """The configured DSN as a libpq URI (``postgresql://...``)."""
    dsn = settings.postgres.dsn
    if dsn is None or not dsn.get_secret_value():
        raise RuntimeError(
            "no Postgres DSN configured; set MYNOTES_PG_DSN in .env or postgres.dsn in mynotes.yaml"
        )
    raw = dsn.get_secret_value()
    parts = urlsplit(raw)
    if parts.scheme.startswith("postgresql+"):
        raw = raw.replace(f"{parts.scheme}:", "postgresql:", 1)
    return raw


def _compose_base(settings: Settings) -> list[str] | None:
    """``docker compose`` base argv when the compose Postgres serves this DSN.

    Returns None when Docker/compose is unavailable, the postgres service is
    not running, or the configured DSN clearly points elsewhere (a loopback
    host with a port other than the compose mapping).
    """
    docker = shutil.which("docker")
    if docker is None:
        return None
    repo_compose = Path(__file__).resolve().parents[3] / "docker-compose.yml"
    base = [docker, "compose"]
    if repo_compose.is_file():
        base += ["-f", str(repo_compose)]
    else:
        base += ["-p", "mynotes"]
    try:
        running = subprocess.run(
            [*base, "ps", "--services", "--filter", "status=running"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if running.returncode != 0 or "postgres" not in running.stdout.split():
            return None
    except (OSError, subprocess.SubprocessError):
        return None
    if (
        settings.postgres.dsn is not None
        and settings.postgres.dsn.get_secret_value()
        and not _dsn_matches_compose(settings, base)
    ):
        return None
    return base


def _dsn_matches_compose(settings: Settings, base: list[str]) -> bool:
    """Whether the DSN's host/port match the compose postgres mapping."""
    parts = urlsplit(_libpq_dsn(settings))
    host = (parts.hostname or "").lower()
    if host not in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return False
    try:
        published = subprocess.run(
            [*base, "port", "postgres", "5432"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if published.returncode != 0 or ":" not in published.stdout:
        return False
    try:
        compose_port = int(published.stdout.strip().rsplit(":", 1)[1])
    except ValueError:
        return False
    return compose_port == (parts.port or 5432)


def _pg_user_db(settings: Settings) -> tuple[str, str]:
    """The (user, database) from the configured DSN, with compose defaults."""
    parts = urlsplit(_libpq_dsn(settings))
    return parts.username or "mynotes", (parts.path or "/mynotes").lstrip("/") or "mynotes"


def dump_command(settings: Settings) -> list[str]:
    """The ``pg_dump -Fc`` command for the configured database.

    Prefers the Docker Compose service when it runs this DSN; falls back to a
    local ``pg_dump`` from ``PATH``.

    Raises:
        RuntimeError: When neither path is available.
    """
    base = _compose_base(settings)
    if base is not None:
        user, database = _pg_user_db(settings)
        return [*base, "exec", "-T", "postgres", "pg_dump", "-U", user, "-d", database, "-Fc"]
    pg_dump = shutil.which("pg_dump")
    if pg_dump is None:
        raise RuntimeError(
            "no way to dump: the Docker Compose postgres is not running and pg_dump is not on PATH"
        )
    return [pg_dump, f"--dbname={_libpq_dsn(settings)}", "--format=custom"]


def restore_command(settings: Settings) -> list[str]:
    """The ``pg_restore --clean --if-exists`` command reading a dump on stdin.

    Raises:
        RuntimeError: When neither the Compose service nor a local
            ``pg_restore`` is available.
    """
    base = _compose_base(settings)
    if base is not None:
        user, database = _pg_user_db(settings)
        return [
            *base,
            "exec",
            "-T",
            "postgres",
            "pg_restore",
            "-U",
            user,
            "-d",
            database,
            "--clean",
            "--if-exists",
        ]
    pg_restore = shutil.which("pg_restore")
    if pg_restore is None:
        raise RuntimeError(
            "no way to restore: the Docker Compose postgres is not running and "
            "pg_restore is not on PATH"
        )
    return [pg_restore, f"--dbname={_libpq_dsn(settings)}", "--clean", "--if-exists"]


def create_backup(settings: Settings, out_dir: Path) -> Path:
    """Write a ``pg_dump -Fc`` archive of the database into ``out_dir``.

    The dump is named ``mynotes_<UTC timestamp>.dump``; when a ``mynotes.yaml``
    configuration file is in use it is copied next to the dump. Secrets live in
    ``.env``, which is never copied.

    Args:
        settings: Application settings (the Postgres DSN).
        out_dir: Target directory; created when missing.

    Returns:
        The path of the written dump file.

    Raises:
        RuntimeError: When pg_dump is unavailable or fails.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"mynotes_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.dump"
    command = dump_command(settings)
    logger.info("backup: running %s", command[0])
    with target.open("wb") as sink:
        proc = subprocess.run(command, stdout=sink, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        target.unlink(missing_ok=True)
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"pg_dump failed (exit {proc.returncode}): {detail}")
    if not target.stat().st_size:
        target.unlink(missing_ok=True)
        raise RuntimeError("pg_dump produced an empty archive; check the database connection")
    _copy_config(out_dir)
    logger.info("backup: wrote %s (%d bytes)", target, target.stat().st_size)
    return target


def _copy_config(out_dir: Path) -> None:
    """Copy the in-use ``mynotes.yaml`` next to the dump; never touch ``.env``."""
    from mynotes.config import _resolve_yaml_path

    config_path = _resolve_yaml_path()
    if config_path is None or not config_path.is_file():
        return
    shutil.copy2(config_path, out_dir / "mynotes.yaml")
    logger.info("backup: copied config %s into %s", config_path, out_dir)


def restore_backup(settings: Settings, dump_file: Path) -> dict[str, Any]:
    """Restore a dump into the configured database and rebuild the indices.

    Runs ``pg_restore --clean --if-exists`` (reading the dump on stdin),
    ``alembic upgrade head`` (a dump from an older schema is brought up to
    date), then rebuilds the chunks and entities indices from Postgres.

    Args:
        settings: Application settings (the Postgres DSN).
        dump_file: The ``pg_dump -Fc`` archive to restore.

    Returns:
        ``{"restored": file, "chunks_index": ..., "entities_index": ...}``.

    Raises:
        RuntimeError: When pg_restore is unavailable or fails.
    """
    if not dump_file.is_file():
        raise RuntimeError(f"dump file not found: {dump_file}")
    command = restore_command(settings)
    logger.info("restore: running %s", command[0])
    with dump_file.open("rb") as source:
        proc = subprocess.run(command, stdin=source, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"pg_restore failed (exit {proc.returncode}): {detail}")
    from mynotes.core.setup import _alembic_upgrade_head

    _alembic_upgrade_head(_libpq_dsn(settings))
    logger.info("restore: schema is up to date (alembic upgrade head)")
    from mynotes.index.rebuild import rebuild_chunks_index, rebuild_entities_index

    chunks = rebuild_chunks_index()
    entities = rebuild_entities_index()
    return {
        "restored": str(dump_file),
        "chunks_index": chunks.get("new"),
        "entities_index": entities.get("new"),
    }


def run_maintenance(settings: Settings, *, dry_run: bool = False) -> dict[str, dict[str, int]]:
    """Run (or count, with ``dry_run``) all three maintenance operations.

    Each operation runs in its own committed transaction, so one failing part
    never rolls back the others.

    Args:
        settings: Application settings (the ``retention`` section).
        dry_run: Report what would happen without deleting anything.

    Returns:
        A mapping from operation name to its row counts.
    """
    from mynotes.embed.client import get_embedder

    report: dict[str, dict[str, int]] = {}
    with session_scope() as session:
        report["purge_tombstones"] = purge_tombstones(session, settings, dry_run=dry_run)
    with session_scope() as session:
        report["clean_caches"] = clean_caches(session, settings, get_embedder(), dry_run=dry_run)
    with session_scope() as session:
        report["prune_ops"] = prune_ops(session, settings, dry_run=dry_run)
    with session_scope() as session:
        report["merge_entity_variants"] = merge_entity_variants(session, settings, dry_run=dry_run)
    with session_scope() as session:
        report["fold_date_entities"] = fold_date_entities_step(session, settings, dry_run=dry_run)
    report["link_lexical_mentions"] = link_lexical_mentions(settings, dry_run=dry_run)
    return report


def merge_entity_variants(
    session: Session, settings: Settings, *, dry_run: bool = False
) -> dict[str, int]:
    """Merge KG entities that differ only by articles/type words ("X plugin" -> "X")."""
    from mynotes.db.models import Workspace
    from mynotes.graph.store import merge_name_variants

    workspace_id = session.execute(
        select(Workspace.id).where(Workspace.name == settings.workspace)
    ).scalar_one_or_none()
    if workspace_id is None:
        return {"entities_merged": 0}
    pairs = merge_name_variants(session, int(workspace_id), dry_run=dry_run)
    return {"entities_merged": len(pairs)}


def fold_date_entities_step(
    session: Session, settings: Settings, *, dry_run: bool = False
) -> dict[str, int]:
    """Fold bare-date entities ("2023-05-20") into attribute values of their sources."""
    from mynotes.db.models import Workspace
    from mynotes.graph.dates import fold_date_entities

    workspace_id = session.execute(
        select(Workspace.id).where(Workspace.name == settings.workspace)
    ).scalar_one_or_none()
    if workspace_id is None:
        return {"entities": 0}
    return fold_date_entities(session, int(workspace_id), dry_run=dry_run).as_dict()


def link_lexical_mentions(settings: Settings, *, dry_run: bool = False) -> dict[str, int]:
    """Rebuild lexical entity mentions (names of known entities found verbatim in chunks)."""
    from mynotes.db.models import Workspace
    from mynotes.graph.lexical import LexicalParams, relink_workspace

    params = LexicalParams.from_settings(settings)
    if params is None:
        return {"versions": 0, "mentions_removed": 0, "mentions_added": 0}
    with session_scope() as session:
        workspace_id = session.execute(
            select(Workspace.id).where(Workspace.name == settings.workspace)
        ).scalar_one_or_none()
    if workspace_id is None:
        return {"versions": 0, "mentions_removed": 0, "mentions_added": 0}
    return relink_workspace(int(workspace_id), params, dry_run=dry_run)


def enqueue_maintenance_jobs(session: Session) -> list[int | None]:
    """Enqueue the three maintenance jobs at MAINTENANCE priority (daily dedupe)."""
    from mynotes.jobs.queue import enqueue

    return [
        enqueue(
            session,
            job_type,
            {},
            dedupe_key=daily_dedupe_key(job_type),
            priority=Priority.MAINTENANCE,
        )
        for job_type in MAINTENANCE_JOB_TYPES
    ]
