"""Drift reconciliation between Postgres and the chunks index (PLAN.md §7.6).

:func:`reconcile` compares the indexable chunks of non-deleted documents (the
same rule as :func:`mynotes.index.documents.indexable_chunk_ids`) with the
documents behind the chunks alias: ids that only exist in Postgres are
``missing``, ids that only exist in OpenSearch are ``extra``, and ids present
in both with a different ``meta_hash`` are ``stale``. With ``fix=True`` the
differences are re-enqueued through the outbox, so the regular syncer (task 22)
repairs them; nothing is written to OpenSearch directly.

The scheduler enqueues ``reconcile`` nightly (task 25); the CLI offers it as
``mynotes index reconcile``. Every run saves its :class:`DriftReport` as JSON
under :func:`platformdirs.user_state_dir` so ``mynotes status`` can show the
last result without a database migration.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import platformdirs
from opensearchpy import OpenSearch
from opensearchpy.helpers import scan as scan_helper
from sqlalchemy import select
from sqlalchemy.orm import Session

from mynotes.config import Settings, get_settings
from mynotes.db.models import Chunk, Document, DocumentVersion, SourceRoot
from mynotes.db.session import session_scope
from mynotes.index.documents import build_chunk_docs
from mynotes.index.opensearch import get_client
from mynotes.index.outbox import enqueue_chunk_deletes, enqueue_chunk_upserts

logger = logging.getLogger(__name__)

_REPORT_FILENAME = "drift_report.json"


@dataclass
class DriftReport:
    """Result of one :func:`reconcile` run.

    Attributes:
        expected: Indexable chunks in Postgres.
        indexed: Documents in OpenSearch (chunks alias).
        missing: In Postgres, not in OpenSearch.
        extra: In OpenSearch, not indexable in Postgres.
        stale: Present in both but with a different metadata hash.
    """

    expected: int
    indexed: int
    missing: list[str]
    extra: list[str]
    stale: list[str]


STATE_KEY = "drift_report"
"""The ``app_state`` key of the last report (the file is the fallback)."""


def report_path() -> Path:
    """Return the path of the saved drift report (user state dir)."""
    return Path(platformdirs.user_state_dir("mynotes")) / _REPORT_FILENAME


def save_drift_report(report: DriftReport) -> Path:
    """Write ``report`` as JSON so ``mynotes status`` can show it later.

    Args:
        report: The report of the finished run.

    Returns:
        The path the report was written to.
    """
    path = report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"saved_at": datetime.now(UTC).isoformat(), **asdict(report)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:  # shared with the web process through the database
        from mynotes.db.state import set_state

        set_state(STATE_KEY, payload)
    except Exception:
        logger.debug("drift report not written to app_state", exc_info=True)
    return path


def load_last_report() -> dict[str, Any] | None:
    """Return the last saved drift report, or None when there is none.

    A missing or unreadable report is not an error for ``mynotes status``;
    problems are logged and reported as no report.
    """
    try:
        from mynotes.db.state import get_state

        shared = get_state(STATE_KEY)
    except Exception:
        shared = None
    if isinstance(shared, dict):
        return shared
    try:
        return json.loads(report_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("could not read the drift report at %s", report_path(), exc_info=True)
        return None


def reconcile(fix: bool = True, batch: int = 1000) -> DriftReport:
    """Compare Postgres with the chunks alias and optionally queue repairs.

    Args:
        fix: When True, enqueue outbox upserts for ``missing`` and ``stale``
            chunk ids and outbox deletes for ``extra`` ids; the syncer applies
            them on its next pass. Chunks whose document cannot be built right
            now (no cached embedding, no current file) cannot be verified and
            are skipped; the rebuild covers them.
        batch: Number of chunks rendered per :func:`build_chunk_docs` call.

    Returns:
        The drift report of this run (also saved for ``mynotes status``).
    """
    settings = get_settings()
    client = get_client()
    alias = settings.opensearch.chunks_alias
    with session_scope() as session:
        expected = expected_chunk_ids(session, settings)
    indexed_ids, indexed_hashes = _indexed_state(client, alias)
    expected_ids = {str(chunk_id) for chunk_id in expected}
    missing = sorted(expected_ids - indexed_ids)
    extra = sorted(indexed_ids - expected_ids)
    stale = _stale_ids(settings, sorted(expected_ids & indexed_ids), indexed_hashes, batch)
    report = DriftReport(
        expected=len(expected_ids),
        indexed=len(indexed_ids),
        missing=missing,
        extra=extra,
        stale=stale,
    )
    if fix and (missing or extra or stale):
        with session_scope() as session:
            upserts = enqueue_chunk_upserts(session, [*missing, *stale])
            deletes = enqueue_chunk_deletes(session, extra)
        logger.info("reconcile fix: %d upsert(s) and %d delete(s) enqueued", upserts, deletes)
    logger.info(
        "reconcile: expected=%d indexed=%d missing=%d extra=%d stale=%d (fix=%s)",
        report.expected,
        report.indexed,
        len(report.missing),
        len(report.extra),
        len(report.stale),
        fix,
    )
    save_drift_report(report)
    return report


def expected_chunk_ids(
    session: Session, settings: Settings, *, source_root_id: int | None = None
) -> list[UUID]:
    """All indexable chunk ids across non-deleted documents (task 22's rule).

    Shared by :func:`reconcile` and the rebuild (task 23) so both always
    consider exactly the chunks the per-document rule of task 22 considers:
    documents of disabled source roots are excluded.

    Args:
        session: An open session/transaction.
        settings: Application settings (``versions.index_old_versions``).
        source_root_id: Only chunks of this source root, when given.

    Returns:
        The chunk ids ordered by id.
    """
    stmt = (
        select(Chunk.id)
        .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
        .join(Document, DocumentVersion.document_id == Document.id)
        .join(SourceRoot, Document.source_root_id == SourceRoot.id)
        .where(Document.status != "deleted", SourceRoot.enabled.is_(True))
        .order_by(Chunk.id)
    )
    if not settings.versions.index_old_versions:
        stmt = stmt.where(DocumentVersion.is_latest.is_(True))
    if source_root_id is not None:
        stmt = stmt.where(Document.source_root_id == source_root_id)
    return list(session.execute(stmt).scalars())


def root_chunk_ids(session: Session, source_root_id: int) -> list[UUID]:
    """Every chunk id of one source root's documents, any version or status.

    Args:
        session: An open session/transaction.
        source_root_id: The source root.

    Returns:
        The chunk ids ordered by id.
    """
    stmt = (
        select(Chunk.id)
        .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
        .join(Document, DocumentVersion.document_id == Document.id)
        .where(Document.source_root_id == source_root_id)
        .order_by(Chunk.id)
    )
    return list(session.execute(stmt).scalars())


def _indexed_state(client: OpenSearch, alias: str) -> tuple[set[str], dict[str, str | None]]:
    """Scroll the alias and return the doc ids plus their ``meta_hash`` values."""
    ids: set[str] = set()
    hashes: dict[str, str | None] = {}
    for doc in scan_helper(
        client,
        index=alias,
        query={"query": {"match_all": {}}, "_source": ["chunk_id", "meta_hash"]},
    ):
        doc_id = str(doc["_id"])
        ids.add(doc_id)
        hashes[doc_id] = (doc.get("_source") or {}).get("meta_hash")
    return ids, hashes


def _stale_ids(
    settings: Settings, common: list[str], indexed_hashes: dict[str, str | None], batch: int
) -> list[str]:
    """Common ids whose fresh Postgres metadata hash differs from the index."""
    model = settings.models.embedding.model
    passage_prefix = settings.models.embedding.passage_prefix
    stale: list[str] = []
    with session_scope() as session:
        for start in range(0, len(common), batch):
            chunk_ids = [UUID(value) for value in common[start : start + batch]]
            fresh = build_chunk_docs(session, chunk_ids, model, passage_prefix)
            for chunk_id in common[start : start + batch]:
                doc = fresh.get(chunk_id)
                if doc is not None and doc["meta_hash"] != indexed_hashes.get(chunk_id):
                    stale.append(chunk_id)
    return stale
