"""Outbox syncer: drain ``index_outbox`` into OpenSearch (PLAN.md §5.5, §7.6).

Every pass claims pending rows with ``FOR UPDATE SKIP LOCKED``, collapses them
by document (the highest row id decides the operation), sends one ``_bulk``
request against the configured aliases and marks the successful groups
processed. Operations are idempotent: an upsert indexes the full document by
id and a delete of a missing document counts as success, so a crash between
the bulk request and the bookkeeping update only repeats work.

Entity documents (``doc_kind='entity'``) are rendered with
:data:`ENTITY_DOC_BUILDER` (:func:`mynotes.index.documents.build_entity_docs`,
installed below, task 41); when it is unset, entity rows are left unprocessed.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk as bulk_helper
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from mynotes.config import Settings, get_settings
from mynotes.db.models import IndexOutbox
from mynotes.db.session import session_scope
from mynotes.index.documents import build_chunk_docs, build_entity_docs
from mynotes.index.opensearch import get_client

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 10
_MAX_ERROR_LENGTH = 2_000
_ERROR_BACKOFF_S = 30.0

# One _bulk request carries at most this many documents: 4096-dim float vectors
# serialize to roughly 40 KB per chunk document, so 200 docs keep every request
# near 8 MB and comfortably below the ~20 MB request budget (task 51).
_BULK_REQUEST_DOCS = 200

_CHUNKS_INDEX = "chunks"
_ENTITIES_INDEX = "entities"

EntityDocBuilder = Callable[[Session, Sequence[str]], dict[str, dict[str, Any] | None]]

ENTITY_DOC_BUILDER: EntityDocBuilder | None = build_entity_docs

_entity_builder_warned = False


@dataclass
class SyncStats:
    """Totals of one :func:`sync_once` pass.

    Attributes:
        processed: Outbox rows marked processed (the successful groups).
        upserts: Upsert actions that reached the index.
        deletes: Delete actions that succeeded (a 404 delete counts).
        failures: Actions that failed; their rows got ``attempts += 1``.
    """

    processed: int = 0
    upserts: int = 0
    deletes: int = 0
    failures: int = 0


@dataclass
class _Group:
    """Outbox rows collapsed to one target document."""

    index_name: str
    doc_kind: str
    doc_id: str
    op: str
    row_ids: list[int] = field(default_factory=list)


@dataclass
class _Planned:
    """One group's prepared bulk action, or a failure decided before sending."""

    group: _Group
    action: dict[str, Any] | None = None
    error: str | None = None


def sync_once(batch_size: int = 500) -> SyncStats:
    """Claim pending outbox rows and push them to OpenSearch.

    Args:
        batch_size: Maximum number of rows to claim per pass.

    Returns:
        The pass totals. Failed groups are not marked processed; their rows
        get ``attempts += 1`` and ``last_error`` instead.
    """
    settings = get_settings()
    aliases = {
        _CHUNKS_INDEX: settings.opensearch.chunks_alias,
        _ENTITIES_INDEX: settings.opensearch.entities_alias,
    }
    stats = SyncStats()
    with session_scope() as session:
        rows = _claim_rows(session, batch_size)
        if not rows:
            return stats
        plans = _plan(session, _collapse(rows), aliases, settings)
        actions = [plan.action for plan in plans if plan.action is not None]
        failed = _send(get_client(), actions) if actions else {}
        _apply(session, plans, failed, stats)
    if stats.failures:
        logger.warning(
            "outbox sync: %d processed, %d upsert(s), %d delete(s), %d failure(s)",
            stats.processed,
            stats.upserts,
            stats.deletes,
            stats.failures,
        )
    elif stats.processed:
        logger.info(
            "outbox sync: %d processed, %d upsert(s), %d delete(s)",
            stats.processed,
            stats.upserts,
            stats.deletes,
        )
    return stats


def sync_loop(stop: threading.Event, idle_sleep_s: float = 2.0) -> None:
    """Drain the outbox until ``stop`` is set (worker background thread).

    Sleeps ``idle_sleep_s`` after a pass that processed nothing and backs off
    for 30 s after an unexpected error (OpenSearch down).

    Args:
        stop: The worker's stop event; the loop returns soon after it is set.
        idle_sleep_s: Sleep when a pass processed no rows.
    """
    logger.info("outbox sync loop started")
    while not stop.is_set():
        try:
            stats = sync_once()
        except Exception:
            logger.exception("outbox sync failed; backing off for %.0fs", _ERROR_BACKOFF_S)
            if stop.wait(_ERROR_BACKOFF_S):
                break
            continue
        if stats.processed == 0 and stop.wait(idle_sleep_s):
            break
    logger.info("outbox sync loop stopped")


def _claim_rows(session: Session, batch_size: int) -> list[IndexOutbox]:
    """Lock up to ``batch_size`` pending rows, skipping rows other workers hold."""
    return list(
        session.execute(
            select(IndexOutbox)
            .where(IndexOutbox.processed_at.is_(None), IndexOutbox.attempts < _MAX_ATTEMPTS)
            .order_by(IndexOutbox.id)
            .limit(batch_size)
            .with_for_update(skip_locked=True)
        ).scalars()
    )


def _collapse(rows: Iterable[IndexOutbox]) -> list[_Group]:
    """Collapse rows by (index_name, doc_id); the highest row id decides the op."""
    groups: dict[tuple[str, str], _Group] = {}
    for row in sorted(rows, key=lambda item: item.id):
        group = groups.setdefault(
            (row.index_name, row.doc_id),
            _Group(row.index_name, row.doc_kind, row.doc_id, row.op),
        )
        group.doc_kind = row.doc_kind
        group.op = row.op
        group.row_ids.append(row.id)
    return list(groups.values())


def _plan(
    session: Session,
    groups: list[_Group],
    aliases: dict[str, str],
    settings: Settings,
) -> list[_Planned]:
    """Turn collapsed groups into bulk actions.

    All chunk upserts are rendered with one :func:`build_chunk_docs` call and
    all entity upserts with one :data:`ENTITY_DOC_BUILDER` call; a None
    document (gone from Postgres, deleted, merged, or not embedded yet) turns
    the op into a delete.
    """
    chunk_upsert_groups = [
        group
        for group in groups
        if group.doc_kind == "chunk" and group.op == "upsert" and group.index_name in aliases
    ]
    chunk_docs = build_chunk_docs(
        session,
        _uuid_or_none(group.doc_id for group in chunk_upsert_groups),
        settings.models.embedding.model,
        settings.models.embedding.passage_prefix,
    )
    entity_upsert_groups = [
        group
        for group in groups
        if group.doc_kind == "entity" and group.op == "upsert" and group.index_name in aliases
    ]
    entity_docs = _entity_docs(session, entity_upsert_groups)
    plans: list[_Planned] = []
    for group in groups:
        alias = aliases.get(group.index_name)
        if alias is None:
            plans.append(_Planned(group, error=f"unknown index_name {group.index_name!r}"))
            continue
        if group.doc_kind == "entity":
            plans.append(_plan_entity(group, alias, entity_docs))
            continue
        if group.op == "delete":
            plans.append(_Planned(group, action=_delete_action(alias, group.doc_id)))
            continue
        chunk_id = _as_uuid(group.doc_id)
        if chunk_id is None:
            plans.append(_Planned(group, error=f"chunk id {group.doc_id!r} is not a UUID"))
            continue
        source = chunk_docs.get(str(chunk_id))
        if source is None:
            # The syncer cannot index this chunk right now, so the index must
            # not keep it either; the delete of a missing doc is a success.
            plans.append(_Planned(group, action=_delete_action(alias, group.doc_id)))
            continue
        plans.append(
            _Planned(
                group,
                action={
                    "_op_type": "index",
                    "_index": alias,
                    "_id": group.doc_id,
                    "_source": source,
                },
            )
        )
    return plans


def _entity_docs(session: Session, groups: list[_Group]) -> dict[str, dict[str, Any] | None] | None:
    """Render the entity upsert groups' documents with the builder hook.

    Returns None when the hook is unset (rows are then left unprocessed, like
    before task 41) and an empty mapping when there is nothing to build.
    """
    global _entity_builder_warned
    if not groups:
        return {}
    if ENTITY_DOC_BUILDER is None:
        if not _entity_builder_warned:
            logger.warning("entity outbox rows pending: no ENTITY_DOC_BUILDER registered yet")
            _entity_builder_warned = True
        return None
    ids: list[str] = []
    for group in groups:
        try:
            ids.append(str(int(group.doc_id)))
        except ValueError:
            logger.warning("entity outbox doc id %r is not an entity id", group.doc_id)
    return ENTITY_DOC_BUILDER(session, ids)


def _plan_entity(
    group: _Group, alias: str, docs: dict[str, dict[str, Any] | None] | None
) -> _Planned:
    """Plan one entity group from the pre-built documents."""
    if docs is None:
        return _Planned(group)
    doc = docs.get(group.doc_id)
    if doc is None:
        return _Planned(group, action=_delete_action(alias, group.doc_id))
    return _Planned(
        group,
        action={"_op_type": "index", "_index": alias, "_id": group.doc_id, "_source": doc},
    )


def _uuid_or_none(values: Iterable[str]) -> list[UUID]:
    """Parse ids, dropping values that are not UUIDs (those fail in :func:`_plan`)."""
    parsed = []
    for value in values:
        chunk_id = _as_uuid(value)
        if chunk_id is not None:
            parsed.append(chunk_id)
    return parsed


def _as_uuid(value: str) -> UUID | None:
    """Parse a chunk id, or None when it is not a UUID."""
    try:
        return UUID(value)
    except ValueError:
        return None


def _delete_action(alias: str, doc_id: str) -> dict[str, Any]:
    """Build a bulk delete action against the alias."""
    return {"_op_type": "delete", "_index": alias, "_id": doc_id}


def _send(client: OpenSearch, actions: list[dict[str, Any]]) -> dict[str, str]:
    """Send one bulk request and map failed doc ids to error messages.

    Args:
        client: The OpenSearch client.
        actions: One action per group.

    Returns:
        ``{doc_id: error}`` for every failed action; a 404 on a delete counts
        as success and is not reported.
    """
    _, errors = bulk_helper(
        client,
        actions,
        raise_on_error=False,
        raise_on_exception=False,
        chunk_size=_BULK_REQUEST_DOCS,
    )
    failed: dict[str, str] = {}
    for item in errors:
        op_type, info = next(iter(item.items()))
        if op_type == "delete" and info.get("status") == 404:
            continue
        doc_id = info.get("_id")
        message = _error_message(info)
        if doc_id is None:
            logger.warning("bulk failure without a document id: %s", message)
            continue
        failed[str(doc_id)] = message
    return failed


def _error_message(info: dict[str, Any]) -> str:
    """Extract a short error string from one bulk failure item."""
    error = info.get("error")
    if isinstance(error, dict):
        return f"{error.get('type', 'error')}: {error.get('reason', error)}"[:_MAX_ERROR_LENGTH]
    text = str(error or info.get("exception") or "bulk request failed")
    return text[:_MAX_ERROR_LENGTH]


def _apply(
    session: Session,
    plans: list[_Planned],
    failed: dict[str, str],
    stats: SyncStats,
) -> None:
    """Mark successful groups processed and failed groups for retry."""
    now = datetime.now(UTC)
    for plan in plans:
        error = plan.error
        if error is None and plan.action is not None:
            error = failed.get(str(plan.action["_id"]))
        if error is not None:
            session.execute(
                update(IndexOutbox)
                .where(IndexOutbox.id.in_(plan.group.row_ids))
                .values(attempts=IndexOutbox.attempts + 1, last_error=error)
            )
            stats.failures += 1
            logger.warning(
                "outbox %s %s/%s failed: %s",
                plan.group.op,
                plan.group.index_name,
                plan.group.doc_id,
                error,
            )
            continue
        session.execute(
            update(IndexOutbox)
            .where(IndexOutbox.id.in_(plan.group.row_ids))
            .values(processed_at=now, last_error=None)
        )
        stats.processed += len(plan.group.row_ids)
        if plan.action is not None and plan.action["_op_type"] == "index":
            stats.upserts += 1
        else:
            stats.deletes += 1
