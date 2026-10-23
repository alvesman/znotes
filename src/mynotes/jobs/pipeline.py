"""Ingestion pipeline handlers ``extract``, ``chunk``, ``embed``, ``tag_llm``,
``link_resolve``, ``kg_extract``, ``resolve_entities``, ``adjudicate_entities``,
``reindex_metadata``, ``delete_document``, ``kg_temporal_refresh`` and
``refresh_summaries`` (PLAN.md §7.7).

``extract`` turns a pending document version into stored IR and document
metadata; ``chunk`` persists the chunker's drafts and hands over to embedding;
``embed`` fills the Postgres embedding cache and queues the outbox rows that
put the chunks into OpenSearch; ``tag_llm`` asks the extract LLM for topic tag
proposals; ``link_resolve`` turns the version's Markdown links and wikilinks
into ``links_to`` relations (task 39); ``kg_extract`` runs the incremental LLM
knowledge-graph extraction over the version's section groups (task 40) and
hands the per-group results to ``resolve_entities`` (task 41), which turns the
cached extractions into canonical entities, mentions and relations and queues
``adjudicate_entities`` for borderline pairs; ``kg_temporal_refresh``
re-derives relation validity from the evidence currency that supersession,
deletion and re-chunking changed, resolves attribute supersession and queues
debounced ``refresh_summaries`` entity cards (task 42);
``reindex_metadata`` and ``delete_document`` queue metadata upserts and index
deletes the scanner asks for after moves, status changes and deletions.
A small follow-up registry lets later tasks (34, 38, 40) add jobs that run
after every successful ``chunk`` without editing this module: they register a
:class:`FollowUp` at import time and the worker imports this module.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from mynotes.chunk.chunker import ChunkDraft, ChunkingParams, chunk_document
from mynotes.chunk.diff import diff_chunks
from mynotes.chunk.tokenizer import get_token_counter
from mynotes.config import Settings
from mynotes.db.models import (
    Chunk,
    DocRelation,
    Document,
    DocumentTag,
    DocumentVersion,
    File,
    KgRelation,
    KgRelationEvidence,
    SourceRoot,
    Tag,
)
from mynotes.db.session import session_scope
from mynotes.embed.cache import ensure_embeddings
from mynotes.extract.ir import IRDocument
from mynotes.extract.registry import get_extractor
from mynotes.graph.extract import ChunkGroup, extract_group, extractable_groups, should_skip
from mynotes.graph.links import resolve_document_links
from mynotes.graph.resolve import (
    MAX_ADJUDICATE_PAIRS_PER_JOB,
    adjudicate_pairs,
    resolve_group,
)
from mynotes.graph.summaries import enqueue_refresh_summaries, refresh_entity_summaries
from mynotes.graph.temporal import (
    current_attribute_values,
    enqueue_temporal_refresh,
    evidence_relation_ids,
    lock_relations,
    refresh_document_relations,
)
from mynotes.index.documents import indexable_chunk_ids
from mynotes.index.outbox import enqueue_chunk_deletes, enqueue_chunk_upserts
from mynotes.index.reconcile import reconcile
from mynotes.jobs.queue import enqueue
from mynotes.jobs.registry import JobContext, Priority, RetryLater, handler
from mynotes.llm.chat import get_chat
from mynotes.llm.limits import get_guard
from mynotes.platform.paths import from_rel_path
from mynotes.scan.hashing import hash_file, hash_text
from mynotes.tags.autotag import (
    TagProposal,
    cut_excerpt,
    propose_tags,
    seconds_until_window_opens,
    tag_vocabulary,
)
from mynotes.tags.folder_rules import folder_tags
from mynotes.tags.manual import frontmatter_tags, hashtag_tags
from mynotes.tags.normalize import normalize_tag
from mynotes.tags.store import accepted_tags, ensure_tag, resolve_alias, set_document_tags
from mynotes.text.normalize import normalize_key
from mynotes.versioning.chains import apply_supersedes_effects, revert_supersedes_effects
from mynotes.versioning.detector import detect_for_document
from mynotes.versioning.retention import versions_to_purge

logger = logging.getLogger(__name__)

_MAX_ERROR_LENGTH = 2_000
_EXTRACTABLE_STATUSES = frozenset({"pending", "failed"})
_CHUNKABLE_STATUSES = frozenset({"extracted", "chunked"})
_EMBEDDABLE_STATUSES = frozenset({"chunked", "embedded"})
_TAGGABLE_STATUSES = frozenset({"chunked", "embedded"})


@dataclass(frozen=True)
class FollowUp:
    """A job type enqueued after every successful ``chunk`` run.

    Attributes:
        job_type: The job type to enqueue (e.g. ``"version_detect"``).
        priority: Claim priority; use :class:`mynotes.jobs.registry.Priority`.
        enabled: Called with the settings; the follow-up is enqueued only when
            it returns True.
        payload: Called with ``(document_id, version_id)`` and must return the
            job payload.
        dedupe: Called with ``(document_id, version_id)`` and must return the
            dedupe key; defaults to ``<job_type>:<version_id>``.
    """

    job_type: str
    priority: int
    enabled: Callable[[Settings], bool] = lambda settings: True
    payload: Callable[[int, int], dict] = lambda document_id, version_id: {"version_id": version_id}
    dedupe: Callable[[int, int], str] | None = None


_FOLLOW_UPS: list[FollowUp] = []


def register_follow_up(f: FollowUp) -> None:
    """Register a follow-up job type to enqueue after every ``chunk`` run.

    Args:
        f: The follow-up definition; registrations cannot be removed.
    """
    _FOLLOW_UPS.append(f)
    logger.debug("registered follow-up job type %r", f.job_type)


register_follow_up(
    FollowUp(
        job_type="version_detect",
        priority=Priority.VERSION_DETECT,
        payload=lambda document_id, version_id: {"document_id": document_id},
        dedupe=lambda document_id, version_id: f"version_detect:{document_id}",
    )
)
register_follow_up(
    FollowUp("tag_llm", Priority.TAG_LLM, enabled=lambda settings: settings.tags.llm_autotag)
)
register_follow_up(
    FollowUp(
        job_type="link_resolve",
        priority=Priority.VERSION_DETECT,
        payload=lambda document_id, version_id: {"document_id": document_id},
        dedupe=lambda document_id, version_id: f"link_resolve:{document_id}",
    )
)
register_follow_up(FollowUp("kg_extract", Priority.KG_EXTRACT))


@handler("extract")
def handle_extract(payload: dict[str, Any], context: JobContext) -> None:
    """Extract one document version into stored IR and enqueue ``chunk``.

    Quietly skips (idempotent no-op): a missing version, a superseded or
    already-processed version, a vanished file and a file whose current
    content hash differs from the version's (the scanner creates the right
    version on its next pass). Extractor errors mark the version ``failed``
    and end the job; ``OSError`` propagates so the queue retries.

    Args:
        payload: ``{"version_id": int}``.
        context: The worker's job context (unused settings).
    """
    version_id = payload["version_id"]
    with session_scope() as session:
        version = session.get(DocumentVersion, version_id)
        if version is None:
            logger.warning("extract job for missing version %s; skipping", version_id)
            return
        document = session.get(Document, version.document_id)
        if document is None:
            logger.warning("extract job for version %s without a document; skipping", version_id)
            return
        if not version.is_latest or version.status not in _EXTRACTABLE_STATUSES:
            return
        file_row = (
            session.get(File, document.current_file_id)
            if document.current_file_id is not None
            else None
        )
        root = session.get(SourceRoot, document.source_root_id)
        if file_row is None or root is None:
            logger.warning(
                "extract of version %s: document %s has no current file or root; skipping",
                version.id,
                document.id,
            )
            return
        path = version_source_path(root, file_row.rel_path, version.content_hash, document.format)
        if path is None or not path.exists():
            logger.info(
                "extract of version %s: %s is gone; waiting for the next scan or upload",
                version.id,
                path or file_row.rel_path,
            )
            return
        if hash_file(path) != version.content_hash:
            logger.info(
                "extract of version %s: %s changed after the scan; skipping", version.id, path
            )
            return
        try:
            extractor = get_extractor(document.format)
            ir = extractor.extract(path)
        except OSError:
            raise
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_LENGTH]
            logger.warning("extract of version %s failed: %s", version.id, message)
            version.status = "failed"
            version.error = message
            return
        _store_ir(session, document, version, ir)


def version_source_path(
    root: SourceRoot, rel_path: str, content_hash: str | None, fmt: str
) -> Path | None:
    """Where the bytes of a version are: the local folder, or the blob store for remote roots.

    Returns None for a remote root whose blob store is not configured or
    whose version has no hash yet.
    """
    if root.kind == "remote":
        if content_hash is None:
            return None
        from mynotes.storage.blobs import StorageNotConfigured, get_blob_store

        try:
            store = get_blob_store()
        except StorageNotConfigured as exc:
            logger.warning("%s", exc)
            return None
        return store.path_for(int(root.workspace_id), int(root.id), content_hash, fmt)
    if root.abs_path is None:
        return None
    return from_rel_path(Path(root.abs_path), rel_path)


def _store_ir(
    session: Session, document: Document, version: DocumentVersion, ir: IRDocument
) -> None:
    """Persist the extraction result and enqueue ``chunk`` for extracted versions."""
    ir_json = ir.to_json()
    version.ir_json = ir_json
    version.frontmatter = ir_json["frontmatter"]
    version.extractor = ir.extractor
    version.extractor_version = ir.extractor_version
    version.doc_date = ir.doc_date
    version.status = "empty" if not ir.blocks else "extracted"
    version.error = "needs_ocr" if "needs_ocr" in ir.warnings else None
    document.title = ir.title
    document.title_norm = normalize_key(ir.title) if ir.title else None
    document.language = ir.language
    document.doc_date = ir.doc_date
    document.updated_at = datetime.now(UTC)
    if version.status == "extracted":
        enqueue(
            session,
            "chunk",
            {"version_id": version.id},
            dedupe_key=f"chunk:{version.id}",
            priority=Priority.CHUNK,
        )


@handler("chunk")
def handle_chunk(payload: dict[str, Any], context: JobContext) -> None:
    """Chunk one extracted document version, persist the chunks and enqueue ``embed``.

    Idempotent: existing chunk rows of the version are deleted first, so a
    re-run rebuilds them. The manual tag sources (folder rules from the file's
    current path, frontmatter, hashtags) are persisted as accepted
    ``document_tag`` rows (task 37) and the accepted set (aliases applied) is
    what goes into the chunks' ``embed_text``. After the chunk rows are
    written, ``embed`` and every enabled registered follow-up are enqueued with
    a per-version dedupe key.

    Args:
        payload: ``{"version_id": int}``.
        context: The worker's job context; provides the chunking settings.
    """
    version_id = payload["version_id"]
    settings = context.settings
    with session_scope() as session:
        version = session.get(DocumentVersion, version_id)
        if version is None:
            logger.warning("chunk job for missing version %s; skipping", version_id)
            return
        if version.status not in _CHUNKABLE_STATUSES:
            return
        document = session.get(Document, version.document_id)
        if document is None or version.ir_json is None:
            logger.warning("chunk of version %s: no document or no stored IR; skipping", version_id)
            return
        file_row = (
            session.get(File, document.current_file_id)
            if document.current_file_id is not None
            else None
        )
        if file_row is None:
            logger.warning("chunk of version %s: document has no current file", version_id)
            return
        ir = IRDocument.from_json(version.ir_json)
        _persist_manual_tags(session, document, version, ir, file_row, settings)
        tags = accepted_tags(session, document.id)
        page_kind = (
            "slide" if document.format == "pptx" else "page" if document.format == "pdf" else None
        )
        params = ChunkingParams(
            target_tokens=settings.chunking.target_tokens,
            max_tokens=settings.chunking.max_tokens,
            overlap_tokens=settings.chunking.overlap_tokens,
        )
        drafts = chunk_document(
            ir,
            rel_path=file_row.rel_path,
            tags=tags,
            params=params,
            counter=get_token_counter(),
            page_kind=page_kind,
        )
        _log_chunk_diff(session, document.id, version.version_no, drafts)
        _replace_chunks(session, document, version, drafts)
        version.status = "chunked"
        if version.is_latest and _demote_old_version_evidence(session, document.id, version.id):
            enqueue_temporal_refresh(session, document.id)
        enqueue(
            session,
            "embed",
            {"version_id": version.id},
            dedupe_key=f"embed:{version.id}",
            priority=Priority.EMBED,
        )
        for follow_up in _FOLLOW_UPS:
            if follow_up.enabled(settings):
                dedupe_key = (
                    follow_up.dedupe(document.id, version.id)
                    if follow_up.dedupe is not None
                    else f"{follow_up.job_type}:{version.id}"
                )
                enqueue(
                    session,
                    follow_up.job_type,
                    follow_up.payload(document.id, version.id),
                    dedupe_key=dedupe_key,
                    priority=follow_up.priority,
                )


def _persist_manual_tags(
    session: Session,
    document: Document,
    version: DocumentVersion,
    ir: IRDocument,
    file_row: File,
    settings: Settings,
) -> None:
    """Persist the manual tag sources of a version (folder, frontmatter, hashtag).

    The folder tags come from the file's current path, so a re-run after a move
    rewrites them. The accepted set (aliases applied) is what the chunker puts
    into ``embed_text`` and the index reads.
    """
    set_document_tags(
        session,
        document.id,
        version.id,
        {
            "folder": folder_tags(
                file_row.rel_path, settings.tags, settings.versions.archive_folders
            ),
            "frontmatter": frontmatter_tags(ir.frontmatter),
            "hashtag": hashtag_tags(ir.hashtags),
        },
    )


@handler("embed")
def handle_embed(payload: dict[str, Any], context: JobContext) -> None:
    """Embed one chunked version through the Postgres cache and queue outbox rows.

    Idempotent: texts already in the embedding cache are never sent to the
    embedder again, and a re-run appends the outbox rows once more (harmless;
    the syncer collapses them). Versions whose status is not ``chunked`` or
    ``embedded`` end the job quietly; embedding errors (``EndpointError``,
    ``EndpointUnavailable``) propagate so the queue and the worker retry them.
    The upsert/delete split is exactly :func:`indexable_chunk_ids` (task 33's
    final rule), so the index ends up holding the indexable chunks of the
    document and nothing else. After a new latest version is embedded, an
    ``apply_retention`` job trims the version history.
    The vectors themselves are not stored on the chunks: the outbox syncer
    reads them back from the cache by :func:`embed_key`.

    Args:
        payload: ``{"version_id": int}``.
        context: The worker's job context; provides the embedder and the
            ``versions`` settings.
    """
    version_id = payload["version_id"]
    with session_scope() as session:
        version = session.get(DocumentVersion, version_id)
        if version is None:
            logger.warning("embed job for missing version %s; skipping", version_id)
            return
        if version.status not in _EMBEDDABLE_STATUSES:
            return
        chunks = list(
            session.execute(
                select(Chunk).where(Chunk.document_version_id == version.id).order_by(Chunk.ordinal)
            ).scalars()
        )
        ensure_embeddings(session, context.embedder, [chunk.embed_text for chunk in chunks])
        version.status = "embedded"
        if not version.is_latest:
            logger.info("embed of version %s: superseded meanwhile; index untouched", version.id)
            return
        upsert_ids = indexable_chunk_ids(session, version.document_id, context.settings)
        indexable = set(upsert_ids)
        document_chunk_ids = list(
            session.execute(
                select(Chunk.id).where(Chunk.document_id == version.document_id)
            ).scalars()
        )
        delete_ids = [chunk_id for chunk_id in document_chunk_ids if chunk_id not in indexable]
        upserts = enqueue_chunk_upserts(session, upsert_ids)
        deletes = enqueue_chunk_deletes(session, delete_ids)
        enqueue(
            session,
            "apply_retention",
            {"document_id": version.document_id},
            dedupe_key=f"apply_retention:{version.document_id}",
            priority=Priority.MAINTENANCE,
        )
        logger.info(
            "embed of version %s: %d chunk(s), %d outbox upsert(s), %d outbox delete(s)",
            version.id,
            len(chunks),
            upserts,
            deletes,
        )


@handler("apply_retention")
def handle_apply_retention(payload: dict[str, Any], context: JobContext) -> None:
    """Purge a document's versions that retention no longer keeps (task 33).

    Enqueued by the ``embed`` handler after a new latest version is embedded.
    The purge list comes from :func:`versions_to_purge`; the purged versions'
    chunks first get outbox deletes (in case they were indexed), then the
    version rows are deleted (chunks, KG mentions and relation evidence
    cascade in the database). ``embedding_cache`` rows are never deleted here;
    the periodic cache cleanup is task 50.

    Args:
        payload: ``{"document_id": int}``.
        context: The worker's job context; provides the retention settings.
    """
    document_id = payload["document_id"]
    with session_scope() as session:
        versions = list(
            session.execute(
                select(DocumentVersion)
                .where(DocumentVersion.document_id == document_id)
                .order_by(DocumentVersion.version_no)
            ).scalars()
        )
        purge_ids = versions_to_purge(versions, datetime.now(UTC), context.settings)
        if not purge_ids:
            return
        chunk_ids = list(
            session.execute(
                select(Chunk.id).where(Chunk.document_version_id.in_(purge_ids)).order_by(Chunk.id)
            ).scalars()
        )
        deletes = enqueue_chunk_deletes(session, chunk_ids)
        session.execute(delete(DocumentVersion).where(DocumentVersion.id.in_(purge_ids)))
    logger.info(
        "apply_retention of document %s: purged %d version(s), %d chunk delete(s)",
        document_id,
        len(purge_ids),
        deletes,
    )


@handler("version_detect")
def handle_version_detect(payload: dict[str, Any], context: JobContext) -> None:
    """Detect successor documents for one document and apply their effects.

    Runs the task-34 detection (move+edit merge, frontmatter signals and
    name/content candidates) and applies the effects of every active
    (``auto``/``confirmed``) supersedes relation it created or updated: the
    older document becomes ``superseded``, points at its chain head, its
    indexed chunks get a metadata reindex and its KG evidence becomes
    non-current. Quietly skips a missing or deleted document.

    Args:
        payload: ``{"document_id": int}``.
        context: The worker's job context; provides the settings.
    """
    document_id = payload["document_id"]
    relation_ids: list[int] = []
    with session_scope() as session:
        relation_ids = detect_for_document(session, document_id, context.settings)
        for relation_id in relation_ids:
            relation = session.get(DocRelation, relation_id)
            if relation is None or relation.type != "supersedes":
                continue
            if relation.status in ("auto", "confirmed"):
                apply_supersedes_effects(session, relation.dst_document_id)
    logger.info(
        "version_detect of document %d: %d relation(s) created or updated",
        document_id,
        len(relation_ids),
    )


@handler("link_resolve")
def handle_link_resolve(payload: dict[str, Any], context: JobContext) -> None:
    """Resolve one document's Markdown links and wikilinks into ``links_to`` (task 39).

    Runs the task-39 resolution over the latest version's stored IR against
    the workspace document index and upserts ``doc_relation(src, dst,
    type='links_to', method='link_parse', confidence=1.0, status='auto')``
    rows with link-count evidence; relations whose link no longer resolves
    are deleted. Enqueued after every ``chunk`` and nightly by the
    ``link-sweep`` (04:15), which is what resolves links to documents created
    later. Quietly skips a missing or deleted document.

    Args:
        payload: ``{"document_id": int}``.
        context: The worker's job context (unused settings).
    """
    document_id = payload["document_id"]
    with session_scope() as session:
        document = session.get(Document, document_id)
        if document is None or document.status == "deleted":
            return
        upserted, removed = resolve_document_links(session, document)
    logger.info(
        "link_resolve of document %s: %d relation(s) upserted, %d removed",
        document_id,
        upserted,
        removed,
    )


@handler("kg_extract")
def handle_kg_extract(payload: dict[str, Any], context: JobContext) -> None:
    """Extract the knowledge graph of one latest embedded version (task 40).

    The version's chunks are grouped into section groups; every non-skipped
    group is extracted through :func:`mynotes.graph.extract.extract_group`,
    which serves results from ``kg_extraction_cache`` whenever the group text
    (plus the extractor version) was extracted before, so a re-run never calls
    the model again. The entities found so far are carried into the next
    group's prompt as known entities. Afterwards a ``resolve_entities`` job
    (task 41) receives the per-group list; the extraction results themselves
    stay in the cache, keyed by content hash.
    Outside the configured ``background_window`` the job is re-queued
    (``RetryLater``) without consuming an attempt, like ``tag_llm``; a latest
    version that is still ``chunked`` (its ``embed`` job has not finished) is
    also re-queued briefly instead of being skipped, so the race between the
    two follow-ups cannot lose the extraction.

    Args:
        payload: ``{"version_id": int}``.
        context: The worker's job context (unused settings).
    """
    version_id = payload["version_id"]
    extracted: list[ChunkGroup] = []
    with session_scope() as session:
        version = session.get(DocumentVersion, version_id)
        if version is None:
            logger.warning("kg_extract job for missing version %s; skipping", version_id)
            return
        if not version.is_latest:
            return
        if version.status == "chunked":
            raise RetryLater(delay_s=30.0, reason="waiting for embed")
        if version.status != "embedded":
            return
        document = session.get(Document, version.document_id)
        if document is None or document.status == "deleted":
            return
        guard = get_guard("llm")
        if not guard.background_allowed_now():
            window = guard.limits.background_window
            assert window is not None, "background_allowed_now() is False without a window"
            raise RetryLater(
                delay_s=seconds_until_window_opens(window, datetime.now().astimezone()),
                reason="outside background_window",
            )
        file_row = (
            session.get(File, document.current_file_id)
            if document.current_file_id is not None
            else None
        )
        ctx: dict[str, Any] = {
            "title": document.title,
            "rel_path": file_row.rel_path if file_row is not None else version.rel_path_at_version,
            "doc_date": version.doc_date,
            "tags": accepted_tags(session, document.id),
            "known_entities": [],
        }
        chat = get_chat("extract")
        for group in extractable_groups(session, version_id):
            extraction = extract_group(session, chat, group, ctx, should_skip(group))
            ctx["known_entities"].extend(
                (entity.name, entity.type) for entity in extraction.entities
            )
            extracted.append(group)
        if extracted:
            enqueue(
                session,
                "resolve_entities",
                {
                    "version_id": version_id,
                    "groups": [
                        {
                            "chunk_ids": [str(chunk_id) for chunk_id in group.chunk_ids],
                            "content_hash": group.content_hash,
                        }
                        for group in extracted
                    ],
                },
                dedupe_key=f"resolve_entities:{version_id}",
                priority=Priority.RESOLVE_ENTITIES,
            )
    logger.info("kg_extract of version %s: %d group(s) extracted", version_id, len(extracted))


@handler("resolve_entities")
def handle_resolve_entities(payload: dict[str, Any], context: JobContext) -> None:
    """Turn cached extractions into canonical graph rows (task 41).

    Every group of the payload is processed in its own transaction:
    :func:`mynotes.graph.resolve.resolve_group` loads the cached extraction,
    deletes the group's previous mentions and relation evidence (idempotency),
    links or creates entities (exact name/alias, then trigram, then k-NN),
    inserts mentions and relation evidence and refreshes the touched entities'
    current mention counts. Missing or error cache rows are skipped, so a
    group the extractor never completed costs nothing. Borderline pairs are
    collected across groups and enqueued as ``adjudicate_entities`` jobs (up
    to 20 pairs per job).

    Args:
        payload: ``{"version_id": int, "groups": [{"chunk_ids": [str],
            "content_hash": str}]}`` as enqueued by ``kg_extract``.
        context: The worker's job context; provides the embedder used to cache
            new entities' vectors.
    """
    version_id = payload["version_id"]
    pending_pairs: list[tuple[int, int]] = []
    document_id: int | None = None
    for group in payload["groups"]:
        with session_scope() as session:
            version = session.get(DocumentVersion, version_id)
            if version is None:
                logger.warning("resolve_entities job for missing version %s; skipping", version_id)
                return
            document = session.get(Document, version.document_id)
            if document is None or document.status == "deleted":
                return
            document_id = int(document.id)
            pending_pairs.extend(
                resolve_group(
                    session,
                    version=version,
                    document=document,
                    group=group,
                    embedder=context.embedder,
                )
            )
    if document_id is not None:
        with session_scope() as session:
            enqueue_temporal_refresh(session, document_id)
    if not pending_pairs:
        return
    enqueued = 0
    with session_scope() as session:
        for start in range(0, len(pending_pairs), MAX_ADJUDICATE_PAIRS_PER_JOB):
            batch = pending_pairs[start : start + MAX_ADJUDICATE_PAIRS_PER_JOB]
            fingerprint = hash_text(json.dumps(sorted(batch)))
            if (
                enqueue(
                    session,
                    "adjudicate_entities",
                    {"pairs": [[new_id, candidate_id] for new_id, candidate_id in batch]},
                    dedupe_key=f"adjudicate_entities:{fingerprint}",
                    priority=Priority.RESOLVE_ENTITIES,
                )
                is not None
            ):
                enqueued += 1
    logger.info(
        "resolve_entities of version %s: %d adjudication job(s) enqueued", version_id, enqueued
    )


@handler("adjudicate_entities")
def handle_adjudicate_entities(payload: dict[str, Any], context: JobContext) -> None:
    """LLM-adjudicate borderline entity pairs and merge or record them (task 41).

    One background ``complete_structured`` call asks, per pair, whether both
    records are the same entity. ``same`` with confidence >= 0.8 merges the
    older entity into the newer one's counterpart via
    :func:`mynotes.graph.store.merge_entities`; every other answered pair is
    recorded in ``kg_entity_distinct`` so resolution never links or adjudicates
    it again. Outside the configured ``background_window`` the job is
    re-queued (``RetryLater``) without consuming an attempt, like ``tag_llm``.

    Args:
        payload: ``{"pairs": [[entity_a, entity_b], ...]}``.
        context: The worker's job context (unused settings).
    """
    pairs = [(int(pair[0]), int(pair[1])) for pair in payload["pairs"]]
    guard = get_guard("llm")
    if not guard.background_allowed_now():
        window = guard.limits.background_window
        assert window is not None, "background_allowed_now() is False without a window"
        raise RetryLater(
            delay_s=seconds_until_window_opens(window, datetime.now().astimezone()),
            reason="outside background_window",
        )
    decided = 0
    with session_scope() as session:
        decided = adjudicate_pairs(session, pairs, get_chat("extract"))
    logger.info("adjudicate_entities: %d/%d pair(s) decided", decided, len(pairs))


@handler("kg_temporal_refresh")
def handle_kg_temporal_refresh(payload: dict[str, Any], context: JobContext) -> None:
    """Re-derive relation validity from the evidence currency of one document (task 42).

    Every relation with evidence from the document's versions gets its
    ``status``/``valid_from``/``valid_to`` recomputed from
    :func:`mynotes.graph.temporal.relation_status`; then attribute
    supersession runs for the affected source entities (older attribute values
    become ``historical``) and a debounced ``refresh_summaries`` job is
    enqueued for the relations' endpoint entities. A deleted document is not
    skipped: its stale relations must reach ``stale`` through this job.

    Args:
        payload: ``{"document_id": int}``.
        context: The worker's job context (unused settings).
    """
    document_id = int(payload["document_id"])
    with session_scope() as session:
        evidence_ids = evidence_relation_ids(session, document_id)
        if not evidence_ids:
            return
        endpoint_ids: set[int] = set()
        source_ids: set[int] = set()
        for src_id, dst_id in session.execute(
            select(KgRelation.src_entity_id, KgRelation.dst_entity_id).where(
                KgRelation.id.in_(evidence_ids)
            )
        ).all():
            endpoint_ids.update((int(src_id), int(dst_id)))
            source_ids.add(int(src_id))
        # Lock everything this job may write (the evidenced relations and the source
        # entities' sibling relations that attribute supersession demotes) in id order,
        # so concurrent refresh jobs of other documents queue instead of deadlocking.
        sibling_ids = session.execute(
            select(KgRelation.id).where(KgRelation.src_entity_id.in_(sorted(source_ids)))
        ).scalars()
        lock_relations(session, [*evidence_ids, *sibling_ids])
        relation_ids = refresh_document_relations(session, document_id)
        for entity_id in sorted(source_ids):
            current_attribute_values(session, entity_id)
        enqueue_refresh_summaries(session, sorted(endpoint_ids))
    logger.info(
        "kg_temporal_refresh of document %d: %d relation(s) refreshed",
        document_id,
        len(relation_ids),
    )


@handler("refresh_summaries")
def handle_refresh_summaries(payload: dict[str, Any], context: JobContext) -> None:
    """Regenerate stale entity summaries ("entity cards") in a batch (task 42).

    Enqueued debounced (``run_after`` 10 minutes out) by
    ``kg_temporal_refresh``, so bursts of changes coalesce. Entities that are
    missing, merged, not stale or mentioned by fewer than two current
    documents are skipped without an LLM call; rewritten summaries queue an
    entity outbox upsert so the index picks them up as descriptions. Outside
    the configured ``background_window`` the job is re-queued (``RetryLater``)
    without consuming an attempt, like ``tag_llm``.

    Args:
        payload: ``{"entity_ids": [int]}``.
        context: The worker's job context (unused settings).
    """
    entity_ids = [int(entity_id) for entity_id in payload["entity_ids"]]
    guard = get_guard("llm")
    if not guard.background_allowed_now():
        window = guard.limits.background_window
        assert window is not None, "background_allowed_now() is False without a window"
        raise RetryLater(
            delay_s=seconds_until_window_opens(window, datetime.now().astimezone()),
            reason="outside background_window",
        )
    with session_scope() as session:
        rewritten = refresh_entity_summaries(session, entity_ids, get_chat("extract"))
    logger.info(
        "refresh_summaries: %d/%d entity summary(ies) rewritten", rewritten, len(entity_ids)
    )


@handler("tag_llm")
def handle_tag_llm(payload: dict[str, Any], context: JobContext) -> None:
    """Propose LLM tags for one document version (task 38).

    Runs once per latest version that is ``chunked``/``embedded`` and has no
    LLM tag rows yet. Outside the configured ``background_window`` the job is
    re-queued (``RetryLater``) without consuming an attempt. Every proposal is
    normalized and alias-resolved; names the document already has accepted are
    skipped. A proposal is auto-accepted when its confidence reaches
    ``tags.llm_auto_accept`` and the tag already exists in the workspace
    vocabulary; everything else stays ``proposed`` for ``mynotes tags review``.
    Proposed rows of previous versions are removed (user decisions persist).
    Auto-accepted tags enqueue a metadata reindex; LLM tags never join
    ``embed_text``, so no re-chunk is needed.

    Args:
        payload: ``{"version_id": int}``.
        context: The worker's job context; provides the settings.
    """
    version_id = payload["version_id"]
    settings = context.settings
    if not settings.tags.llm_autotag:
        return
    with session_scope() as session:
        version = session.get(DocumentVersion, version_id)
        if version is None:
            logger.warning("tag_llm job for missing version %s; skipping", version_id)
            return
        document = session.get(Document, version.document_id)
        if document is None or not version.is_latest or version.status not in _TAGGABLE_STATUSES:
            return
        guard = get_guard("llm")
        if not guard.background_allowed_now():
            window = guard.limits.background_window
            assert window is not None, "background_allowed_now() is False without a window"
            raise RetryLater(
                delay_s=seconds_until_window_opens(window, datetime.now().astimezone()),
                reason="outside background_window",
            )
        if _has_llm_rows(session, document.id, version.id):
            return
        if version.ir_json is None:
            logger.warning("tag_llm of version %s: no stored IR; skipping", version.id)
            return
        accepted = accepted_tags(session, document.id)
        ir = IRDocument.from_json(version.ir_json)
        file_row = (
            session.get(File, document.current_file_id)
            if document.current_file_id is not None
            else None
        )
        proposals = propose_tags(
            get_chat("extract"),
            title=document.title,
            rel_path=file_row.rel_path if file_row is not None else version.rel_path_at_version,
            headings=[block.text for block in ir.blocks if block.kind == "heading"],
            excerpt=cut_excerpt(ir.plain_text()),
            existing_tags=accepted,
            vocabulary=tag_vocabulary(session, document.workspace_id),
            language=document.language,
        )
        stored, auto_accepted = _store_proposals(
            session,
            document,
            version,
            proposals,
            set(accepted),
            settings.tags.llm_auto_accept,
        )
        dropped = _drop_stale_proposals(session, document.id, version.id)
        if auto_accepted:
            enqueue(
                session,
                "reindex_metadata",
                {"document_id": document.id},
                dedupe_key=f"reindex_metadata:{document.id}",
                priority=Priority.EMBED,
            )
    logger.info(
        "tag_llm of version %s: %d row(s) stored, %d auto-accepted, %d stale proposal(s) removed",
        version_id,
        stored,
        auto_accepted,
        dropped,
    )


def _has_llm_rows(session: Session, document_id: int, version_id: int) -> bool:
    """Whether ``source='llm'`` tag rows already exist for this version."""
    return bool(
        session.execute(
            select(func.count())
            .select_from(DocumentTag)
            .where(
                DocumentTag.document_id == document_id,
                DocumentTag.source == "llm",
                DocumentTag.document_version_id == version_id,
            )
        ).scalar_one()
    )


def _store_proposals(
    session: Session,
    document: Document,
    version: DocumentVersion,
    proposals: Sequence[TagProposal],
    accepted_names: set[str],
    auto_accept: float,
) -> tuple[int, int]:
    """Persist proposals as ``document_tag(source='llm')`` rows.

    A re-proposal over an existing ``proposed`` row refreshes it (confidence,
    reason, version, recomputed status); a ``rejected`` row is never touched.

    Returns:
        ``(stored, auto_accepted)``: the rows written and how many of those
        landed as ``accepted``.
    """
    workspace_id = document.workspace_id
    stored = 0
    auto_accepted = 0
    for proposal in proposals:
        name = resolve_alias(session, workspace_id, normalize_tag(proposal.tag))
        if not name or name in accepted_names:
            continue
        accepted_names.add(name)
        exists = session.execute(
            select(Tag.id).where(Tag.workspace_id == workspace_id, Tag.name == name)
        ).scalar_one_or_none()
        if exists is None:
            status = "proposed"
        else:
            status = "accepted" if proposal.confidence >= auto_accept else "proposed"
        tag_id = ensure_tag(session, workspace_id, name)
        result = session.execute(
            pg_insert(DocumentTag)
            .values(
                document_id=document.id,
                tag_id=tag_id,
                source="llm",
                confidence=proposal.confidence,
                status=status,
                reason=proposal.reason,
                document_version_id=version.id,
            )
            .on_conflict_do_update(
                index_elements=["document_id", "tag_id", "source"],
                set_={
                    "confidence": proposal.confidence,
                    "reason": proposal.reason,
                    "status": status,
                    "document_version_id": version.id,
                },
                where=DocumentTag.status == "proposed",
            )
        )
        if not result.rowcount:
            logger.info(
                "tag_llm: %r was already rejected for document %s; keeping the decision",
                name,
                document.id,
            )
            continue
        stored += 1
        if status == "accepted":
            auto_accepted += 1
    return stored, auto_accepted


def _drop_stale_proposals(session: Session, document_id: int, version_id: int) -> int:
    """Remove ``proposed`` LLM rows of the document's other versions."""
    result = session.execute(
        delete(DocumentTag).where(
            DocumentTag.document_id == document_id,
            DocumentTag.source == "llm",
            DocumentTag.status == "proposed",
            DocumentTag.document_version_id.is_distinct_from(version_id),
        )
    )
    return int(result.rowcount or 0)


@handler("reindex_metadata")
def handle_reindex_metadata(payload: dict[str, Any], context: JobContext) -> None:
    """Re-queue index upserts for a document's indexable chunks.

    Used after moves, status changes and tag changes: the syncer rebuilds the
    documents from Postgres with the cached embeddings, so nothing is
    re-embedded and only metadata fields (path, tags, status) change in the
    index. After a move the folder tags are recomputed first; when they
    changed, the latest version is re-chunked (``embed_text`` contains the
    tags), which re-embeds only the chunks whose ``embed_text`` changed.
    Quietly skips a missing or deleted document.

    Args:
        payload: ``{"document_id": int}``.
        context: The worker's job context; provides the settings.
    """
    document_id = payload["document_id"]
    with session_scope() as session:
        document = session.get(Document, document_id)
        if document is None or document.status == "deleted":
            return
        file_row = (
            session.get(File, document.current_file_id)
            if document.current_file_id is not None
            else None
        )
        if file_row is None:
            logger.warning("reindex_metadata of document %s: no current file", document_id)
            return
        retagged = _refresh_folder_tags(session, document, file_row, context.settings)
        queued = enqueue_chunk_upserts(
            session, indexable_chunk_ids(session, document_id, context.settings)
        )
        if retagged:
            _rechunk_latest_version(session, document)
    logger.info(
        "reindex_metadata of document %s: %d outbox upsert(s)%s",
        document_id,
        queued,
        "; folder tags changed, re-chunking" if retagged else "",
    )


def _refresh_folder_tags(
    session: Session, document: Document, file_row: File, settings: Settings
) -> bool:
    """Recompute the folder tags after a move; True when the accepted set changed."""
    return set_document_tags(
        session,
        document.id,
        document.latest_version_id,
        {
            "folder": folder_tags(
                file_row.rel_path, settings.tags, settings.versions.archive_folders
            )
        },
    )


def _rechunk_latest_version(session: Session, document: Document) -> None:
    """Re-chunk a document's latest version so ``embed_text`` picks up new tags."""
    if document.latest_version_id is None:
        return
    version = session.get(DocumentVersion, document.latest_version_id)
    if version is None or version.ir_json is None:
        return
    version.status = "chunked"
    enqueue(
        session,
        "chunk",
        {"version_id": version.id},
        dedupe_key=f"chunk:{version.id}",
        priority=Priority.CHUNK,
    )


@handler("delete_document")
def handle_delete_document(payload: dict[str, Any], context: JobContext) -> None:
    """Queue index deletes for every chunk of a deleted document.

    The chunk rows themselves stay in Postgres until the tombstone purge
    (task 50); only their OpenSearch documents are removed. The document's KG
    evidence becomes non-current (task 42) and, when it is the newer side of
    an active ``supersedes`` relation, that relation is deleted and the older
    document's effects are undone (status back to ``current``, its evidence
    current again, reindex queued) — then a ``kg_temporal_refresh`` job
    re-derives the affected relations' validity.

    Args:
        payload: ``{"document_id": int}``.
        context: The worker's job context (unused settings).
    """
    document_id = payload["document_id"]
    with session_scope() as session:
        chunk_ids = list(
            session.execute(select(Chunk.id).where(Chunk.document_id == document_id)).scalars()
        )
        queued = enqueue_chunk_deletes(session, chunk_ids)
        demoted = 0
        reverted = 0
        if chunk_ids:
            demoted = _demote_chunk_evidence(session, chunk_ids)
        successors = list(
            session.execute(
                select(DocRelation)
                .where(
                    DocRelation.src_document_id == document_id,
                    DocRelation.type == "supersedes",
                    DocRelation.status.in_(("auto", "confirmed")),
                )
                .order_by(DocRelation.id)
            ).scalars()
        )
        for relation in successors:
            session.delete(relation)
            revert_supersedes_effects(session, relation.dst_document_id)
            reverted += 1
        if demoted or reverted:
            enqueue_temporal_refresh(session, document_id)
    logger.info(
        "delete_document of document %s: %d outbox delete(s), %d evidence row(s) demoted, "
        "%d supersedes relation(s) undone",
        document_id,
        queued,
        demoted,
        reverted,
    )


def _demote_chunk_evidence(session: Session, chunk_ids: Sequence[Any]) -> int:
    """Mark the KG evidence of the given chunks non-current; returns the rows."""
    result = session.execute(
        update(KgRelationEvidence)
        .where(
            KgRelationEvidence.chunk_id.in_(chunk_ids),
            KgRelationEvidence.is_current.is_(True),
        )
        .values(is_current=False)
    )
    return int(result.rowcount or 0)


def _demote_old_version_evidence(session: Session, document_id: int, latest_version_id: int) -> int:
    """Mark the other versions' KG evidence non-current; returns the rows.

    Called by the ``chunk`` handler when a new version becomes the latest one:
    evidence attached to the document's older versions no longer backs current
    facts, so the ``kg_temporal_refresh`` job can demote their relations.
    """
    result = session.execute(
        update(KgRelationEvidence)
        .where(
            KgRelationEvidence.document_version_id.in_(
                select(DocumentVersion.id).where(
                    DocumentVersion.document_id == document_id,
                    DocumentVersion.id.is_distinct_from(latest_version_id),
                )
            ),
            KgRelationEvidence.is_current.is_(True),
        )
        .values(is_current=False)
    )
    return int(result.rowcount or 0)


@handler("reconcile")
def handle_reconcile(payload: dict[str, Any], context: JobContext) -> None:
    """Reconcile the chunks index with Postgres, repairing drift by default.

    Enqueued nightly by the scheduler (task 25) and runnable ad hoc through
    ``mynotes index reconcile``. The report is saved for ``mynotes status``.

    Args:
        payload: ``{"fix": bool}`` (optional; defaults to True). With
            ``fix=False`` the run only reports the drift it finds.
        context: The worker's job context (unused).
    """
    reconcile(fix=bool(payload.get("fix", True)))


def _log_chunk_diff(
    session: Session, document_id: int, version_no: int, drafts: list[ChunkDraft]
) -> None:
    """Log how the new chunks match the previous version's by content hash."""
    previous_id = session.execute(
        select(DocumentVersion.id)
        .where(
            DocumentVersion.document_id == document_id,
            DocumentVersion.version_no < version_no,
        )
        .order_by(DocumentVersion.version_no.desc())
        .limit(1)
    ).scalar_one_or_none()
    old_hashes: list[str] = []
    if previous_id is not None:
        old_hashes = list(
            session.execute(
                select(Chunk.content_hash)
                .where(Chunk.document_version_id == previous_id)
                .order_by(Chunk.ordinal)
            ).scalars()
        )
    diff = diff_chunks(old_hashes, drafts)
    logger.info(
        "chunk diff for document %s version %s: %d reused, %d added, %d removed",
        document_id,
        version_no,
        len(diff.reused),
        len(diff.added),
        len(diff.removed),
    )


def _replace_chunks(
    session: Session, document: Document, version: DocumentVersion, drafts: list[ChunkDraft]
) -> None:
    """Delete the version's chunk rows and insert the new drafts in their place.

    The replaced rows keep their old chunk ids, so their indexed documents (a
    coalesced edit re-chunks a version that was already indexed) are queued
    for deletion from OpenSearch first; a 404 delete counts as success.
    """
    stale_ids = list(
        session.execute(select(Chunk.id).where(Chunk.document_version_id == version.id)).scalars()
    )
    if stale_ids:
        enqueue_chunk_deletes(session, stale_ids)
    session.execute(delete(Chunk).where(Chunk.document_version_id == version.id))
    # Core executemany insert instead of one ORM object per chunk: a 20k-file
    # corpus creates roughly 300k chunk rows and the unit-of-work path showed
    # up as pure insert overhead in the task-51 measurements.
    if drafts:
        session.execute(
            insert(Chunk),
            [
                {
                    "document_version_id": version.id,
                    "document_id": document.id,
                    "ordinal": draft.ordinal,
                    "content_hash": draft.content_hash,
                    "text": draft.text,
                    "embed_text": draft.embed_text,
                    "page": draft.page,
                    "slide": draft.slide,
                    "section": draft.section,
                    "heading_path": draft.heading_path,
                    "is_table": draft.is_table,
                    "token_count": draft.token_count,
                    "language": draft.language,
                    "is_latest": version.is_latest,
                }
                for draft in drafts
            ],
        )
    if version.is_latest:
        session.execute(
            update(Chunk)
            .where(
                Chunk.document_id == document.id,
                Chunk.document_version_id != version.id,
            )
            .values(is_latest=False)
        )
