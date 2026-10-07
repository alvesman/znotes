"""Successor detection between documents (PLAN.md §7.5).

For one document D, :func:`detect_for_document` combines three signals into
``supersedes`` relations on ``(src=newer, dst=older, type='supersedes')``:

1. the move+edit merge: a rename with an edit looks like a MISSING/DELETED
   file plus a NEW file in the same folder; with exactly one such candidate at
   similarity >= 0.85 the two documents are merged into one identity (the new
   version moves under the old document) instead of creating a relation;
2. explicit frontmatter (``supersedes``, ``superseded_by``, ``status:
   obsolete|archived|superseded``) — always wins, confidence 1.0, status
   ``confirmed``;
3. name/content succession among current documents: same normalized stem (or
   same normalized title) inside the same folder, across a child/sibling
   archive folder, or within a shared ``project:*`` tag, confirmed by MinHash
   Jaccard and the cosine of the mean chunk embeddings, ordered by version
   number, then doc_date, then mtime.

Relations are upserted without ever overwriting a ``confirmed`` or
``rejected`` decision (user decisions win) and are never downgraded from
``auto`` to ``proposed``. The effects of active relations (status changes,
reindex, KG evidence) live in :mod:`mynotes.versioning.chains`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from sqlalchemy import ColumnElement, or_, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from mynotes.config import Settings
from mynotes.db.models import (
    Chunk,
    DocRelation,
    Document,
    DocumentTag,
    DocumentVersion,
    File,
    Tag,
)
from mynotes.embed.cache import get_cached
from mynotes.index.documents import indexable_chunk_ids
from mynotes.index.outbox import enqueue_chunk_deletes
from mynotes.jobs.queue import enqueue
from mynotes.jobs.registry import Priority
from mynotes.platform.paths import normalize_path_key
from mynotes.scan.hashing import hash_text
from mynotes.text.normalize import normalize_key
from mynotes.versioning.similarity import cosine, jaccard, mean_embedding, pair_score
from mynotes.versioning.stems import parse_stem

logger = logging.getLogger(__name__)

_MERGE_SCORE_THRESHOLD = 0.85
_OBSOLETE_STATUSES = frozenset({"obsolete", "archived", "superseded"})
_RELATION_RANK = {"proposed": 0, "auto": 1, "confirmed": 2}
_FINAL_STATUSES = frozenset({"confirmed", "rejected"})

_DETECT_LOCK_NAMESPACE = 7340
"""First key of the per-workspace advisory lock that serialises version detection."""


@dataclass
class Candidate:
    """One scored (older, newer) document pair.

    Attributes:
        older_id: The document the relation points at (``dst``).
        newer_id: The document the relation comes from (``src``); for an
            ambiguous order this is the document being processed.
        score: Combined similarity in [0, 1] (:func:`pair_score`).
        jaccard: The shingle Jaccard of the two documents' texts.
        cosine: Cosine of the two documents' mean chunk embeddings.
        name_signal: True when both files share the same normalized stem.
        archive_move: True when one of the two lives under an archive folder.
        order_reason: Which signal decided which document is newer.
    """

    older_id: int
    newer_id: int
    score: float
    jaccard: float
    cosine: float
    name_signal: bool
    archive_move: bool
    order_reason: Literal["version_number", "doc_date", "mtime", "ambiguous"]


@dataclass(frozen=True)
class SideInfo:
    """The ordering signals of one side of a candidate pair."""

    document_id: int
    version_number: int | None
    doc_date: datetime | None
    mtime_ns: int | None


def pick_newer(
    first: SideInfo, second: SideInfo
) -> tuple[SideInfo | None, Literal["version_number", "doc_date", "mtime", "ambiguous"]]:
    """Decide which of two document sides is the newer one.

    A higher version number wins first, then the later doc_date, then the
    later file mtime; a missing value counts as the oldest. When no signal
    distinguishes the sides the order is ambiguous (None returned).

    Args:
        first: One side.
        second: The other side.

    Returns:
        The newer side and the reason, or ``(None, "ambiguous")``.
    """
    if first.version_number is not None or second.version_number is not None:
        left = first.version_number if first.version_number is not None else -1
        right = second.version_number if second.version_number is not None else -1
        if left != right:
            return (first, "version_number") if left > right else (second, "version_number")
    if first.doc_date is not None or second.doc_date is not None:
        left = first.doc_date or datetime.min.replace(tzinfo=UTC)
        right = second.doc_date or datetime.min.replace(tzinfo=UTC)
        if left != right:
            return (first, "doc_date") if left > right else (second, "doc_date")
    if first.mtime_ns is not None or second.mtime_ns is not None:
        left = first.mtime_ns if first.mtime_ns is not None else -1
        right = second.mtime_ns if second.mtime_ns is not None else -1
        if left != right:
            return (first, "mtime") if left > right else (second, "mtime")
    return None, "ambiguous"


def decide(c: Candidate, settings: Settings) -> Literal["auto", "proposed", "none"]:
    """Map a candidate's score to a relation decision.

    Args:
        c: The scored candidate.
        settings: Application settings (``versions.auto_threshold``,
            ``versions.propose_threshold``).

    Returns:
        ``auto`` when the score reaches the auto threshold with an
        unambiguous order, else ``proposed`` when it reaches the propose
        threshold (an ambiguous order is never ``auto``), else ``none``.
    """
    if c.order_reason != "ambiguous" and c.score >= settings.versions.auto_threshold:
        return "auto"
    if c.score >= settings.versions.propose_threshold:
        return "proposed"
    return "none"


def find_candidates(session: Session, document_id: int, settings: Settings) -> list[Candidate]:
    """Find and score name/content succession candidates for one document.

    The pool is restricted to current documents of the same workspace whose
    file sits in the same folder, in a child/sibling archive folder, or that
    share a ``project:*`` tag with the document; identical content hashes
    (scanner duplicates) never become candidates. Every surviving pair is
    scored with :func:`mynotes.versioning.similarity.pair_score`.

    Args:
        session: An open session/transaction.
        document_id: The document to detect successors for.
        settings: Application settings (archive folders, embedding model).

    Returns:
        Scored candidates; below-threshold pairs are included and filtered by
        :func:`decide`.
    """
    document = session.get(Document, document_id)
    if document is None or document.status == "deleted":
        return []
    d_file = (
        session.get(File, document.current_file_id)
        if document.current_file_id is not None
        else None
    )
    d_version = (
        session.get(DocumentVersion, document.latest_version_id)
        if document.latest_version_id is not None
        else None
    )
    if d_file is None or d_version is None:
        return []
    archive_names = {normalize_key(name) for name in settings.versions.archive_folders}
    d_stem = parse_stem(Path(d_file.rel_path).name)
    d_folder = _folder_of(d_file.rel_path)
    d_project_tags = _project_tag_ids(session, document.id)

    conditions = _folder_scope_conditions(d_folder, archive_names, File.rel_path)
    if d_project_tags:
        conditions.append(
            Document.id.in_(
                select(DocumentTag.document_id).where(DocumentTag.tag_id.in_(d_project_tags))
            )
        )
    if not conditions:
        return []
    pool = session.execute(
        select(Document, File, DocumentVersion)
        .join(File, File.id == Document.current_file_id)
        .join(DocumentVersion, DocumentVersion.id == Document.latest_version_id)
        .where(
            Document.workspace_id == document.workspace_id,
            Document.status == "current",
            Document.id != document.id,
            or_(*conditions),
        )
    ).all()
    sharing = _project_sharing_ids(session, [row[0].id for row in pool], d_project_tags)

    candidates: list[Candidate] = []
    for o_doc, o_file, o_version in pool:
        if o_version.content_hash == d_version.content_hash:
            continue
        o_stem = parse_stem(Path(o_file.rel_path).name)
        o_folder = _folder_of(o_file.rel_path)
        same_folder = o_folder == d_folder
        archive_pair = _archive_folders_related(d_folder, o_folder, archive_names)
        same_stem = bool(d_stem.stem) and o_stem.stem == d_stem.stem
        title_match = bool(document.title_norm) and o_doc.title_norm == document.title_norm
        share_project = o_doc.id in sharing
        if not (
            (same_stem and (same_folder or archive_pair or share_project))
            or (title_match and (same_folder or share_project))
        ):
            continue
        jac, cos = _content_signals(session, d_version, o_version, settings)
        archive_move = _in_archive(d_folder, archive_names) or _in_archive(o_folder, archive_names)
        d_side = SideInfo(document.id, d_stem.version_number, document.doc_date, d_version.mtime_ns)
        o_side = SideInfo(o_doc.id, o_stem.version_number, o_doc.doc_date, o_version.mtime_ns)
        newer, reason = pick_newer(d_side, o_side)
        if newer is None:
            newer_id, older_id = document.id, o_doc.id
        else:
            newer_id = newer.document_id
            older_id = o_side.document_id if newer_id == document.id else document.id
        candidates.append(
            Candidate(
                older_id=older_id,
                newer_id=newer_id,
                score=pair_score(jac, cos, name_signal=same_stem, archive_move=archive_move),
                jaccard=jac,
                cosine=cos,
                name_signal=same_stem,
                archive_move=archive_move,
                order_reason=reason,
            )
        )
    return candidates


def detect_for_document(session: Session, document_id: int, settings: Settings) -> list[int]:
    """Run the full detection for one document and upsert its relations.

    The move+edit merge runs first (it consumes the document instead of
    creating relations), then the explicit frontmatter signals, then the
    scored name/content candidates. The effects of active relations are
    applied by the ``version_detect`` job handler.

    Args:
        session: An open session/transaction; rows are written but not
            committed.
        document_id: The document to process.
        settings: Application settings.

    Returns:
        The ids of the created or updated ``supersedes`` relations.
    """
    document = session.get(Document, document_id)
    if document is None or document.status == "deleted":
        return []
    # One detection at a time per workspace (transaction-level advisory lock, released at
    # commit): the jobs of two documents of one family write relation rows and document
    # statuses in orders that no sorting fully controls (frontmatter targets, merges), and
    # the synthetic smoke test caught them deadlocking. Detection is sub-second database
    # work, so serialising it costs nothing visible.
    session.execute(
        text("SELECT pg_advisory_xact_lock(:namespace, :key)"),
        {"namespace": _DETECT_LOCK_NAMESPACE, "key": int(document.workspace_id)},
    )
    if _try_move_edit_merge(session, document, settings):
        return []
    touched = _explicit_relations(session, document, settings)
    # Fixed (newer, older) order: the jobs of two documents of one family write their
    # shared relations in the same order, so they cannot deadlock on each other's rows.
    candidates = sorted(
        find_candidates(session, document_id, settings), key=lambda c: (c.newer_id, c.older_id)
    )
    for candidate in candidates:
        verdict = decide(candidate, settings)
        if verdict == "none":
            continue
        relation_id = _upsert_supersedes(
            session,
            src=candidate.newer_id,
            dst=candidate.older_id,
            method="name_content",
            confidence=candidate.score,
            status=verdict,
            evidence={
                "jaccard": candidate.jaccard,
                "cosine": candidate.cosine,
                "name_signal": candidate.name_signal,
                "archive_move": candidate.archive_move,
                "order_reason": candidate.order_reason,
            },
        )
        if relation_id is not None:
            touched.append(relation_id)
    return touched


def _try_move_edit_merge(session: Session, document: Document, settings: Settings) -> bool:
    """Merge a rename-with-edit pair into one document identity.

    When ``document`` (the new file) has exactly one same-folder document
    whose file went missing or was deleted within ``delete_grace_seconds * 2``
    and the pair scores at least 0.85, the new document's latest version moves
    under the old document (next version number) and the new document is
    deleted. A zero grace period takes effect immediately, so the rename
    window cannot be time-bounded and any missing/deleted file in the folder
    is eligible.

    Args:
        session: An open session/transaction.
        document: The document of the new file.
        settings: Application settings (``scan.delete_grace_seconds``,
            embedding model).

    Returns:
        True when the identity merge happened.
    """
    d_file = (
        session.get(File, document.current_file_id)
        if document.current_file_id is not None
        else None
    )
    d_version = (
        session.get(DocumentVersion, document.latest_version_id)
        if document.latest_version_id is not None
        else None
    )
    if d_file is None or d_version is None:
        return False
    folder = _folder_of(d_file.rel_path)
    window = timedelta(seconds=2 * settings.scan.delete_grace_seconds)
    conditions: list[ColumnElement[bool]] = [File.status.in_(("missing", "deleted"))]
    if window > timedelta(0):
        cutoff = datetime.now(UTC) - window
        conditions.append(or_(File.missing_since >= cutoff, File.deleted_at >= cutoff))
    rows = session.execute(
        select(Document, File)
        .join(File, File.id == Document.current_file_id)
        .where(
            Document.workspace_id == document.workspace_id,
            Document.id != document.id,
            File.source_root_id == d_file.source_root_id,
            *conditions,
        )
    ).all()
    candidates = [(doc, file) for doc, file in rows if _folder_of(file.rel_path) == folder]
    if len(candidates) != 1:
        return False
    o_doc, o_file = candidates[0]
    o_version = (
        session.get(DocumentVersion, o_doc.latest_version_id)
        if o_doc.latest_version_id is not None
        else None
    )
    if o_version is None:
        return False
    d_stem = parse_stem(Path(d_file.rel_path).name)
    same_stem = bool(d_stem.stem) and parse_stem(Path(o_file.rel_path).name).stem == d_stem.stem
    jac, cos = _content_signals(session, d_version, o_version, settings)
    score = pair_score(jac, cos, name_signal=same_stem, archive_move=False)
    if score < _MERGE_SCORE_THRESHOLD:
        return False
    _merge_into(
        session,
        settings=settings,
        document=document,
        o_doc=o_doc,
        d_file=d_file,
        d_version=d_version,
        o_file=o_file,
    )
    logger.info(
        "version_detect: move+edit merge of document %d into %d"
        " (score %.3f, jaccard %.3f, cosine %.3f)",
        document.id,
        o_doc.id,
        score,
        jac,
        cos,
    )
    return True


def _merge_into(
    session: Session,
    *,
    settings: Settings,
    document: Document,
    o_doc: Document,
    d_file: File,
    d_version: DocumentVersion,
    o_file: File,
) -> None:
    """Move ``d_version`` under ``o_doc``, repoint the file and delete ``document``."""
    now = datetime.now(UTC)
    if o_file.id != d_file.id:
        session.execute(
            update(File).where(File.id == o_file.id).values(status="deleted", deleted_at=now)
        )
    o_latest = session.get(DocumentVersion, o_doc.latest_version_id)
    assert o_latest is not None
    o_latest.is_latest = False
    d_version.document_id = o_doc.id
    d_version.version_no = o_latest.version_no + 1
    d_version.is_latest = True
    session.execute(
        update(Chunk)
        .where(Chunk.document_version_id == d_version.id)
        .values(document_id=o_doc.id, is_latest=True)
    )
    session.execute(
        update(Chunk)
        .where(Chunk.document_id == o_doc.id, Chunk.document_version_id != d_version.id)
        .values(is_latest=False)
    )
    session.execute(update(File).where(File.id == d_file.id).values(document_id=o_doc.id))
    o_doc.current_file_id = d_file.id
    o_doc.latest_version_id = d_version.id
    o_doc.status = "current"
    o_doc.updated_at = now
    indexable = set(indexable_chunk_ids(session, o_doc.id, settings))
    stale_ids = [
        chunk_id
        for chunk_id in session.execute(
            select(Chunk.id).where(Chunk.document_id.in_([o_doc.id, document.id]))
        ).scalars()
        if chunk_id not in indexable
    ]
    enqueue_chunk_deletes(session, stale_ids)
    session.delete(document)
    session.flush()
    enqueue(
        session,
        "reindex_metadata",
        {"document_id": o_doc.id},
        dedupe_key=f"reindex_metadata:{o_doc.id}",
        priority=Priority.EMBED,
    )
    enqueue(
        session,
        "delete_document",
        {"document_id": document.id},
        dedupe_key=f"delete_document:{document.id}",
        priority=Priority.EMBED,
    )


def _explicit_relations(session: Session, document: Document, settings: Settings) -> list[int]:
    """Create the frontmatter-mandated relations of ``document`` (always wins).

    ``supersedes`` creates ``document supersedes target``; ``superseded_by``
    creates ``target supersedes document``; both resolve their value by rel
    path (relative to the document's folder, then to the root) and then by
    ``title_norm``, always as ``confirmed`` relations with confidence 1.0.
    ``status: obsolete|archived|superseded`` archives the document itself.

    Args:
        session: An open session/transaction.
        document: The document being processed.
        settings: Application settings (unused; kept for symmetry).

    Returns:
        The ids of the created or updated relations.
    """
    version = (
        session.get(DocumentVersion, document.latest_version_id)
        if document.latest_version_id is not None
        else None
    )
    frontmatter = version.frontmatter if version is not None else {}
    touched: list[int] = []
    status_value = normalize_key(str(frontmatter.get("status") or ""))
    if status_value in _OBSOLETE_STATUSES and document.status != "archived":
        document.status = "archived"
        document.updated_at = datetime.now(UTC)
        enqueue(
            session,
            "reindex_metadata",
            {"document_id": document.id},
            dedupe_key=f"reindex_metadata:{document.id}",
            priority=Priority.EMBED,
        )
    for value in _as_list(frontmatter.get("supersedes")):
        target = _resolve_reference(session, document, value)
        if target is None:
            continue
        relation_id = _upsert_supersedes(
            session,
            src=document.id,
            dst=target.id,
            method="frontmatter",
            confidence=1.0,
            status="confirmed",
            evidence={"frontmatter": "supersedes", "value": str(value)},
        )
        if relation_id is not None:
            touched.append(relation_id)
    for value in _as_list(frontmatter.get("superseded_by")):
        target = _resolve_reference(session, document, value)
        if target is None:
            continue
        relation_id = _upsert_supersedes(
            session,
            src=target.id,
            dst=document.id,
            method="frontmatter",
            confidence=1.0,
            status="confirmed",
            evidence={"frontmatter": "superseded_by", "value": str(value)},
        )
        if relation_id is not None:
            touched.append(relation_id)
    return touched


def _resolve_reference(session: Session, document: Document, value: object) -> Document | None:
    """Resolve a frontmatter reference to a document, or None.

    The value is matched as a rel path relative to the document's folder and
    then to the source root (exact or accent/case-folded), and finally as the
    normalized title of a non-deleted document in the workspace.
    """
    reference = str(value).strip().strip("/")
    if not reference:
        return None
    d_file = (
        session.get(File, document.current_file_id)
        if document.current_file_id is not None
        else None
    )
    if d_file is not None:
        folder = _folder_of(d_file.rel_path)
        relative = f"{folder}/{reference}" if folder else reference
        for candidate_path in (relative, reference):
            target_file = (
                session.execute(
                    select(File)
                    .where(
                        File.source_root_id == d_file.source_root_id,
                        File.status != "deleted",
                        or_(
                            File.rel_path == candidate_path,
                            File.rel_path_norm == normalize_path_key(candidate_path),
                        ),
                    )
                    .order_by(File.id)
                )
                .scalars()
                .first()
            )
            if target_file is not None and target_file.document_id is not None:
                target = session.get(Document, target_file.document_id)
                if target is not None and target.status != "deleted" and target.id != document.id:
                    return target
    norm = normalize_key(reference)
    if norm:
        target = (
            session.execute(
                select(Document)
                .where(
                    Document.workspace_id == document.workspace_id,
                    Document.title_norm == norm,
                    Document.status != "deleted",
                )
                .order_by(Document.id)
            )
            .scalars()
            .first()
        )
        if target is not None and target.id != document.id:
            return target
    logger.info(
        "version_detect: frontmatter reference %r of document %d did not resolve",
        reference,
        document.id,
    )
    return None


def _upsert_supersedes(
    session: Session,
    *,
    src: int,
    dst: int,
    method: str,
    confidence: float,
    status: str,
    evidence: dict,
) -> int | None:
    """Upsert the ``supersedes`` relation (src=newer, dst=older).

    Existing ``confirmed``/``rejected`` relations are never touched (user
    decisions win) and an existing ``auto`` relation is never downgraded to
    ``proposed``. The two documents of a pair are detected by two separate
    ``version_detect`` jobs that may run at the same time, so the insert runs
    in a savepoint: when the other job wins the race (unique violation), the
    row it wrote is re-read and updated under the same rules.

    Args:
        session: An open session/transaction.
        src: The newer document id.
        dst: The older document id.
        method: ``frontmatter`` or ``name_content``.
        confidence: The relation confidence (the score, or 1.0 explicit).
        status: ``auto``, ``proposed`` or ``confirmed``.
        evidence: JSON evidence stored on the relation.

    Returns:
        The relation id, or None when nothing was written.
    """
    if src == dst:
        return None

    def find_existing() -> DocRelation | None:
        return session.execute(
            select(DocRelation).where(
                DocRelation.src_document_id == src,
                DocRelation.dst_document_id == dst,
                DocRelation.type == "supersedes",
            )
        ).scalar_one_or_none()

    def update_existing(existing: DocRelation) -> int | None:
        if existing.status in _FINAL_STATUSES:
            return None
        if _RELATION_RANK[status] < _RELATION_RANK[existing.status]:
            return None
        existing.method = method
        existing.confidence = confidence
        existing.status = status
        existing.evidence = evidence
        existing.updated_at = datetime.now(UTC)
        return existing.id

    existing = find_existing()
    if existing is not None:
        return update_existing(existing)
    relation = DocRelation(
        src_document_id=src,
        dst_document_id=dst,
        type="supersedes",
        method=method,
        confidence=confidence,
        status=status,
        evidence=evidence,
    )
    try:
        with session.begin_nested():
            session.add(relation)
            session.flush()
    except IntegrityError:
        logger.debug("supersedes %d->%d inserted concurrently; updating it", src, dst)
        session.expire_all()
        existing = find_existing()
        return update_existing(existing) if existing is not None else None
    return relation.id


def _content_signals(
    session: Session, d_version: DocumentVersion, o_version: DocumentVersion, settings: Settings
) -> tuple[float, float]:
    """Return (jaccard, cosine) for two document versions.

    Texts are the plain chunk texts of the versions; vectors are the mean of
    the cached chunk embeddings (0.0 cosine when either side has none yet).
    """
    model = settings.models.embedding.model
    prefix = settings.models.embedding.passage_prefix
    d_text, d_mean = _text_and_vector(session, d_version.id, model, prefix)
    o_text, o_mean = _text_and_vector(session, o_version.id, model, prefix)
    jac = jaccard(d_text, o_text)
    cos = cosine(d_mean, o_mean) if d_mean is not None and o_mean is not None else 0.0
    return jac, cos


def _text_and_vector(
    session: Session, version_id: int, model: str, prefix: str
) -> tuple[str, list[float] | None]:
    """Return a version's plain text and the mean of its cached embeddings."""
    rows = session.execute(
        select(Chunk.text, Chunk.embed_text)
        .where(Chunk.document_version_id == version_id)
        .order_by(Chunk.ordinal)
    ).all()
    text = " ".join(row[0] for row in rows)
    keys = [hash_text(prefix + embed_text) for _, embed_text in rows]
    cached = get_cached(session, model, keys)
    vectors = [cached[key] for key in keys if key in cached]
    return text, (mean_embedding(vectors) if vectors else None)


def _project_tag_ids(session: Session, document_id: int) -> list[int]:
    """Return the ids of the document's ``project:*`` tags."""
    return list(
        session.execute(
            select(DocumentTag.tag_id)
            .join(Tag, Tag.id == DocumentTag.tag_id)
            .where(DocumentTag.document_id == document_id, Tag.name.like("project:%"))
        ).scalars()
    )


def _project_sharing_ids(
    session: Session, pool_ids: list[int], project_tag_ids: list[int]
) -> set[int]:
    """Return the pool documents that share one of the given project tags."""
    if not project_tag_ids or not pool_ids:
        return set()
    return set(
        session.execute(
            select(DocumentTag.document_id).where(
                DocumentTag.document_id.in_(pool_ids),
                DocumentTag.tag_id.in_(project_tag_ids),
            )
        ).scalars()
    )


def _folder_of(rel_path: str) -> str:
    """Return the containing folder of a relative path ("" for the root)."""
    return rel_path.rsplit("/", 1)[0] if "/" in rel_path else ""


def _in_archive(folder: str, archive_names: set[str]) -> bool:
    """Return True when any folder segment is an archive folder."""
    return folder != "" and any(part in archive_names for part in folder.split("/"))


def _folder_scope_conditions(
    folder: str, archive_names: set[str], rel_path_column
) -> list[ColumnElement[bool]]:
    """Build the SQL ``rel_path`` conditions for the candidate scope.

    The conditions accept files in the folder itself, inside child archive
    folders, and inside sibling archive folders. When the document lives in a
    top-level archive folder the candidate may sit in any other top-level
    folder, so an always-true condition is returned.
    """
    conditions: list[ColumnElement[bool]] = []
    if folder:
        conditions.append(rel_path_column.like(folder + "/%"))
        parent = folder.rsplit("/", 1)[0]
    else:
        conditions.append(rel_path_column.notlike("%/%"))
        parent = ""
    for name in sorted(archive_names):
        prefix = f"{folder}/{name}" if folder else name
        conditions.append(rel_path_column.like(prefix + "/%"))
        sibling = f"{parent}/{name}" if parent else name
        if sibling != prefix:
            conditions.append(rel_path_column.like(sibling + "/%"))
    parts = folder.split("/") if folder else []
    for index, name in enumerate(parts):
        if name in archive_names:
            ancestor_parent = "/".join(parts[:index])
            if ancestor_parent:
                conditions.append(rel_path_column.like(ancestor_parent + "/%"))
            else:
                conditions.append(rel_path_column.is_not(None))
            break
    return conditions


def _archive_folders_related(folder_a: str, folder_b: str, archive_names: set[str]) -> bool:
    """Return True when one folder sits in an archive folder that is a child
    or a sibling of the other folder."""
    if folder_a == folder_b:
        return False
    for doc_folder, other in ((folder_a, folder_b), (folder_b, folder_a)):
        parts = doc_folder.split("/") if doc_folder else []
        for index, name in enumerate(parts):
            if name not in archive_names:
                continue
            archive_folder = "/".join(parts[: index + 1])
            if other == "/".join(parts[:index]):
                return True
            other_parent = other.rsplit("/", 1)[0] if "/" in other else ""
            if "/".join(parts[:index]) == other_parent and archive_folder != other:
                return True
    return False


def _as_list(value: object) -> list[str]:
    """Flatten a frontmatter value into non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_as_list(item))
        return flattened
    return [str(value)]
