"""Deterministic link resolution for the structural graph (PLAN.md §8.1).

Markdown links and wikilinks collected by the extractors (task 17) are
resolved against the document index of the workspace and stored as active
``links_to`` relations (``method='link_parse'``, confidence 1.0). The
``link_resolve`` handler runs after every ``chunk`` and nightly (the 04:15
sweep), so a link written before its target document existed resolves on the
next sweep. Relations whose link disappeared from the source are deleted.
"""

from __future__ import annotations

import logging
import posixpath
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import unquote

from sqlalchemy import and_, delete, select, text
from sqlalchemy.orm import Session

from mynotes.db.models import DocRelation, Document, DocumentVersion, File
from mynotes.extract.ir import IRDocument, Link
from mynotes.text.normalize import normalize_key, normalize_path_key

logger = logging.getLogger(__name__)

_EVIDENCE_TEXT_LIMIT = 5
_FINAL_STATUSES = frozenset({"confirmed", "rejected"})


@dataclass(frozen=True)
class DocIndexEntry:
    """One resolvable link target in the workspace.

    Attributes:
        document_id: The document the file belongs to.
        rel_path: The current file path, POSIX separators.
        folder: The parent folder of ``rel_path`` (``""`` for top-level files).
        stem_norm: The folded file stem (``normalize_key`` of the file name
            without its last extension).
        title_norm: The folded document title, or None when untitled.
        path_norm: The folded ``rel_path``; empty for hand-built entries, in
            which case :func:`resolve_link` computes it on demand.
    """

    document_id: int
    rel_path: str
    folder: str
    stem_norm: str
    title_norm: str | None
    path_norm: str = ""


def file_stem(rel_path: str) -> str:
    """Return the file name of ``rel_path`` without its last extension.

    Args:
        rel_path: A relative path with ``/`` separators.

    Returns:
        The stem (``"notes/idea.md"`` → ``"idea"``); hidden dotfiles keep
        their whole name.
    """
    name = posixpath.basename(rel_path)
    dot = name.rfind(".")
    return name[:dot] if dot > 0 else name


def wikilink_stem(target: str) -> str:
    """Return the stem of a wikilink target: its basename, ``.md`` dropped.

    Args:
        target: A wikilink target that may include folders and may end in
            ``.md`` (``"a/b/Plano.md"`` → ``"Plano"``).

    Returns:
        The basename without a trailing ``.md`` (any case).
    """
    base = posixpath.basename(target)
    return base[:-3] if base.casefold().endswith(".md") else base


def resolve_link(link: Link, source_rel_path: str, index: Sequence[DocIndexEntry]) -> int | None:
    """Resolve one extracted link to a document id, or None.

    Markdown links are resolved relative to the source's folder with POSIX
    semantics and matched by exact ``rel_path``, then case/accent-insensitively.
    Wikilinks are matched by exact ``rel_path`` (with and without ``.md``),
    then by folded file stem (preferring the same folder, then the shortest
    path distance, then the most recent document), then by folded title.

    Args:
        link: The link as written in the source (kind ``url``, ``markdown``
            or ``wikilink``).
        source_rel_path: The source document's file path, POSIX separators.
        index: The resolvable documents of the workspace (see
            :func:`build_index`).

    Returns:
        The target document id, or None for external URLs, unresolvable
        targets and links pointing at the document itself.
    """
    if link.kind == "url":
        return None
    if link.kind == "markdown":
        return _resolve_markdown(link.target, source_rel_path, index)
    if link.kind == "wikilink":
        return _resolve_wikilink(link.target, source_rel_path, index)
    return None


def _resolve_markdown(
    target: str, source_rel_path: str, index: Sequence[DocIndexEntry]
) -> int | None:
    """Resolve a Markdown link target against the index."""
    resolved = _markdown_target(target, source_rel_path)
    if resolved is None:
        return None
    for entry in index:
        if entry.rel_path == resolved:
            return _target_id(entry, source_rel_path)
    key = normalize_path_key(resolved)
    for entry in index:
        if _entry_path_norm(entry) == key:
            return _target_id(entry, source_rel_path)
    return None


def _markdown_target(target: str, source_rel_path: str) -> str | None:
    """Normalize a Markdown link into a workspace-relative path.

    The ``#anchor`` and ``?query`` parts are dropped, the rest URL-decoded and
    resolved relative to the source's folder with ``posixpath.normpath``.
    """
    path = target.split("#", 1)[0].split("?", 1)[0].strip()
    if not path:
        return None
    base = posixpath.dirname(source_rel_path)
    return posixpath.normpath(posixpath.join(base, unquote(path)))


def _resolve_wikilink(
    target: str, source_rel_path: str, index: Sequence[DocIndexEntry]
) -> int | None:
    """Resolve a wikilink by rel_path, then stem, then title."""
    target = target.strip()
    if not target:
        return None
    for candidate in (target, f"{target}.md"):
        for entry in index:
            if entry.rel_path == candidate:
                return _target_id(entry, source_rel_path)
    source_folder = posixpath.dirname(source_rel_path)
    stem_key = normalize_key(wikilink_stem(target))
    matches = [entry for entry in index if entry.stem_norm == stem_key]
    if matches:
        return _target_id(_rank_first(matches, source_folder), source_rel_path)
    title_key = normalize_key(target)
    matches = [entry for entry in index if entry.title_norm == title_key]
    if matches:
        return _target_id(_rank_first(matches, source_folder), source_rel_path)
    return None


def _rank_first(candidates: list[DocIndexEntry], source_folder: str) -> DocIndexEntry:
    """Pick the best candidate: same folder, shortest path, most recent."""
    return min(candidates, key=lambda entry: _entry_rank(entry, source_folder))


def _entry_rank(entry: DocIndexEntry, source_folder: str) -> tuple[int, int, int]:
    """Sort key over stem/title candidates (see :func:`resolve_link`)."""
    return (
        0 if entry.folder == source_folder else 1,
        _folder_distance(source_folder, entry.folder),
        -entry.document_id,
    )


def _folder_distance(source: str, target: str) -> int:
    """Number of POSIX path steps between two folders (0 when equal)."""
    if source == target:
        return 0
    return len(posixpath.relpath(target or ".", source or ".").split("/"))


def _target_id(entry: DocIndexEntry, source_rel_path: str) -> int | None:
    """The entry's document id, or None for a link to the document itself."""
    return None if entry.rel_path == source_rel_path else entry.document_id


def _entry_path_norm(entry: DocIndexEntry) -> str:
    """The folded rel_path of an entry, computed on demand when missing."""
    return entry.path_norm if entry.path_norm else normalize_path_key(entry.rel_path)


def build_index(session: Session, workspace_id: int) -> list[DocIndexEntry]:
    """Build the link-target index for one workspace.

    Args:
        session: An open session/transaction.
        workspace_id: The workspace whose documents are resolvable targets.

    Returns:
        One entry per non-deleted document with a current, non-deleted file,
        ordered by document id.
    """
    rows = session.execute(
        select(Document.id, File.rel_path, Document.title_norm)
        .join(File, File.id == Document.current_file_id)
        .where(
            Document.workspace_id == workspace_id,
            Document.status != "deleted",
            File.status != "deleted",
        )
        .order_by(Document.id)
    ).all()
    return [
        DocIndexEntry(
            document_id=int(document_id),
            rel_path=rel_path,
            folder=posixpath.dirname(rel_path),
            stem_norm=normalize_key(file_stem(rel_path)),
            title_norm=title_norm,
            path_norm=normalize_path_key(rel_path),
        )
        for document_id, rel_path, title_norm in rows
    ]


def resolve_document_links(session: Session, document: Document) -> tuple[int, int]:
    """Re-resolve the latest version's links of one document.

    The workspace index is built once per call (per job run). Every resolved
    link target is upserted as a ``links_to`` relation (``method='link_parse'``,
    confidence 1.0, status ``auto``) whose evidence holds the link count and up
    to five link texts. Existing ``confirmed``/``rejected`` relations are never
    touched (user decisions win); relations of this source that no longer
    resolve are deleted.

    Args:
        session: An open session/transaction; rows are written but not
            committed.
        document: The source document; a missing latest version or stored IR
            is an idempotent no-op.

    Returns:
        ``(upserted, removed)``: how many relations were written (inserts and
        updates) and how many stale ones were deleted.
    """
    version = (
        session.get(DocumentVersion, document.latest_version_id)
        if document.latest_version_id is not None
        else None
    )
    if version is None or version.ir_json is None:
        return 0, 0
    ir = IRDocument.from_json(version.ir_json)
    file_row = (
        session.get(File, document.current_file_id)
        if document.current_file_id is not None
        else None
    )
    source_rel_path = file_row.rel_path if file_row is not None else version.rel_path_at_version
    index = build_index(session, document.workspace_id)
    counts: dict[int, int] = {}
    texts: dict[int, list[str]] = {}
    for link in ir.links:
        target_id = resolve_link(link, source_rel_path, index)
        if target_id is None or target_id == document.id:
            continue
        counts[target_id] = counts.get(target_id, 0) + 1
        evidence_texts = texts.setdefault(target_id, [])
        if len(evidence_texts) < _EVIDENCE_TEXT_LIMIT:
            evidence_texts.append(link.text or link.target)
    upserted = 0
    for target_id, count in counts.items():
        evidence = {"count": count, "texts": texts[target_id]}
        if _upsert_links_to(session, document.id, target_id, evidence):
            upserted += 1
    removed = _delete_stale_relations(session, document.id, set(counts))
    return upserted, removed


def _upsert_links_to(session: Session, src: int, dst: int, evidence: dict) -> bool:
    """Insert or refresh one ``links_to`` relation; user decisions are final."""
    existing = session.execute(
        select(DocRelation).where(
            DocRelation.src_document_id == src,
            DocRelation.dst_document_id == dst,
            DocRelation.type == "links_to",
        )
    ).scalar_one_or_none()
    if existing is not None:
        if existing.status in _FINAL_STATUSES:
            return False
        existing.method = "link_parse"
        existing.confidence = 1.0
        existing.status = "auto"
        existing.evidence = evidence
        existing.updated_at = datetime.now(UTC)
        return True
    session.add(
        DocRelation(
            src_document_id=src,
            dst_document_id=dst,
            type="links_to",
            method="link_parse",
            confidence=1.0,
            status="auto",
            evidence=evidence,
        )
    )
    session.flush()
    return True


def _delete_stale_relations(session: Session, src: int, resolved: set[int]) -> int:
    """Delete the source's ``links_to`` relations that no longer resolve."""
    relations = (
        session.execute(
            select(DocRelation).where(
                DocRelation.src_document_id == src, DocRelation.type == "links_to"
            )
        )
        .scalars()
        .all()
    )
    stale = [relation for relation in relations if relation.dst_document_id not in resolved]
    if stale:
        session.execute(delete(DocRelation).where(DocRelation.id.in_([r.id for r in stale])))
    return len(stale)


def documents_with_resolvable_links(session: Session) -> list[int]:
    """Return the ids of documents whose latest version has resolvable links.

    Args:
        session: An open session/transaction.

    Returns:
        Non-deleted documents whose latest version's stored IR contains at
        least one wikilink or Markdown (relative) link, ordered by id; URL-only
        documents are skipped. This feeds the nightly ``link-sweep``.
    """
    has_link = text(
        "EXISTS (SELECT 1 FROM jsonb_array_elements(document_version.ir_json -> 'links')"
        " AS link WHERE link ->> 'kind' IN ('wikilink', 'markdown'))"
    )
    rows = session.execute(
        select(Document.id)
        .join(
            DocumentVersion,
            and_(
                DocumentVersion.document_id == Document.id,
                DocumentVersion.is_latest.is_(True),
            ),
        )
        .where(
            Document.status != "deleted",
            DocumentVersion.ir_json.is_not(None),
            has_link,
        )
        .order_by(Document.id)
    ).scalars()
    return [int(document_id) for document_id in rows]
