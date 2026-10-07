"""The document version timeline (PLAN.md §5.1, §9 step 3).

:func:`document_history` lists a document's retained snapshots newest first,
with the sections that changed against the previous snapshot; the plain
:func:`history_timeline_text` renders that timeline as a compact context block
(version notes for history-intent answers, task 35). Old snapshots stay out of
the search index unless ``versions.index_old_versions`` is on, but they always
appear here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from mynotes.db.models import Chunk, DocumentVersion


@dataclass
class VersionInfo:
    """One retained snapshot in a document's version timeline.

    Attributes:
        version_no: 1-based snapshot number within the document.
        created_at: When the snapshot was recorded.
        doc_date: The document's best known date at this version, or None.
        rel_path: File path relative to the source root at this version.
        status: Pipeline status of the version (``embedded``, ``failed``, ...).
        chunk_count: Number of chunk rows of this version.
        is_latest: Whether this is the document's latest snapshot.
        changed_sections: Sections whose chunk content hashes differ from the
            previous retained version (added, removed or changed), sorted.
    """

    version_no: int
    created_at: datetime
    doc_date: datetime | None
    rel_path: str
    status: str
    chunk_count: int
    is_latest: bool
    changed_sections: list[str]


def document_history(session: Session, document_id: int) -> list[VersionInfo]:
    """List a document's retained versions, newest first, with per-version changes.

    ``changed_sections`` compares the chunk ``content_hash`` sets per
    ``section`` between consecutive retained versions; the oldest version has
    no previous snapshot and reports an empty list. Chunks without a section
    label never count as a changed section.

    Args:
        session: An open session/transaction.
        document_id: The document to describe; an unknown (or version-less)
            document yields an empty list.

    Returns:
        One :class:`VersionInfo` per retained version, newest first.
    """
    versions = list(
        session.execute(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.version_no.desc())
        ).scalars()
    )
    if not versions:
        return []
    section_hashes, chunk_counts = _sections_and_counts(session, document_id)
    infos: list[VersionInfo] = []
    for position, version in enumerate(versions):
        previous = versions[position + 1] if position + 1 < len(versions) else None
        if previous is None:
            changed: list[str] = []
        else:
            changed = _changed_sections(
                section_hashes.get(version.id, {}), section_hashes.get(previous.id, {})
            )
        infos.append(
            VersionInfo(
                version_no=version.version_no,
                created_at=version.created_at,
                doc_date=version.doc_date,
                rel_path=version.rel_path_at_version,
                status=version.status,
                chunk_count=chunk_counts.get(version.id, 0),
                is_latest=version.is_latest,
                changed_sections=changed,
            )
        )
    return infos


def history_timeline_text(infos: Sequence[VersionInfo], title: str) -> str:
    """Render a version timeline as one compact context line (task 35).

    Args:
        infos: The timeline entries, newest first as produced by
            :func:`document_history`.
        title: The document title to show.

    Returns:
        A block like ``Version history of "Plano Alfa": v3 2026-09-10 (latest;
        changed: Orçamento, Prazos) · v2 2026-06-02 (changed: Equipa) · v1
        2026-03-15``. Each version shows its ``doc_date`` (comparable with
        ``--as-of`` filters), falling back to ``created_at`` when unknown;
        versions without changes show only their date.
    """
    parts: list[str] = []
    for info in infos:
        date = info.doc_date if info.doc_date is not None else info.created_at
        part = f"v{info.version_no} {date:%Y-%m-%d}"
        notes: list[str] = []
        if info.is_latest:
            notes.append("latest")
        if info.changed_sections:
            notes.append("changed: " + ", ".join(info.changed_sections))
        if notes:
            part += " (" + "; ".join(notes) + ")"
        parts.append(part)
    return f'Version history of "{title}": ' + " · ".join(parts)


def _sections_and_counts(
    session: Session, document_id: int
) -> tuple[dict[int, dict[str, set[str]]], dict[int, int]]:
    """Chunk content-hash sets per section plus chunk counts, keyed by version id."""
    rows = session.execute(
        select(Chunk.document_version_id, Chunk.section, Chunk.content_hash)
        .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
        .where(DocumentVersion.document_id == document_id)
        .order_by(Chunk.document_version_id, Chunk.ordinal)
    ).all()
    hashes: dict[int, dict[str, set[str]]] = {}
    counts: dict[int, int] = {}
    for version_id, section, content_hash in rows:
        counts[version_id] = counts.get(version_id, 0) + 1
        if section is None:
            continue
        hashes.setdefault(version_id, {}).setdefault(section, set()).add(content_hash)
    return hashes, counts


def _changed_sections(new: dict[str, set[str]], old: dict[str, set[str]]) -> list[str]:
    """Sections added, removed or changed between two versions, sorted."""
    changed = [
        section
        for section in new.keys() | old.keys()
        if new.get(section, set()) != old.get(section, set())
    ]
    return sorted(changed)
