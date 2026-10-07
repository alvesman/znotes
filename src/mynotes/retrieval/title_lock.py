"""Anchor documents the question names: a quoted title or a written date.

When a question quotes a title ("what does the note "Plano Alfa" say about
X?"), the note with that title is the subject, however the lexical and vector
scores rank it: notes with similar titles (weekly meetings, one note per
project with the same template) often outrank it. :func:`lock_quoted_titles`
moves every candidate chunk of a document whose title (or file name) equals a
quoted phrase to the front, in their original order; when no candidate comes
from such a document, the document is looked up by title and its first chunks
are loaded from the index (fusion source ``title``). Expanders then see the
locked order, so a chain question's bridge is the quoted note.

Matching is exact on the folded title (:func:`normalize_key`) or the file
stem; a quoted phrase that is not a title changes nothing.

Dates work the same way (:func:`lock_named_dates`): a current-intent question
that writes a full date ("Na reunião de 2026-01-12, ...") anchors the documents
whose title or path contains that date, or that are dated that day when at
most :data:`MAX_SAME_DAY` documents share the date (a vault imported on one day
must not anchor hundreds of notes). Weekly notes otherwise lose to the newest
week, which the reranker scores the same and the recency boost prefers.

Anchors are applied twice: before fusion (so expanders see the anchored note as
the top hit and a missing note is loaded) and after reranking
(:func:`front_anchored`), because the reranker reorders everything.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from sqlalchemy import func, or_, select

from mynotes.config import get_settings
from mynotes.db.models import Chunk, Document, File
from mynotes.db.session import session_scope
from mynotes.retrieval.hybrid import _to_hit, hit_matches_filters
from mynotes.retrieval.types import SearchHit
from mynotes.text.normalize import normalize_key

if TYPE_CHECKING:
    from mynotes.retrieval.analyze import QueryAnalysis

logger = logging.getLogger(__name__)

LOADED_CHUNKS = 3
"""Chunks loaded for a quoted note that retrieval did not return at all."""

MAX_SAME_DAY = 3
"""A document dated on a named day anchors only when at most this many share the date."""

MAX_DATED_DOCUMENTS = 6
"""Documents loaded for one named date (several series can meet on the same day)."""

ANCHOR_MIN_SHARE = 0.2
"""After reranking, an anchored hit moves to the front only when its rerank score is at
least this share of the best score: a note that merely carries the date stays put."""


def _title_keys(title: str | None, path: str) -> set[str]:
    keys = {normalize_key(PurePosixPath(path).stem)}
    if title:
        keys.add(normalize_key(title))
    return {key for key in keys if key}


def lock_quoted_titles(hits: Sequence[SearchHit], analysis: QueryAnalysis) -> list[SearchHit]:
    """Return ``hits`` with the chunks of quoted-title documents first.

    Args:
        hits: The hybrid hits in rank order.
        analysis: The query analysis (``quoted``, ``filters``).

    Returns:
        The reordered (and possibly extended) list, ranks renumbered; the
        input list when nothing is quoted or no title matches.
    """
    quoted = [normalize_key(phrase) for phrase in getattr(analysis, "quoted", ())]
    quoted = [phrase for phrase in quoted if phrase]
    if not quoted:
        return list(hits)
    matched_docs: list[int] = []
    for hit in hits:
        if hit.document_id in matched_docs:
            continue
        if _title_keys(hit.title, hit.path) & set(quoted):
            matched_docs.append(hit.document_id)
    locked: list[SearchHit] = []
    if matched_docs:
        locked = [hit for hit in hits if hit.document_id in matched_docs]
        rest = [hit for hit in hits if hit.document_id not in matched_docs]
    else:
        try:
            locked = _load_quoted_documents(quoted, analysis)
        except Exception:
            logger.warning("quoted title lookup failed; ignoring it", exc_info=True)
            locked = []
        rest = list(hits)
    if not locked:
        return list(hits)
    for hit in locked:
        hit.sources = {**hit.sources, "title": 1}
    reordered = [*locked, *rest]
    for rank, hit in enumerate(reordered, start=1):
        hit.rank = rank
    return reordered


def _load_quoted_documents(quoted: Sequence[str], analysis: QueryAnalysis) -> list[SearchHit]:
    """Find current documents titled like a quoted phrase and load their first chunks."""
    from mynotes.index.opensearch import get_client

    settings = get_settings()
    patterns = [f"%{phrase}%" for phrase in quoted]
    with session_scope() as session:
        rows = session.execute(
            select(Document.id, Document.title, File.rel_path, Document.latest_version_id)
            .join(File, File.id == Document.current_file_id)
            .where(
                Document.status == "current",
                or_(
                    *(func.lower(Document.title).like(p) for p in patterns),
                    *(func.lower(File.rel_path).like(p) for p in patterns),
                ),
            )
            .limit(50)
        ).all()
        chosen = [
            (int(doc_id), int(version_id))
            for doc_id, title, rel_path, version_id in rows
            if version_id is not None and _title_keys(title, rel_path) & set(quoted)
        ]
        if not chosen:
            return []
        selected = [
            str(chunk_id)
            for (chunk_id,) in session.execute(
                select(Chunk.id)
                .where(
                    Chunk.document_version_id.in_([version for _doc, version in chosen]),
                    Chunk.ordinal < LOADED_CHUNKS,
                )
                .order_by(Chunk.document_version_id, Chunk.ordinal)
            ).all()
        ]
    if not selected:
        return []
    response = get_client().mget(
        index=settings.opensearch.chunks_alias,
        body={"docs": [{"_id": cid, "_source": {"excludes": ["embedding"]}} for cid in selected]},
    )
    hits: list[SearchHit] = []
    for doc in response.get("docs") or []:
        if not doc.get("found"):
            continue
        hit = _to_hit(doc, rank=len(hits) + 1)
        if hit_matches_filters(hit, analysis.filters):
            hit.sources = {"title": 1}
            hits.append(hit)
    return hits


def _anchor_sources(hit: SearchHit) -> bool:
    return "title" in hit.sources or "date" in hit.sources


def lock_named_dates(hits: Sequence[SearchHit], analysis: QueryAnalysis) -> list[SearchHit]:
    """Return ``hits`` with the documents of the question's named dates first (loaded if missing).

    Args:
        hits: The candidate hits in rank order.
        analysis: The query analysis (``named_dates``, ``filters``).
    """
    dates = list(getattr(analysis, "named_dates", ()))
    if not dates:
        return list(hits)
    keys = [value.isoformat() for value in dates]
    present = [hit for hit in hits if any(key in f"{hit.title or ''} {hit.path}" for key in keys)]
    try:
        loaded = _load_dated_documents(dates, analysis, exclude={hit.document_id for hit in hits})
    except Exception:
        logger.warning("named date lookup failed; ignoring it", exc_info=True)
        loaded = []
    same_day = {hit.document_id for hit in loaded}
    anchored = [*present, *[hit for hit in hits if hit.document_id in same_day]]
    if not anchored and not loaded:
        return list(hits)
    for hit in [*anchored, *loaded]:
        hit.sources = {**hit.sources, "date": 1}
    rest = [hit for hit in hits if hit not in anchored]
    reordered = [*anchored, *loaded, *rest]
    for rank, hit in enumerate(reordered, start=1):
        hit.rank = rank
    return reordered


def front_anchored(hits: Sequence[SearchHit]) -> list[SearchHit]:
    """After reranking, move anchored hits (quoted title, named date) to the front.

    Only hits whose rerank score is at least :data:`ANCHOR_MIN_SHARE` of the best
    score move; their relative (reranked) order is kept.
    """
    scores = [hit.rerank_score for hit in hits if hit.rerank_score is not None]
    floor = ANCHOR_MIN_SHARE * max(scores) if scores else 0.0
    front = [
        hit
        for hit in hits
        if _anchor_sources(hit) and (hit.rerank_score is None or hit.rerank_score >= floor)
    ]
    if not front:
        return list(hits)
    reordered = [*front, *[hit for hit in hits if hit not in front]]
    for rank, hit in enumerate(reordered, start=1):
        hit.rank = rank
    return reordered


def _load_dated_documents(
    dates: Sequence[date], analysis: QueryAnalysis, *, exclude: set[int]
) -> list[SearchHit]:
    """Current documents named or dated like ``dates`` that are not in ``exclude``."""
    from mynotes.index.opensearch import get_client

    settings = get_settings()
    keys = [value.isoformat() for value in dates]
    with session_scope() as session:
        named = session.execute(
            select(Document.id, Document.latest_version_id)
            .join(File, File.id == Document.current_file_id)
            .where(
                Document.status == "current",
                or_(
                    *(Document.title.contains(key) for key in keys),
                    *(File.rel_path.contains(key) for key in keys),
                ),
            )
            .limit(MAX_DATED_DOCUMENTS * 2)
        ).all()
        same_day: list[tuple[int, int | None]] = []
        for value in dates:
            rows = session.execute(
                select(Document.id, Document.latest_version_id)
                .where(Document.status == "current", func.date(Document.doc_date) == value)
                .limit(MAX_SAME_DAY + 1)
            ).all()
            if len(rows) <= MAX_SAME_DAY:
                same_day.extend((int(i), v) for i, v in rows)
        chosen: list[int] = []
        for doc_id, version_id in [*((int(i), v) for i, v in named), *same_day]:
            if version_id is None or doc_id in exclude or int(version_id) in chosen:
                continue
            chosen.append(int(version_id))
            if len(chosen) >= MAX_DATED_DOCUMENTS:
                break
        if not chosen:
            return []
        selected = [
            str(chunk_id)
            for (chunk_id,) in session.execute(
                select(Chunk.id)
                .where(Chunk.document_version_id.in_(chosen), Chunk.ordinal < LOADED_CHUNKS)
                .order_by(Chunk.document_version_id, Chunk.ordinal)
            ).all()
        ]
    if not selected:
        return []
    response = get_client().mget(
        index=settings.opensearch.chunks_alias,
        body={"docs": [{"_id": cid, "_source": {"excludes": ["embedding"]}} for cid in selected]},
    )
    hits: list[SearchHit] = []
    for doc in response.get("docs") or []:
        if not doc.get("found"):
            continue
        hit = _to_hit(doc, rank=len(hits) + 1)
        if hit_matches_filters(hit, analysis.filters):
            hit.sources = {"date": 1}
            hits.append(hit)
    return hits


SERIES_LIMIT = 40
"""Notes of one series folder loaded for a change question."""

SERIES_CHUNKS = 2
"""Leading chunks loaded per series note (status lines sit near the top of a note)."""


def expand_series(hits: Sequence[SearchHit], analysis: QueryAnalysis) -> list[SearchHit]:
    """For a change question, add the other dated notes of the top hit's folder.

    "Em que reunião X passou a Y?" needs every week of the series in the pool:
    twenty near-identical weekly notes compete with other series, and retrieval
    returns only some weeks, often not the week of the change. The siblings are
    placed after the existing hits (source ``series``), so the reranker scores
    them all and :func:`mynotes.retrieval.rerank.order_change_cluster` can pick
    the earliest note stating the value.
    """
    if not hits or not getattr(analysis, "asks_change", False):
        return list(hits)
    folder = str(PurePosixPath(hits[0].path).parent)
    if folder in ("", "."):
        return list(hits)
    try:
        loaded = _load_folder_documents(folder, analysis, exclude={hit.document_id for hit in hits})
    except Exception:
        logger.warning("series expansion failed; ignoring it", exc_info=True)
        return list(hits)
    # Mark the whole series (the notes already retrieved and the loaded ones) so the change
    # ordering stays inside the folder that retrieval matched to the question.
    for hit in hits:
        if str(PurePosixPath(hit.path).parent) == folder:
            hit.sources = {**hit.sources, "series": 1}
    out = [*hits, *loaded]
    for rank, hit in enumerate(out, start=1):
        hit.rank = rank
    return out


def _load_folder_documents(
    folder: str, analysis: QueryAnalysis, *, exclude: set[int]
) -> list[SearchHit]:
    """The dated current notes directly in ``folder`` (not subfolders), newest first."""
    from mynotes.index.opensearch import get_client

    settings = get_settings()
    prefix = folder.rstrip("/") + "/"
    with session_scope() as session:
        rows = session.execute(
            select(Document.id, Document.latest_version_id, File.rel_path)
            .join(File, File.id == Document.current_file_id)
            .where(
                Document.status == "current",
                Document.doc_date.is_not(None),
                File.rel_path.startswith(prefix),
            )
            .order_by(Document.doc_date.desc())
            .limit(SERIES_LIMIT * 2)
        ).all()
        versions = [
            int(version_id)
            for doc_id, version_id, rel_path in rows
            if version_id is not None
            and int(doc_id) not in exclude
            and "/" not in str(rel_path)[len(prefix) :]
        ][:SERIES_LIMIT]
        if not versions:
            return []
        selected = [
            str(chunk_id)
            for (chunk_id,) in session.execute(
                select(Chunk.id)
                .where(Chunk.document_version_id.in_(versions), Chunk.ordinal < SERIES_CHUNKS)
                .order_by(Chunk.document_version_id, Chunk.ordinal)
            ).all()
        ]
    if not selected:
        return []
    response = get_client().mget(
        index=settings.opensearch.chunks_alias,
        body={"docs": [{"_id": cid, "_source": {"excludes": ["embedding"]}} for cid in selected]},
    )
    hits: list[SearchHit] = []
    for doc in response.get("docs") or []:
        if not doc.get("found"):
            continue
        hit = _to_hit(doc, rank=len(hits) + 1)
        if hit_matches_filters(hit, analysis.filters):
            hit.sources = {"series": 1}
            hits.append(hit)
    return hits
