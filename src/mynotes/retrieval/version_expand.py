"""Version-aware retrieval: chain expansion and version notes (PLAN.md §9 step 3/6, task 35).

:class:`VersionExpander` implements the ``version`` fusion source: under
*current* intent it pulls the best-matching chunks of a chain head when an
``archived`` or duplicate document ranks high, and under *history* intent it
searches the whole supersedes chain of the top hits (superseded documents
included, old snapshots only when ``versions.index_old_versions`` is on).
:class:`VersionNotes` implements the ``version_notes`` extra block: one line
per active supersedes relation of the selected documents plus the snapshot
timeline of documents edited in place, so "what changed" questions get an
answer even though snapshot content is not indexed by default.

Both register themselves on import; :mod:`mynotes.retrieval.pipeline` imports
this module so a plain ``retrieve()`` call gets version-aware results.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, aliased

from mynotes.config import get_settings
from mynotes.core.acl import path_allowed
from mynotes.db.models import DocRelation, Document, File
from mynotes.db.session import session_scope
from mynotes.retrieval.assemble import register_extra_block
from mynotes.retrieval.hybrid import hybrid_search
from mynotes.retrieval.pipeline import register_expander
from mynotes.retrieval.types import SearchHit
from mynotes.versioning.chains import head_of_chain
from mynotes.versioning.history import document_history, history_timeline_text

if TYPE_CHECKING:
    from mynotes.config import Settings
    from mynotes.retrieval.analyze import QueryAnalysis
    from mynotes.retrieval.pipeline import RetrievalResult

logger = logging.getLogger(__name__)

_ACTIVE_STATUSES = ("auto", "confirmed")

_VERSION_EXPANSION_SIZE = 3
"""Hits fetched per restricted chain search."""

_VERSION_EXPANSION_TOP = 10
"""Hybrid hits inspected by the expander."""


class VersionExpander:
    """The ``version`` fusion source (PLAN.md §9 step 3).

    Attributes:
        name: The fusion source name used by the client-side RRF.
    """

    name = "version"

    def __init__(
        self, top_hits: int = _VERSION_EXPANSION_TOP, size: int = _VERSION_EXPANSION_SIZE
    ) -> None:
        """Args:
        top_hits: How many of the leading hybrid hits to inspect.
        size: Hits to fetch per restricted chain search.
        """
        self._top_hits = top_hits
        self._size = size

    def enabled(self, settings: Settings, analysis: QueryAnalysis) -> bool:
        """Whether this expander contributes for the given query (always)."""
        return True

    def expand(self, analysis: QueryAnalysis, hybrid_hits: Sequence[SearchHit]) -> list[SearchHit]:
        """Return the version-expansion candidate hits, best first.

        Args:
            analysis: The query analysis (text, filters, intent).
            hybrid_hits: The hybrid search hits in rank order.

        Returns:
            Hits from the head documents (current intent) or from the whole
            chains (history intent); empty when no hit is part of a chain.
        """
        candidates = list(hybrid_hits)[: self._top_hits]
        if not candidates:
            return []
        with session_scope() as session:
            if analysis.intent == "history":
                return self._expand_history(session, analysis, candidates)
            return self._expand_current(session, analysis, candidates)

    def _expand_current(
        self, session: Session, analysis: QueryAnalysis, candidates: list[SearchHit]
    ) -> list[SearchHit]:
        """Search the head of the chain for archived/duplicate hit documents."""
        heads: list[int] = []
        unresolved: list[int] = []
        searched: set[int] = set()
        for hit in candidates:
            head = head_of_chain(session, hit.document_id)
            if head != hit.document_id:
                if head not in searched:
                    searched.add(head)
                    heads.append(head)
            else:
                unresolved.append(hit.document_id)
        duplicate_targets = _duplicate_targets(session, unresolved)
        for document_id, duplicate in duplicate_targets.items():
            head = head_of_chain(session, duplicate)
            if head != document_id and head not in searched:
                searched.add(head)
                heads.append(head)
        expanded: list[SearchHit] = []
        for head in heads:
            filters = replace(analysis.filters, document_ids=[head])
            expanded.extend(hybrid_search(analysis.text, filters, size=self._size))
        return expanded

    def _expand_history(
        self, session: Session, analysis: QueryAnalysis, candidates: list[SearchHit]
    ) -> list[SearchHit]:
        """Search the whole supersedes chain of every distinct hit document."""
        settings = get_settings()
        expanded: list[SearchHit] = []
        seen_documents: set[int] = set()
        searched_chains: set[frozenset[int]] = set()
        for hit in candidates:
            if hit.document_id in seen_documents:
                continue
            seen_documents.add(hit.document_id)
            members = _chain_members(session, hit.document_id)
            chain = frozenset(members)
            if len(members) < 2 or chain in searched_chains:
                continue
            searched_chains.add(chain)
            filters = replace(
                analysis.filters,
                document_ids=members,
                include_superseded=True,
                include_old_versions=settings.versions.index_old_versions,
            )
            expanded.extend(hybrid_search(analysis.text, filters, size=self._size))
        return expanded


class VersionNotes:
    """The ``version_notes`` extra block (PLAN.md §9 step 6).

    Attributes:
        name: The provider name, rendered as the ``### Version notes`` heading.
    """

    name = "version_notes"

    def block(self, result: RetrievalResult, selected: Sequence[SearchHit]) -> str | None:
        """Render the version notes of the selected documents, or None.

        One line per active supersedes relation touching a selected document
        (``"newer" supersedes "older" (date)``), plus the snapshot timeline of
        documents edited in place (more than one retained version) — under
        history intent those timelines are what "what changed" questions get,
        because snapshot content is not indexed by default. A relation line
        appears only when both endpoint documents lie inside the query's ACL
        prefixes, so notes never name a restricted document (task 52).

        Args:
            result: The pipeline result (its analysis carries the intent and
                the ACL filters).
            selected: The hits chosen for the context.

        Returns:
            The block text, or None when there is nothing to say.
        """
        entries: list[tuple[int, str | None, str]] = []
        seen: set[int] = set()
        for hit in selected:
            if hit.document_id in seen:
                continue
            seen.add(hit.document_id)
            entries.append((hit.document_id, hit.title, hit.path))
        if not entries:
            return None
        acl_prefixes = result.analysis.filters.acl_prefixes
        with session_scope() as session:
            lines = _relation_lines(
                session, [document_id for document_id, _, _ in entries], acl_prefixes
            )
            lines.extend(_timeline_lines(session, result.analysis.intent, entries))
        return "\n".join(lines) if lines else None


def register() -> None:
    """(Re-)register the expander and the notes provider with the registries.

    Called at import and by tests that unregistered them for isolation;
    registering replaces any entry with the same name.
    """
    register_expander(VersionExpander())
    register_extra_block(VersionNotes())


def _duplicate_targets(session: Session, document_ids: Sequence[int]) -> dict[int, int]:
    """The active ``duplicate_of`` target of each document (lowest id wins)."""
    if not document_ids:
        return {}
    rows = session.execute(
        select(DocRelation.src_document_id, DocRelation.dst_document_id)
        .where(
            DocRelation.src_document_id.in_(document_ids),
            DocRelation.type == "duplicate_of",
            DocRelation.status.in_(_ACTIVE_STATUSES),
        )
        .order_by(DocRelation.id)
    ).all()
    targets: dict[int, int] = {}
    for src, dst in rows:
        targets.setdefault(src, dst)
    return targets


def _chain_members(session: Session, document_id: int) -> list[int]:
    """The active supersedes chain around ``document_id``, oldest first.

    Follows active (``auto``/``confirmed``) supersedes relations in both
    directions; cycles are broken on the first revisit and, when several
    relations leave one document, the lowest relation id is followed (the
    same rules as :func:`~mynotes.versioning.chains.head_of_chain`).
    """
    seen = {document_id}
    older_ids: list[int] = []
    current = document_id
    while True:
        older = session.execute(
            select(DocRelation.dst_document_id)
            .where(
                DocRelation.src_document_id == current,
                DocRelation.type == "supersedes",
                DocRelation.status.in_(_ACTIVE_STATUSES),
            )
            .order_by(DocRelation.id)
            .limit(1)
        ).scalar_one_or_none()
        if older is None or older in seen:
            break
        seen.add(older)
        older_ids.append(older)
        current = older
    newer_ids: list[int] = []
    current = document_id
    while True:
        newer = session.execute(
            select(DocRelation.src_document_id)
            .where(
                DocRelation.dst_document_id == current,
                DocRelation.type == "supersedes",
                DocRelation.status.in_(_ACTIVE_STATUSES),
            )
            .order_by(DocRelation.id)
            .limit(1)
        ).scalar_one_or_none()
        if newer is None or newer in seen:
            break
        seen.add(newer)
        newer_ids.append(newer)
        current = newer
    return [*reversed(older_ids), document_id, *newer_ids]


def _relation_lines(
    session: Session,
    document_ids: Sequence[int],
    acl_prefixes: Sequence[str] | None = None,
) -> list[str]:
    """One ``"newer" supersedes "older" (date)`` line per visible active relation.

    With ``acl_prefixes``, a line is only produced when both endpoint paths are
    allowed, so restricted callers never see the other side's title or path
    (task 52; the same exact-path rule as the hybrid search)."""
    if not document_ids:
        return []
    newer_doc = aliased(Document)
    older_doc = aliased(Document)
    newer_file = aliased(File)
    older_file = aliased(File)
    rows = session.execute(
        select(
            DocRelation,
            newer_doc.title,
            newer_doc.doc_date,
            newer_file.rel_path,
            older_doc.title,
            older_doc.doc_date,
            older_file.rel_path,
        )
        .join(newer_doc, newer_doc.id == DocRelation.src_document_id)
        .join(older_doc, older_doc.id == DocRelation.dst_document_id)
        .outerjoin(newer_file, newer_file.id == newer_doc.current_file_id)
        .outerjoin(older_file, older_file.id == older_doc.current_file_id)
        .where(
            DocRelation.type == "supersedes",
            DocRelation.status.in_(_ACTIVE_STATUSES),
            or_(
                DocRelation.src_document_id.in_(document_ids),
                DocRelation.dst_document_id.in_(document_ids),
            ),
        )
        .order_by(DocRelation.id)
    ).all()
    lines: list[str] = []
    for relation, newer_title, newer_date, newer_path, older_title, _older_date, older_path in rows:
        if acl_prefixes is not None and not (
            path_allowed(newer_path or "", acl_prefixes)
            and path_allowed(older_path or "", acl_prefixes)
        ):
            continue
        newer_label = newer_title or newer_path or f"document {relation.src_document_id}"
        older_label = older_title or older_path or f"document {relation.dst_document_id}"
        if newer_label == older_label and (newer_path or older_path):
            newer_label = newer_path or newer_label
            older_label = older_path or older_label
        line = f'"{newer_label}" supersedes "{older_label}"'
        if newer_date is not None:
            line += f" ({newer_date:%Y-%m-%d})"
        lines.append(line)
    return lines


def _timeline_lines(
    session: Session,
    intent: str,
    entries: Sequence[tuple[int, str | None, str]],
) -> list[str]:
    """Snapshot timelines of the selected documents (task 33's timeline text).

    History intent asks for the timelines; documents edited in place (more
    than one retained version) provide "what changed" context under any
    intent. A document with a single retained snapshot has no timeline to
    tell, so it is skipped either way.
    """
    del intent
    lines: list[str] = []
    for document_id, title, path in entries:
        infos = document_history(session, document_id)
        if len(infos) < 2:
            continue
        lines.append(history_timeline_text(infos, title or path))
    return lines


register()
