"""Lexical mention linking: attach known entities whose name appears verbatim in a chunk.

The LLM extractor misses mentions in short notes and lists ("assenta na
ferramenta Stanza" yields "ferramentas externas" but no Stanza), and a note
extracted before the entity existed can never have linked to it. This pass
scans a chunk's normalized text for the names and aliases of entities the
workspace already knows and adds the ``kg_mention`` rows the extractor did not
produce. It never creates entities or relations.

Precision guards (``graph.lexical_*`` settings): only named-thing types
(Technology, Project, Person, Organization by default), names of at least
``lexical_min_length`` characters, whole-word matches on the accent/case-folded
text, a per-chunk cap, and a rarity bound: entities with more than
``lexical_max_mentions`` extractor mentions are hubs ("dataview", "obsidian")
and are never linked lexically. Lexical mentions carry
:data:`LEXICAL_CONFIDENCE` so they can be told apart from extractor mentions
(confidence 1.0) and rebuilt idempotently (``mynotes graph link-mentions``).

:func:`mynotes.graph.resolve.resolve_group` runs the pass for every resolved
group (forward direction: new text against known entities); the workspace
pass (:func:`relink_workspace`, nightly maintenance and the CLI) covers the
other direction: entities created after the text was resolved.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from mynotes.config import Settings
from mynotes.db.models import (
    Chunk,
    Document,
    DocumentVersion,
    KgEntity,
    KgEntityAlias,
    KgMention,
)
from mynotes.text.normalize import normalize_key

logger = logging.getLogger(__name__)

LEXICAL_CONFIDENCE = 0.6
"""``kg_mention.confidence`` of a lexically linked mention (extractor mentions: 1.0)."""

EXTRACTOR_MENTION_CONFIDENCE = 1.0
"""Confidence of an extractor mention; ``mention_count_current`` counts only these."""
_EXTRACTOR_CONFIDENCE = EXTRACTOR_MENTION_CONFIDENCE


@dataclass(frozen=True)
class LexicalParams:
    """The precision guards of the lexical pass (``graph.lexical_*`` settings)."""

    min_length: int = 4
    max_mentions: int = 50
    max_chunk_share: float = 0.02
    max_ratio: float = 8.0
    per_chunk: int = 8
    types: frozenset[str] = frozenset({"Technology", "Project", "Person", "Organization"})

    @classmethod
    def from_settings(cls, settings: Settings) -> LexicalParams | None:
        """The parameters from the ``graph`` section, or None when the pass is off."""
        graph = settings.graph
        if not graph.lexical_mentions:
            return None
        return cls(
            min_length=graph.lexical_min_length,
            max_mentions=graph.lexical_max_mentions,
            max_chunk_share=graph.lexical_max_chunk_share,
            max_ratio=graph.lexical_max_ratio,
            per_chunk=graph.lexical_per_chunk,
            types=frozenset(graph.lexical_types),
        )


@dataclass(frozen=True)
class LexicalCandidate:
    """One name (canonical or alias, normalized) of a linkable entity."""

    entity_id: int
    key: str


_GENERIC_FLOOR_CHUNKS = 20
"""A key is generic when it occurs in more than max(this, share x chunks) chunks."""


@dataclass
class LexicalCache:
    """Per-workspace counts reused across groups and versions of one pass."""

    workspace_id: int
    total_chunks: int | None = None
    chunk_frequency: dict[str, int] | None = None

    def generic_limit(self, session: Session, share: float) -> int:
        """The chunk count above which a key is too common to link."""
        if self.total_chunks is None:
            self.total_chunks = int(
                session.execute(
                    select(func.count())
                    .select_from(Chunk)
                    .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
                    .join(Document, Document.id == DocumentVersion.document_id)
                    .where(
                        Document.workspace_id == self.workspace_id,
                        Document.status == "current",
                        DocumentVersion.is_latest.is_(True),
                    )
                ).scalar_one()
            )
        return max(_GENERIC_FLOOR_CHUNKS, int(share * self.total_chunks))

    def frequency(self, session: Session, key: str) -> int:
        """How many current, latest chunks of the workspace contain ``key``.

        Case-insensitive on the raw text (accents are not folded, so an
        accented name's count is a slight underestimate: more permissive).
        """
        if self.chunk_frequency is None:
            self.chunk_frequency = {}
        cached = self.chunk_frequency.get(key)
        if cached is not None:
            return cached
        count = int(
            session.execute(
                select(func.count())
                .select_from(Chunk)
                .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
                .join(Document, Document.id == DocumentVersion.document_id)
                .where(
                    Document.workspace_id == self.workspace_id,
                    Document.status == "current",
                    DocumentVersion.is_latest.is_(True),
                    func.strpos(func.lower(Chunk.text), key) > 0,
                )
            ).scalar_one()
        )
        self.chunk_frequency[key] = count
        return count


def usable_alias(alias_key: str, canonical_key: str) -> bool:
    """Whether an alias is specific enough to link by itself.

    A single letters-only word is used only when it is a word of the entity's
    own name ("marta" for "Marta Quelhas"): extractor aliases such as
    "square" for an ISO standard or "flow" for a proof of concept are common
    words. Multi-word aliases and aliases with digits or punctuation
    ("k8s", "#dv/where", "acme-org/field-app") are always used.

    Args:
        alias_key: The normalized alias.
        canonical_key: The entity's normalized name.
    """
    if " " in alias_key or not alias_key.isalpha():
        return True
    return alias_key in canonical_key.split()


def _word_bounded(key: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(key)}(?!\w)")


def find_lexical_mentions(
    norm_text: str, candidates: Iterable[LexicalCandidate], per_chunk: int
) -> list[LexicalCandidate]:
    """The candidates whose key occurs as a whole word in ``norm_text``.

    One result per entity (its longest matching key), longest keys first so a
    chunk's cap keeps the most specific names, then by entity id for
    determinism.

    Args:
        norm_text: The chunk text through :func:`normalize_key`.
        candidates: Names to look for; ``key`` must already be normalized.
        per_chunk: Maximum number of entities returned.

    Returns:
        At most ``per_chunk`` candidates, distinct by entity.
    """
    best: dict[int, str] = {}
    for candidate in candidates:
        key = candidate.key
        if not key or key not in norm_text:
            continue
        if len(key) <= len(best.get(candidate.entity_id, "")):
            continue
        if _word_bounded(key).search(norm_text):
            best[candidate.entity_id] = key
    ordered = sorted(best.items(), key=lambda item: (-len(item[1]), item[0]))
    return [LexicalCandidate(entity_id, key) for entity_id, key in ordered[:per_chunk]]


def lexical_candidates(
    session: Session,
    workspace_id: int,
    text: str,
    params: LexicalParams,
    cache: LexicalCache | None = None,
) -> list[LexicalCandidate]:
    """The workspace's linkable names that occur somewhere in ``text``.

    PostgreSQL prefilters with ``strpos`` over active entities of the allowed
    types and their aliases; :func:`find_lexical_mentions` confirms word
    boundaries per chunk. Three rarity bounds, all counting extractor mentions
    only so the pass's own rows never change who qualifies: the entity's
    extractor mentions (``max_mentions``: hubs), the key's chunk frequency
    (``max_chunk_share`` of the workspace's chunks, at least 20: "table",
    "where", "user" typed as an entity), and the ratio between the two
    (``max_ratio``: a name the extractor found once but that occurs in forty
    chunks, "Setup", "curl", or the alias "endpoint", is a common word, not a
    missed mention).

    Args:
        session: An open session.
        workspace_id: The workspace to search.
        text: The normalized text (one chunk or a whole version) to prefilter by.
        params: The precision guards.
        cache: Counts shared across calls of one pass (created when omitted).

    Returns:
        Distinct ``(entity_id, key)`` pairs; empty when ``text`` is empty.
    """
    if not text or not params.types:
        return []
    if cache is None:
        cache = LexicalCache(workspace_id)
    extractor_mentions = (
        select(func.count())
        .select_from(KgMention)
        .where(
            KgMention.entity_id == KgEntity.id,
            KgMention.confidence >= _EXTRACTOR_CONFIDENCE,
        )
        .scalar_subquery()
    )
    base = [
        KgEntity.workspace_id == workspace_id,
        KgEntity.status == "active",
        KgEntity.type.in_(sorted(params.types)),
        extractor_mentions <= params.max_mentions,
    ]
    names = session.execute(
        select(KgEntity.id, KgEntity.norm_name, extractor_mentions).where(
            *base,
            func.length(KgEntity.norm_name) >= params.min_length,
            func.strpos(text, KgEntity.norm_name) > 0,
        )
    ).all()
    aliases = session.execute(
        select(KgEntity.id, KgEntityAlias.alias_norm, extractor_mentions, KgEntity.norm_name)
        .join(KgEntityAlias, KgEntityAlias.entity_id == KgEntity.id)
        .where(
            *base,
            func.length(KgEntityAlias.alias_norm) >= params.min_length,
            func.strpos(text, KgEntityAlias.alias_norm) > 0,
        )
    ).all()
    rows = [(entity_id, key, mentions) for entity_id, key, mentions in names]
    rows.extend(
        (entity_id, key, mentions)
        for entity_id, key, mentions, canonical in aliases
        if usable_alias(str(key), str(canonical))
    )
    seen: set[tuple[int, str]] = set()
    result: list[LexicalCandidate] = []
    limit: int | None = None
    for entity_id, key, mentions in rows:
        pair = (int(entity_id), str(key))
        if pair in seen:
            continue
        seen.add(pair)
        if limit is None:
            limit = cache.generic_limit(session, params.max_chunk_share)
        frequency = cache.frequency(session, pair[1])
        if frequency > limit or frequency > params.max_ratio * max(1, int(mentions or 0)):
            continue
        result.append(LexicalCandidate(*pair))
    return result


def link_lexical_mentions(
    session: Session,
    *,
    workspace_id: int,
    version_id: int,
    chunks: Sequence[tuple[UUID, str]],
    skip: set[tuple[int, UUID]],
    params: LexicalParams,
    dry_run: bool = False,
    cache: LexicalCache | None = None,
) -> tuple[set[int], int]:
    """Add lexical mentions for ``chunks`` and return the touched entity ids.

    Args:
        session: An open session/transaction; rows are added, not committed.
        workspace_id: The chunks' workspace.
        version_id: The document version the chunks belong to.
        chunks: ``(chunk_id, normalized text)`` pairs.
        skip: ``(entity_id, chunk_id)`` pairs that already have a mention.
        params: The precision guards.
        dry_run: Count the mentions without inserting them.
        cache: Counts shared across calls of one pass.

    Returns:
        ``(touched entity ids, mentions added)``.
    """
    if not chunks:
        return set(), 0
    candidates = lexical_candidates(
        session, workspace_id, " | ".join(text for _chunk_id, text in chunks), params, cache
    )
    if not candidates:
        return set(), 0
    touched: set[int] = set()
    added = 0
    for chunk_id, text in chunks:
        for match in find_lexical_mentions(text, candidates, params.per_chunk):
            if (match.entity_id, chunk_id) in skip:
                continue
            touched.add(match.entity_id)
            added += 1
            if not dry_run:
                session.add(
                    KgMention(
                        entity_id=match.entity_id,
                        chunk_id=chunk_id,
                        document_version_id=version_id,
                        surface_form=match.key,
                        confidence=LEXICAL_CONFIDENCE,
                    )
                )
    return touched, added


def relink_version(
    session: Session,
    *,
    workspace_id: int,
    version_id: int,
    params: LexicalParams,
    dry_run: bool = False,
    cache: LexicalCache | None = None,
) -> tuple[int, int]:
    """Rebuild the lexical mentions of one version against today's entities.

    Deletes the version's lexical mentions, re-links every chunk and refreshes
    the touched entities' current mention counts.

    Returns:
        ``(mentions removed, mentions added)``; a dry run returns ``(0, n)`` where
        ``n`` is what a real run would add after removing the current rows.
    """
    from mynotes.graph.resolve import _recount_mentions

    chunk_rows = session.execute(
        select(Chunk.id, Chunk.text)
        .where(Chunk.document_version_id == version_id)
        .order_by(Chunk.ordinal)
    ).all()
    chunk_ids = [UUID(str(chunk_id)) for chunk_id, _text in chunk_rows]
    if not chunk_ids:
        return 0, 0
    previous = {
        int(entity_id)
        for (entity_id,) in session.execute(
            select(KgMention.entity_id).where(
                KgMention.chunk_id.in_(chunk_ids),
                KgMention.confidence < _EXTRACTOR_CONFIDENCE,
            )
        ).all()
    }
    removed = 0
    if not dry_run:
        removed = int(
            session.execute(
                delete(KgMention).where(
                    KgMention.chunk_id.in_(chunk_ids),
                    KgMention.confidence < _EXTRACTOR_CONFIDENCE,
                )
            ).rowcount
            or 0
        )
    skip = {
        (int(entity_id), UUID(str(chunk_id)))
        for entity_id, chunk_id in session.execute(
            select(KgMention.entity_id, KgMention.chunk_id).where(
                KgMention.chunk_id.in_(chunk_ids),
                KgMention.confidence >= _EXTRACTOR_CONFIDENCE,
            )
        ).all()
    }
    touched, added = link_lexical_mentions(
        session,
        workspace_id=workspace_id,
        version_id=version_id,
        chunks=[(UUID(str(chunk_id)), normalize_key(text)) for chunk_id, text in chunk_rows],
        skip=skip,
        params=params,
        dry_run=dry_run,
        cache=cache,
    )
    if not dry_run:
        _recount_mentions(session, touched | previous)
    return removed, added


def relink_workspace(
    workspace_id: int,
    params: LexicalParams,
    *,
    dry_run: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Rebuild the lexical mentions of every current document's latest version.

    One transaction per version, so an interruption loses at most one
    document and the worker's own resolution jobs are never blocked long.

    Args:
        workspace_id: The workspace to relink.
        params: The precision guards.
        dry_run: Count without writing.
        progress: Optional ``(done, total)`` callback after each version.

    Returns:
        ``{"versions": n, "mentions_removed": n, "mentions_added": n}``.
    """
    from mynotes.db.session import session_scope

    with session_scope() as session:
        version_ids = [
            int(version_id)
            for (version_id,) in session.execute(
                select(DocumentVersion.id)
                .join(Document, Document.id == DocumentVersion.document_id)
                .where(
                    Document.workspace_id == workspace_id,
                    Document.status == "current",
                    DocumentVersion.is_latest.is_(True),
                )
                .order_by(DocumentVersion.id)
            ).all()
        ]
    removed_total = added_total = 0
    cache = LexicalCache(workspace_id)
    for index, version_id in enumerate(version_ids, start=1):
        with session_scope() as session:
            removed, added = relink_version(
                session,
                workspace_id=workspace_id,
                version_id=version_id,
                params=params,
                dry_run=dry_run,
                cache=cache,
            )
        removed_total += removed
        added_total += added
        if progress is not None:
            progress(index, len(version_ids))
    logger.info(
        "lexical mentions%s: %d version(s), %d removed, %d added",
        " (dry run)" if dry_run else "",
        len(version_ids),
        removed_total,
        added_total,
    )
    return {
        "versions": len(version_ids),
        "mentions_removed": removed_total,
        "mentions_added": added_total,
    }
