"""Incremental LLM knowledge-graph extraction (PLAN.md §8.3, task 40).

A document version is split into *section groups* (consecutive chunks of the
same section and page run, up to ~1,500 tokens; tables stand alone). Every
non-skipped group is sent to the extract LLM as a background call, validated
against the :class:`Extraction` schema and cached in ``kg_extraction_cache`` by
the group's content hash and :data:`mynotes.graph.ontology.EXTRACTOR_VERSION`,
so re-running a job or re-processing an unchanged section never calls the model
again. A group whose answer does not parse even after the repair retry (most
often a dense passage whose JSON ran past the output cap) is split at
paragraph boundaries and extracted in parts, up to :data:`_MAX_SPLIT_DEPTH`
times; the merged result is cached under the group's own hash. The handler
carries the entities found so far into the next group's
prompt (known entities) and hands the per-group list to ``resolve_entities``
(task 41); the extraction results themselves stay in the cache.

LLM concurrency: several ``kg_extract`` jobs run in parallel across the
worker's job threads; the total number of in-flight LLM calls is capped by the
background pool (``models.llm.limits.background_concurrency``, default 8).
Because each job waits on that pool while holding a worker thread, run
``mynotes worker --concurrency 10`` so all 8 LLM slots can actually be used.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from mynotes.db.models import Chunk, DocumentVersion, KgExtractionCache
from mynotes.graph.ontology import ENTITY_TYPES, EXTRACTOR_VERSION, RELATION_TYPES
from mynotes.llm.chat import ChatMessage, ChatModel
from mynotes.llm.structured import StructuredOutputError, complete_structured
from mynotes.scan.hashing import hash_text
from mynotes.text.normalize import normalize_key

logger = logging.getLogger(__name__)

_MAX_GROUP_TOKENS = 1500
_MIN_GROUP_TOKENS = 40
_BOILERPLATE_DOTS_SHARE = 0.3
_MAX_KNOWN_ENTITIES = 50
_EXTRACT_MAX_TOKENS = 8000
_MAX_SPLIT_DEPTH = 2
_ERROR_CHARS = 500
_MAX_ENTITIES = 40
_MAX_RELATIONS = 60

_DOTTED_LINE_RE = re.compile(r"\.{4,}\s*\d+$")

_BOILERPLATE_SECTIONS = frozenset(
    {
        "indice",
        "index",
        "table of contents",
        "sumario",
        "contents",
        "copyright",
        "legal",
        "termos",
    }
)

_ENTITY_DEFINITIONS: dict[str, str] = {
    "Project": "a named project or initiative the notes plan, track or discuss",
    "Idea": "an idea, proposal, hypothesis or open question",
    "Person": "a person: author, stakeholder, colleague or contact",
    "Organization": "a company, team, institution or other group of people",
    "Technology": "a tool, framework, language, product or technical system",
    "Concept": "a topic, method or abstract notion that fits no other type",
    "Decision": "a decision that was made, with its outcome",
    "Requirement": "a stated requirement, constraint or acceptance criterion",
    "Milestone": "a named goal, deadline, release or planned event (not a bare date)",
    "Risk": "a stated risk, threat, blocker or concern",
    "Metric": "a measurable quantity such as a budget, size, score or duration",
}

_RELATION_DEFINITIONS: dict[str, str] = {
    "part_of": "X is part of Y",
    "depends_on": "X depends on Y",
    "uses": "X uses Y",
    "proposes": "X proposes Y",
    "decided": "X decided Y (Y is a Decision)",
    "owned_by": "X is owned by Y",
    "works_on": "X works on Y",
    "targets": "X targets Y (a Metric or Milestone)",
    "has_requirement": "X has requirement Y",
    "has_risk": "X has risk Y",
    "measured_by": "X is measured by Y (a Metric)",
    "scheduled_for": "X is scheduled for Y (a Milestone or date)",
    "replaces": "X replaces Y",
    "contradicts": "X contradicts Y",
    "related_to": "generic relation when nothing else fits",
}

_ENTITIES_ONLY_INSTRUCTION = "Return entities only; relations must be an empty list."

_JSON_SHAPE = (
    'JSON shape: {"entities": [{"name": ..., "type": ..., "description": ..., "aliases": [...]}], '
    '"relations": [{"source": ..., "target": ..., "type": ..., "raw_label": ..., '
    '"description": ..., "attribute": null, "value": null, "quote": ..., "confidence": 0.0}]}'
)


@dataclass
class ChunkRowFull:
    """The chunk fields the KG extractor needs, in ordinal order."""

    id: UUID
    text: str
    section: str | None
    is_table: bool
    page: int | None
    slide: int | None
    token_count: int


@dataclass
class ChunkGroup:
    """One extraction unit: consecutive same-section chunks (or one table).

    Attributes:
        chunk_ids: The chunk rows the group's text was built from.
        section: The shared section title, or None.
        text: The chunk texts joined with blank lines.
        content_hash: ``hash_text(text + "\\x1f" + EXTRACTOR_VERSION)``; the
            cache key of the group's extraction result.
        token_count: Sum of the chunks' token counts.
        is_table: True for a single-table group (entity-only extraction).
    """

    chunk_ids: list[UUID]
    section: str | None
    text: str
    content_hash: str
    token_count: int
    is_table: bool = field(default=False)


class ExtractedEntity(BaseModel):
    """One entity as the model wrote it, before resolution (task 41)."""

    name: str = Field(min_length=1, max_length=120)
    type: str
    description: str = Field(default="", max_length=300)
    aliases: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("type")
    @classmethod
    def _map_unknown_type(cls, value: str) -> str:
        """Map a type outside the ontology to ``"Concept"``."""
        return value if value in ENTITY_TYPES else "Concept"


class ExtractedRelation(BaseModel):
    """One relation as the model wrote it, before resolution (task 41)."""

    source: str
    target: str
    type: str
    raw_label: str = ""
    description: str = Field(default="", max_length=300)
    attribute: str | None = None
    value: str | None = None
    quote: str = Field(default="", max_length=300)
    confidence: float = Field(default=0.7, ge=0, le=1)

    @field_validator("type")
    @classmethod
    def _map_unknown_type(cls, value: str) -> str:
        """Map a type outside the ontology to ``"related_to"``."""
        return value if value in RELATION_TYPES else "related_to"


class Extraction(BaseModel):
    """The validated model answer for one section group."""

    entities: list[ExtractedEntity] = Field(default_factory=list, max_length=_MAX_ENTITIES)
    relations: list[ExtractedRelation] = Field(default_factory=list, max_length=_MAX_RELATIONS)

    @model_validator(mode="after")
    def _drop_dangling_relations(self) -> Extraction:
        """Drop relations whose source or target is not an extracted entity name."""
        names = {normalize_key(entity.name) for entity in self.entities}
        self.relations = [
            relation
            for relation in self.relations
            if normalize_key(relation.source) in names and normalize_key(relation.target) in names
        ]
        return self


def _page_key(chunk: ChunkRowFull) -> int | None:
    """The PDF page or PPTX slide a chunk belongs to (None for plain notes)."""
    return chunk.page if chunk.page is not None else chunk.slide


def build_groups(
    chunks: Sequence[ChunkRowFull], max_tokens: int = _MAX_GROUP_TOKENS
) -> list[ChunkGroup]:
    """Merge consecutive chunks into section groups within the token budget.

    Chunks merge while they share the section title and the same page/slide
    run and the total stays within ``max_tokens``; a table chunk always forms
    its own single-chunk group.

    Args:
        chunks: The version's chunks in ordinal order.
        max_tokens: Group token budget.

    Returns:
        The groups in document order.
    """
    groups: list[ChunkGroup] = []
    current: list[ChunkRowFull] = []

    def flush() -> None:
        if not current:
            return
        text = "\n\n".join(part.text for part in current)
        groups.append(
            ChunkGroup(
                chunk_ids=[part.id for part in current],
                section=current[0].section,
                text=text,
                content_hash=hash_text(text + "\x1f" + EXTRACTOR_VERSION),
                token_count=sum(part.token_count for part in current),
            )
        )
        current.clear()

    for chunk in chunks:
        if chunk.is_table:
            flush()
            groups.append(
                ChunkGroup(
                    chunk_ids=[chunk.id],
                    section=chunk.section,
                    text=chunk.text,
                    content_hash=hash_text(chunk.text + "\x1f" + EXTRACTOR_VERSION),
                    token_count=chunk.token_count,
                    is_table=True,
                )
            )
            continue
        if (
            current
            and current[0].section == chunk.section
            and _page_key(current[-1]) == _page_key(chunk)
            and sum(part.token_count for part in current) + chunk.token_count <= max_tokens
        ):
            current.append(chunk)
            continue
        flush()
        current.append(chunk)
    flush()
    return groups


def extraction_groups(
    chunks: Sequence[ChunkRowFull], max_tokens: int = _MAX_GROUP_TOKENS
) -> list[ChunkGroup]:
    """The groups the extractor works with: :func:`build_groups` plus
    :func:`merge_small_groups`, so notes made of short sections are extracted too."""
    return merge_small_groups(build_groups(chunks, max_tokens), max_tokens=max_tokens)


def merge_small_groups(
    groups: Sequence[ChunkGroup],
    *,
    min_tokens: int = _MIN_GROUP_TOKENS,
    max_tokens: int = _MAX_GROUP_TOKENS,
) -> list[ChunkGroup]:
    """Merge runs of short, consecutive, non-table groups into extractable ones.

    Notes made of several short sections (a few lines each) would otherwise
    produce only groups below ``min_tokens``, which :func:`should_skip` drops,
    and the note would never reach the knowledge graph. Consecutive groups that
    are each below ``min_tokens`` (or that follow such a run while the run is
    still below ``min_tokens``) are merged, within ``max_tokens``; tables and
    groups already above the minimum are left alone.

    Args:
        groups: The groups in document order.
        min_tokens: The extractor's minimum group size.
        max_tokens: The group token budget.

    Returns:
        The groups in document order, with short runs merged.
    """
    merged: list[ChunkGroup] = []
    run: list[ChunkGroup] = []

    def run_tokens() -> int:
        return sum(group.token_count for group in run)

    def flush_run() -> None:
        if not run:
            return
        if len(run) == 1:
            merged.append(run[0])
        else:
            text = "\n\n".join(group.text for group in run)
            sections = list(dict.fromkeys(group.section for group in run if group.section))
            merged.append(
                ChunkGroup(
                    chunk_ids=[chunk_id for group in run for chunk_id in group.chunk_ids],
                    section=" / ".join(sections) if sections else None,
                    text=text,
                    content_hash=hash_text(text + "\x1f" + EXTRACTOR_VERSION),
                    token_count=run_tokens(),
                )
            )
        run.clear()

    for group in groups:
        small = group.token_count < min_tokens
        joins_run = bool(run) and run_tokens() < min_tokens
        if (
            group.is_table
            or not (small or joins_run)
            or run_tokens() + group.token_count > max_tokens
        ):
            flush_run()
            if group.is_table or not small:
                merged.append(group)
                continue
        run.append(group)
        if run_tokens() >= min_tokens:
            flush_run()
    flush_run()
    return merged


def should_skip(group: ChunkGroup) -> Literal["skip", "entities_only", "full"]:
    """Decide how a group is extracted.

    Args:
        group: The section group.

    Returns:
        ``"skip"`` for a group below 40 tokens or boilerplate (a TOC/legal
        section title, or text that is mostly leader dots and page numbers),
        ``"entities_only"`` for table groups, ``"full"`` otherwise.
    """
    if group.token_count < _MIN_GROUP_TOKENS:
        return "skip"
    if _is_boilerplate(group.section, group.text):
        return "skip"
    if group.is_table:
        return "entities_only"
    return "full"


def _is_boilerplate(section: str | None, text: str) -> bool:
    """Whether the section is a TOC/legal part or mostly leader dots."""
    if section is not None and normalize_key(section) in _BOILERPLATE_SECTIONS:
        return True
    lines = text.splitlines()
    if not lines:
        return False
    dotted = sum(1 for line in lines if _DOTTED_LINE_RE.search(line))
    return dotted / len(lines) > _BOILERPLATE_DOTS_SHARE


def _system_prompt() -> str:
    """The English system prompt: role, ontology and rules.

    Rule 7 (no bare-date entities) is also enforced at resolution time by
    :func:`mynotes.graph.dates.fold_dates`, so adding it did not bump
    ``EXTRACTOR_VERSION``: cached and new extractions give the same graph.
    Rules 8 and 9 (brevity, a soft cap on the counts) only change how much the
    model writes, not what the graph holds, and did not bump it either: a
    model that pretty-printed 300-character quotes ran past the output cap on
    passages of a few hundred tokens.
    """
    entity_lines = "\n".join(f"- {name}: {_ENTITY_DEFINITIONS[name]}" for name in ENTITY_TYPES)
    relation_lines = "\n".join(
        f"- {name}: {_RELATION_DEFINITIONS[name]}" for name in RELATION_TYPES
    )
    return (
        "You extract a knowledge graph from personal project notes.\n"
        "\n"
        "Entity types:\n"
        f"{entity_lines}\n"
        "\n"
        "Relation types:\n"
        f"{relation_lines}\n"
        "\n"
        "Rules:\n"
        "1. Only facts stated in the text.\n"
        "2. Use the known entity names when they refer to the same thing.\n"
        "3. Keep names as written; add aliases for abbreviations.\n"
        "4. Use `attribute`/`value` for measurable facts (budget, deadline, size) "
        "with the exact value.\n"
        "5. `quote` = the shortest supporting excerpt.\n"
        "6. Reply only with JSON for the schema.\n"
        "7. A bare date, month or year is never an entity: put it in `attribute`/`value` "
        "of the relation (e.g. scheduled_for with attribute `deadline`).\n"
        "8. Be brief: `description` is one short sentence at most, `quote` at most 15 words, "
        "and the JSON is written compactly, without indentation or line breaks.\n"
        "9. Keep the entities and relations that matter for the notes, at most 30 entities and "
        "40 relations for a passage; skip trivia.\n"
        "\n"
        f"{_JSON_SHAPE}"
    )


def build_kg_messages(
    group: ChunkGroup,
    *,
    title: str | None,
    rel_path: str,
    doc_date: datetime | None,
    tags: Sequence[str],
    known_entities: Sequence[tuple[str, str]],
    mode: str,
) -> list[ChatMessage]:
    """Build the system and user messages for one group's extraction call.

    Args:
        group: The section group to extract from.
        title: The document title, or None.
        rel_path: The document's path relative to its source root.
        doc_date: The document date, or None.
        tags: The document's accepted tags.
        known_entities: ``(name, type)`` pairs from earlier groups of the same
            document; the first 50 are sent.
        mode: ``"full"`` or ``"entities_only"`` (from :func:`should_skip`).

    Returns:
        One system message with the ontology and rules, one user message with
        the document context and the section text.
    """
    known = list(known_entities)[:_MAX_KNOWN_ENTITIES]
    lines = [f"Title: {title or '(untitled)'}", f"Path: {rel_path}"]
    if doc_date is not None:
        lines.append(f"Date: {doc_date:%Y-%m-%d}")
    if tags:
        lines.append("Tags: " + ", ".join(tags))
    if group.section:
        lines.append(f"Section: {group.section}")
    lines.append(
        "Known entities: "
        + (", ".join(f"{name} ({entity_type})" for name, entity_type in known) or "(none)")
    )
    lines.append("Text:\n" + group.text)
    if mode == "entities_only":
        lines.append(_ENTITIES_ONLY_INSTRUCTION)
    return [
        ChatMessage(role="system", content=_system_prompt()),
        ChatMessage(role="user", content="\n\n".join(lines)),
    ]


def _chat_model_name(chat: ChatModel) -> str:
    """The chat model's name for the cache row (``"unknown"`` for fakes)."""
    cfg = getattr(chat, "cfg", None)
    model = getattr(cfg, "model", None)
    return model if isinstance(model, str) and model else "unknown"


def _cache_result(session: Session, content_hash: str, model: str, result: dict) -> None:
    """Insert one cache row, ignoring a row another worker already wrote."""
    session.execute(
        pg_insert(KgExtractionCache)
        .values(
            content_hash=content_hash,
            extractor_version=EXTRACTOR_VERSION,
            model=model,
            result=result,
        )
        .on_conflict_do_nothing(
            index_elements=["content_hash", "extractor_version"],
        )
    )


def extract_group(
    session: Session, chat: ChatModel, group: ChunkGroup, ctx: dict, mode: str
) -> Extraction:
    """Extract one section group, served from the cache whenever possible.

    On a cache hit the stored JSON is returned without an LLM call. On a miss
    the model is called once (plus the structured-output repair retry) and the
    validated result is cached. A :class:`StructuredOutputError` is logged and
    cached as ``{"error": ...}`` so the same text is never retried forever;
    an empty extraction is returned.

    Args:
        session: An open session/transaction; cache rows are written but not
            committed.
        chat: The chat model (normally ``get_chat("extract")``).
        group: The section group.
        ctx: Document context with ``title``, ``rel_path``, ``doc_date``,
            ``tags`` and ``known_entities``.
        mode: ``"full"`` or ``"entities_only"``.

    Returns:
        The cached or freshly extracted entities and relations.
    """
    cached = session.get(KgExtractionCache, (group.content_hash, EXTRACTOR_VERSION))
    if cached is not None:
        return Extraction.model_validate(cached.result)
    try:
        extraction = _extract_or_split(session, chat, group, ctx, mode, depth=0)
    except StructuredOutputError as exc:
        logger.warning(
            "kg extraction of group %s failed after the repair retry: %s",
            group.content_hash[:12],
            exc.error,
        )
        _cache_result(
            session, group.content_hash, _chat_model_name(chat), {"error": exc.error[:_ERROR_CHARS]}
        )
        return Extraction()
    _cache_result(
        session, group.content_hash, _chat_model_name(chat), extraction.model_dump(mode="json")
    )
    return extraction


def _extract_or_split(
    session: Session, chat: ChatModel, group: ChunkGroup, ctx: dict, mode: str, *, depth: int
) -> Extraction:
    """Call the model for ``group``; on a failed answer extract its parts instead.

    The parts come from :func:`split_group` and are extracted in document
    order, each one seeing the entities the earlier parts found. A part's
    result is cached under the part's own hash (so a retry never repeats a
    part that worked); the merged result is the caller's to cache.

    Raises:
        StructuredOutputError: When the answer fails and the group cannot be
            split any further (a single paragraph, or ``depth`` at the limit),
            or when every part failed too.
    """
    messages = build_kg_messages(
        group,
        title=ctx.get("title"),
        rel_path=ctx.get("rel_path") or "",
        doc_date=ctx.get("doc_date"),
        tags=ctx.get("tags") or [],
        known_entities=ctx.get("known_entities") or [],
        mode=mode,
    )
    try:
        return complete_structured(
            chat, messages, Extraction, priority="background", max_tokens=_EXTRACT_MAX_TOKENS
        )
    except StructuredOutputError as exc:
        parts = split_group(group) if depth < _MAX_SPLIT_DEPTH else [group]
        if len(parts) < 2:
            raise
        failure = exc
        logger.info(
            "kg extraction of group %s (%d tokens) failed (%s); extracting it in %d parts",
            group.content_hash[:12],
            group.token_count,
            exc.error[:120],
            len(parts),
        )
    results: list[Extraction] = []
    found: list[tuple[str, str]] = []
    for part in parts:
        part_ctx = {**ctx, "known_entities": [*(ctx.get("known_entities") or []), *found]}
        cached = session.get(KgExtractionCache, (part.content_hash, EXTRACTOR_VERSION))
        if cached is not None:
            result = Extraction.model_validate(cached.result)
        else:
            try:
                result = _extract_or_split(session, chat, part, part_ctx, mode, depth=depth + 1)
            except StructuredOutputError as exc:
                logger.warning(
                    "kg extraction of part %s of group %s failed after the repair retry: %s",
                    part.content_hash[:12],
                    group.content_hash[:12],
                    exc.error,
                )
                _cache_result(
                    session,
                    part.content_hash,
                    _chat_model_name(chat),
                    {"error": exc.error[:_ERROR_CHARS]},
                )
                continue
            _cache_result(
                session, part.content_hash, _chat_model_name(chat), result.model_dump(mode="json")
            )
        found.extend((entity.name, entity.type) for entity in result.entities)
        results.append(result)
    if not results:
        raise failure
    return merge_extractions(results)


def split_group(group: ChunkGroup) -> list[ChunkGroup]:
    """Split a group into two halves of about equal length at paragraph boundaries.

    Paragraphs are the blank-line-separated parts of the text (chunks were
    joined that way); a text without blank lines is split at line breaks.
    The halves keep the group's chunk ids, section and table flag; the
    group's token count is shared between them by character length.

    Returns:
        Two groups, or ``[group]`` when the text has no boundary to split at.
    """
    for separator in ("\n\n", "\n"):
        pieces = [piece for piece in group.text.split(separator) if piece.strip()]
        if len(pieces) < 2:
            continue
        half = sum(len(piece) for piece in pieces) / 2
        first: list[str] = []
        consumed = 0
        for piece in pieces[:-1]:
            if first and abs(consumed + len(piece) - half) > abs(consumed - half):
                break
            first.append(piece)
            consumed += len(piece)
        halves = [separator.join(first), separator.join(pieces[len(first) :])]
        first_tokens = max(1, round(group.token_count * len(halves[0]) / max(1, len(group.text))))
        tokens = [first_tokens, max(1, group.token_count - first_tokens)]
        return [
            ChunkGroup(
                chunk_ids=list(group.chunk_ids),
                section=group.section,
                text=text,
                content_hash=hash_text(text + "\x1f" + EXTRACTOR_VERSION),
                token_count=token_count,
                is_table=group.is_table,
            )
            for text, token_count in zip(halves, tokens, strict=True)
        ]
    return [group]


def merge_extractions(parts: Sequence[Extraction]) -> Extraction:
    """Merge the extractions of a split group: entities by name and type, relations by key.

    The first occurrence of an entity wins; relations that repeat (same
    source, target, type, attribute and value) are kept once. The schema's
    caps on entities and relations apply to the merged lists.
    """
    entities: dict[tuple[str, str], ExtractedEntity] = {}
    relations: dict[tuple[str, ...], ExtractedRelation] = {}
    for part in parts:
        for entity in part.entities:
            entities.setdefault((normalize_key(entity.name), entity.type), entity)
        for relation in part.relations:
            key = (
                normalize_key(relation.source),
                normalize_key(relation.target),
                relation.type,
                relation.attribute or "",
                relation.value or "",
            )
            relations.setdefault(key, relation)
    return Extraction(
        entities=list(entities.values())[:_MAX_ENTITIES],
        relations=list(relations.values())[:_MAX_RELATIONS],
    )


def load_chunk_rows(session: Session, version_id: int) -> list[ChunkRowFull]:
    """The chunks of one version in ordinal order, as extractor rows."""
    rows = session.execute(
        select(
            Chunk.id,
            Chunk.text,
            Chunk.section,
            Chunk.is_table,
            Chunk.page,
            Chunk.slide,
            Chunk.token_count,
        )
        .where(Chunk.document_version_id == version_id)
        .order_by(Chunk.ordinal)
    ).all()
    return [ChunkRowFull(*row) for row in rows]


def latest_embedded_version_ids(session: Session) -> list[int]:
    """The ids of every latest version that finished embedding, in id order."""
    return list(
        session.execute(
            select(DocumentVersion.id)
            .where(DocumentVersion.is_latest.is_(True), DocumentVersion.status == "embedded")
            .order_by(DocumentVersion.id)
        ).scalars()
    )


def extractable_groups(session: Session, version_id: int) -> list[ChunkGroup]:
    """The version's groups that the extractor would actually send to the LLM."""
    return [
        group
        for group in extraction_groups(load_chunk_rows(session, version_id))
        if should_skip(group) != "skip"
    ]


def cached_group_hashes(session: Session, content_hashes: Sequence[str]) -> set[str]:
    """Which of ``content_hashes`` already have a cached extraction result."""
    if not content_hashes:
        return set()
    hashes = list(content_hashes)
    return set(
        session.execute(
            select(KgExtractionCache.content_hash).where(
                KgExtractionCache.extractor_version == EXTRACTOR_VERSION,
                KgExtractionCache.content_hash.in_(hashes),
            )
        ).scalars()
    )


def versions_missing_extraction(session: Session) -> list[int]:
    """Latest embedded versions with at least one group not in the cache.

    Args:
        session: An open session/transaction.

    Returns:
        Version ids in id order; this is the ``graph backfill`` work list.
    """
    pending: list[int] = []
    for version_id in latest_embedded_version_ids(session):
        groups = extractable_groups(session, version_id)
        if not groups:
            continue
        if len(cached_group_hashes(session, [group.content_hash for group in groups])) < len(
            groups
        ):
            pending.append(version_id)
    return pending


def clear_error_cache(session: Session) -> int:
    """Delete the cache rows of this extractor version that hold an error instead of a result.

    A group whose extraction failed (and, for a split group, its parts) is
    cached as ``{"error": ...}`` so a job never retries it on its own. After
    an extractor fix, dropping those rows lets ``graph backfill`` queue the
    affected documents again.

    Returns:
        The number of rows deleted.
    """
    from sqlalchemy import delete

    result = session.execute(
        delete(KgExtractionCache).where(
            KgExtractionCache.extractor_version == EXTRACTOR_VERSION,
            KgExtractionCache.result.has_key("error"),
        )
    )
    return int(result.rowcount or 0)


def groups_overview(session: Session) -> dict[str, int]:
    """Count the extractable groups of all latest embedded versions and their cache state.

    Args:
        session: An open session/transaction.

    Returns:
        ``{"versions", "total", "cached", "pending"}``; pending is an estimate
        that ignores versions appearing while the command runs.
    """
    total = 0
    cached = 0
    versions = 0
    for version_id in latest_embedded_version_ids(session):
        groups = extractable_groups(session, version_id)
        versions += 1
        total += len(groups)
        if groups:
            cached += len(cached_group_hashes(session, [group.content_hash for group in groups]))
    return {"versions": versions, "total": total, "cached": cached, "pending": total - cached}
