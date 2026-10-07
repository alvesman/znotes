"""Context assembly for the answer pipeline (PLAN.md §9 step 6).

Context assembler: the best chunks of the first
``retrieval.max_docs`` documents are selected, the strongest hits are expanded
to their section (or page) through a :class:`ChunkStore`, snippets are grouped
under attribution headers per (document, page-or-slide) and everything is
fitted into ``retrieval.context_tokens``. Registered extra-block providers
(version notes, graph facts) append their blocks at the end under their own
heading. The bundle carries citations in order of first appearance so the
answer can use ``[n]`` markers.
"""

from __future__ import annotations

import itertools
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from mynotes.chunk.tokenizer import TokenCounter, get_token_counter
from mynotes.config import get_settings
from mynotes.db.models import Chunk
from mynotes.retrieval.types import SearchHit

if TYPE_CHECKING:
    from mynotes.retrieval.pipeline import RetrievalResult

logger = logging.getLogger(__name__)

_EXPANSION_TOP_HITS = 2
"""Hits whose section or page is expanded with store chunks."""

_EXPANSION_TOKENS = 600
"""Token budget per expanded hit."""

_MIN_EXPANSION_TOKENS = 15
"""Chunks under this many tokens are skipped."""

_EXTRA_BLOCK_SHARE = 0.2
"""Share of ``context_tokens`` the extra blocks may consume together."""

_PREFIX_DEDUP_CHARS = 80
"""Snippet dedup: first whitespace-collapsed characters of the text."""

_SENTENCE_BOUNDARY = re.compile(r"[.!?](?=\s|\Z)")


@dataclass
class Citation:
    """One citation of the assembled context.

    Attributes:
        n: The 1-based marker used in the answer (``[n]``).
        document_id: The cited document id.
        title: The document title, or None.
        path: The file path relative to its source root.
        page: The cited PDF page, or None.
        slide: The cited PPTX slide, or None.
        section: The cited section label, or None.
        doc_date: The document date, or None.
        is_current: Whether the document is ``current`` and its latest version.
        format: The document format (``md``, ``pdf``, ...), or None when the
            hit carried none.
    """

    n: int
    document_id: int
    title: str | None
    path: str
    page: int | None
    slide: int | None
    section: str | None
    doc_date: datetime | None
    is_current: bool
    format: str | None = None


@dataclass
class ContextBundle:
    """The assembled LLM context.

    Attributes:
        text: The context text passed to the model.
        citations: The citations in order of first appearance in ``text``.
        tokens: Token count of ``text``.
        has_table: Whether the context carries tabular content.
        extra_blocks: The extra-block texts already included in ``text``.
    """

    text: str
    citations: list[Citation]
    tokens: int
    has_table: bool
    extra_blocks: list[str] = field(default_factory=list)


@dataclass
class ChunkRow:
    """One expansion row of a :class:`ChunkStore` query."""

    ordinal: int
    text: str
    page: int | None
    slide: int | None
    section: str | None
    token_count: int


class ChunkStore(Protocol):
    """Read access to the chunk table for section and page expansion."""

    def section_chunks(self, document_id: int, version_id: int, section: str) -> list[ChunkRow]:
        """All chunks of one document version's section, ordered by ordinal."""
        ...

    def page_chunks(
        self, document_id: int, version_id: int, page: int | None, slide: int | None
    ) -> list[ChunkRow]:
        """All chunks of one document version's page (or slide), ordered by ordinal."""
        ...


class ExtraBlockProvider(Protocol):
    """A pluggable block appended to the context (version notes, graph facts).

    Attributes:
        name: The provider name, also the heading: ``version_notes`` renders
            as ``### Version notes``.
    """

    name: str

    def block(self, result: RetrievalResult, selected: Sequence[SearchHit]) -> str | None:
        """Return this provider's block text, or None to contribute nothing."""
        ...


_EXTRA_PROVIDERS: list[ExtraBlockProvider] = []


def register_extra_block(provider: ExtraBlockProvider) -> None:
    """Add an extra-block provider, replacing one with the same name."""
    for position, existing in enumerate(_EXTRA_PROVIDERS):
        if existing.name == provider.name:
            _EXTRA_PROVIDERS[position] = provider
            return
    _EXTRA_PROVIDERS.append(provider)


def unregister_extra_block(name: str) -> None:
    """Remove the provider registered under ``name`` (no-op when absent)."""
    _EXTRA_PROVIDERS[:] = [provider for provider in _EXTRA_PROVIDERS if provider.name != name]


@dataclass
class _Snippet:
    """One rendered text piece with its own page attribution."""

    document_id: int
    text: str
    page: int | None
    slide: int | None
    section: str | None
    ordinal: int | None
    tokens: int = 0


@dataclass
class _Group:
    """Snippets of one (document, page-or-slide) under a shared header."""

    document_id: int
    page: int | None
    slide: int | None
    hit: SearchHit
    n: int = 0
    header_tokens: int = 0
    snippets: list[_Snippet] = field(default_factory=list)


def assemble_context(
    result: RetrievalResult,
    *,
    store: ChunkStore | None = None,
    counter: TokenCounter | None = None,
) -> ContextBundle:
    """Assemble the LLM context from a retrieval result.

    The hit order is trusted (reranked or fused). At most
    ``retrieval.max_docs`` distinct documents and ``retrieval.max_chunks``
    hits are used; the first two selected hits are expanded to their section
    or page through ``store`` when one is given. The total is fitted into
    ``retrieval.context_tokens``, with extra blocks capped at 20% of it.

    Args:
        result: The pipeline result whose hits feed the context.
        store: Expansion source; without one only the hit texts are used.
        counter: Token counter; defaults to the process-wide one.

    Returns:
        The :class:`ContextBundle` for the answer prompt.
    """
    settings = get_settings()
    counter = counter if counter is not None else get_token_counter()
    if not result.hits:
        return ContextBundle(text="", citations=[], tokens=0, has_table=False, extra_blocks=[])

    allowed = _allowed_documents(
        result.hits,
        settings.retrieval.max_docs,
        extra_max=settings.retrieval.extra_docs_max,
        min_ratio=settings.retrieval.extra_docs_min_ratio,
    )
    selected = _select_hits(result.hits, allowed, settings.retrieval.max_chunks)
    extra_blocks, extra_tokens = _collect_extra_blocks(
        result, selected, counter, settings.retrieval.context_tokens
    )
    snippets = _collect_snippets(selected, store)
    groups = _group_snippets(snippets, result.hits)

    _fit_budget(groups, settings.retrieval.context_tokens - extra_tokens, counter)
    citations = [_citation(group, n) for n, group in enumerate(groups, start=1)]
    text = _render(groups, extra_blocks)
    return ContextBundle(
        text=text,
        citations=citations,
        tokens=counter.count(text) if text else 0,
        has_table=any(hit.is_table for hit in selected) or _has_markdown_table(text),
        extra_blocks=[block for _, block in extra_blocks],
    )


def _hit_score(hit: SearchHit) -> float:
    """The reranker score, or the fused score when the hits were not reranked."""
    return float(hit.rerank_score if hit.rerank_score is not None else hit.score)


def _allowed_documents(
    hits: Sequence[SearchHit], max_docs: int, *, extra_max: int = 0, min_ratio: float = 1.0
) -> set[int]:
    """The first ``max_docs`` distinct document ids in hit order, plus close runners-up.

    The score-gap rule (``retrieval.extra_docs_min_ratio``): up to ``extra_max``
    further documents are admitted, in hit order, while the best score of each
    is at least ``min_ratio`` times the top hit's score; the first document
    below the ratio ends the extension. ``min_ratio`` 1.0 keeps the flat cap.
    """
    allowed: list[int] = []
    for hit in hits:
        if hit.document_id not in allowed and len(allowed) < max_docs:
            allowed.append(hit.document_id)
    if extra_max <= 0 or min_ratio >= 1.0 or not hits:
        return set(allowed)
    top = _hit_score(hits[0])
    if top <= 0:
        return set(allowed)
    for hit in hits:
        if hit.document_id in allowed:
            continue
        if len(allowed) >= max_docs + extra_max or _hit_score(hit) < min_ratio * top:
            break
        allowed.append(hit.document_id)
    return set(allowed)


def _select_hits(hits: Sequence[SearchHit], allowed: set[int], max_chunks: int) -> list[SearchHit]:
    """The first ``max_chunks`` qualifying hits from the allowed documents.

    Table hits always qualify; other hits must carry text.
    """
    selected: list[SearchHit] = []
    for hit in hits:
        if len(selected) >= max_chunks:
            break
        if hit.document_id not in allowed:
            continue
        if hit.is_table or hit.text.strip():
            selected.append(hit)
    return selected


def _collect_snippets(selected: Sequence[SearchHit], store: ChunkStore | None) -> list[_Snippet]:
    """Turn the selected hits into deduplicated snippets, expanding the top two.

    Dedup happens by ``(document_id, ordinal)`` for expansion rows and by the
    first 80 whitespace-collapsed characters for everything, so overlapping
    expansion text never repeats the hit text.
    """
    snippets: list[_Snippet] = []
    seen_ordinals: set[tuple[int, int]] = set()
    seen_prefixes: set[str] = set()

    def add(
        document_id: int,
        text: str,
        page: int | None,
        slide: int | None,
        section: str | None,
        ordinal: int | None,
    ) -> None:
        stripped = text.strip()
        if not stripped:
            return
        prefix = re.sub(r"\s+", " ", stripped)[:_PREFIX_DEDUP_CHARS]
        if prefix in seen_prefixes:
            return
        if ordinal is not None:
            key = (document_id, ordinal)
            if key in seen_ordinals:
                return
            seen_ordinals.add(key)
        seen_prefixes.add(prefix)
        snippets.append(
            _Snippet(
                document_id=document_id,
                text=stripped,
                page=page,
                slide=slide,
                section=section,
                ordinal=ordinal,
            )
        )

    for position, hit in enumerate(selected):
        add(hit.document_id, hit.text, hit.page, hit.slide, hit.section, None)
        if store is None or position >= _EXPANSION_TOP_HITS:
            continue
        for row in _expand_hit(hit, store):
            add(hit.document_id, row.text, row.page, row.slide, row.section, row.ordinal)
    return snippets


def _expand_hit(hit: SearchHit, store: ChunkStore) -> list[ChunkRow]:
    """Expand one hit to its section, falling back to its page.

    A section holding nothing but the seed (heading-only chunks), or one the
    budget cuts back to just the seed, falls through to page expansion, like
    section-expansion flow.
    """
    if hit.section:
        rows = store.section_chunks(hit.document_id, hit.version_id, hit.section)
        if rows:
            index = _find_seed_index(rows, hit)
            expanded = _expand_outward(rows, index if index is not None else 0)
            if len(expanded) > 1:
                return expanded
    rows = store.page_chunks(hit.document_id, hit.version_id, hit.page, hit.slide)
    index = _find_seed_index(rows, hit)
    if index is None:
        return []
    return _expand_outward(rows, index)


def _find_seed_index(rows: Sequence[ChunkRow], hit: SearchHit) -> int | None:
    """Locate the hit inside the store rows by text, then by page and slide."""
    for index, row in enumerate(rows):
        if row.text == hit.text:
            return index
    for index, row in enumerate(rows):
        if row.page == hit.page and row.slide == hit.slide:
            return index
    return None


def _expand_outward(
    rows: Sequence[ChunkRow], seed_index: int, max_tokens: int = _EXPANSION_TOKENS
) -> list[ChunkRow]:
    """Rows around ``seed_index``, seed first, then neighbours by distance.

    Outward expansion: the seed is always included, chunks
    under :data:`_MIN_EXPANSION_TOKENS` are skipped and a chunk larger than
    the remaining budget ends the expansion.
    """
    if not rows or not 0 <= seed_index < len(rows):
        return []
    expanded = [rows[seed_index]]
    budget = max_tokens - max(rows[seed_index].token_count, 0)
    for distance in range(1, len(rows)):
        for index in (seed_index - distance, seed_index + distance):
            if not 0 <= index < len(rows):
                continue
            row = rows[index]
            if row.token_count < _MIN_EXPANSION_TOKENS:
                continue
            if row.token_count > budget:
                return expanded
            expanded.append(row)
            budget -= row.token_count
    return expanded


def _group_snippets(snippets: Sequence[_Snippet], hits: Sequence[SearchHit]) -> list[_Group]:
    """Group snippets per (document, page-or-slide) in first-appearance order."""
    doc_hits: dict[int, SearchHit] = {}
    for hit in hits:
        doc_hits.setdefault(hit.document_id, hit)

    groups: list[_Group] = []
    by_key: dict[tuple[int, int | None, int | None], _Group] = {}
    for snippet in snippets:
        key = (snippet.document_id, snippet.page, snippet.slide)
        group = by_key.get(key)
        if group is None:
            group = _Group(
                document_id=snippet.document_id,
                page=snippet.page,
                slide=snippet.slide,
                hit=doc_hits[snippet.document_id],
            )
            by_key[key] = group
            groups.append(group)
        group.snippets.append(snippet)
    for n, group in enumerate(groups, start=1):
        group.n = n
    return groups


def _collect_extra_blocks(
    result: RetrievalResult,
    selected: Sequence[SearchHit],
    counter: TokenCounter,
    context_tokens: int,
) -> tuple[list[tuple[str, str]], int]:
    """Collect the registered providers' blocks within the extra-block budget.

    Blocks are truncated at a sentence boundary when they exceed the remaining
    share; a failing provider is logged and skipped.

    Returns:
        The ``(name, block)`` pairs and the total tokens they consume.
    """
    cap = int(context_tokens * _EXTRA_BLOCK_SHARE)
    collected: list[tuple[str, str]] = []
    used = 0
    for provider in list(_EXTRA_PROVIDERS):
        try:
            block = provider.block(result, selected)
        except Exception:
            logger.warning("extra block %r failed; skipping it", provider.name, exc_info=True)
            continue
        if not block or not block.strip():
            continue
        allowance = cap - used
        if allowance <= 0:
            logger.info("extra block %r skipped: extra-block budget exhausted", provider.name)
            continue
        block = block.strip()
        tokens = counter.count(block)
        if tokens > allowance:
            block = _truncate_at_sentence(block, allowance, counter)
            tokens = counter.count(block)
        if not block:
            continue
        collected.append((provider.name, block))
        used += tokens
    return collected, used


def _fit_budget(groups: list[_Group], budget: int, counter: TokenCounter) -> None:
    """Drop and truncate snippets in place until the context fits ``budget``.

    Whole snippets are dropped from the end, lowest-ranked groups first; the
    top hit's snippet is never dropped. When only it remains and the budget is
    still exceeded, it is truncated at a sentence boundary.
    """
    for group in groups:
        group.header_tokens = counter.count(_header_line(group))
        for snippet in group.snippets:
            snippet.tokens = counter.count(snippet.text)

    current = sum(
        group.header_tokens + sum(snippet.tokens for snippet in group.snippets) for group in groups
    )
    while current > budget:
        victim_group: _Group | None = None
        victim: _Snippet | None = None
        for group in reversed(groups):
            droppable = group.snippets[1:] if group is groups[0] else group.snippets
            if droppable:
                victim_group, victim = group, droppable[-1]
                break
        if victim is None or victim_group is None:
            break
        victim_group.snippets.remove(victim)
        current -= victim.tokens
        if not victim_group.snippets:
            current -= victim_group.header_tokens
            groups.remove(victim_group)

    if current > budget and groups and groups[0].snippets:
        last = groups[0].snippets[-1]
        allowance = budget - (current - last.tokens)
        if allowance > 0:
            last.text = _truncate_at_sentence(last.text, allowance, counter)


def _citation(group: _Group, n: int) -> Citation:
    """Build the citation of one surviving group."""
    section = next((snippet.section for snippet in group.snippets if snippet.section), None)
    return Citation(
        n=n,
        document_id=group.document_id,
        title=group.hit.title,
        path=group.hit.path,
        page=group.page,
        slide=group.slide,
        section=section,
        doc_date=group.hit.doc_date,
        is_current=group.hit.doc_status == "current" and group.hit.is_latest,
        format=group.hit.format,
    )


def _header_line(group: _Group) -> str:
    """The attribution header of one group: ``[n] title — path — p./slide — date``."""
    hit = group.hit
    parts: list[str] = []
    if hit.title:
        parts.append(hit.title)
    parts.append(hit.path)
    if group.page is not None:
        parts.append(f"p.{group.page}")
    elif group.slide is not None:
        parts.append(f"slide {group.slide}")
    if hit.doc_date is not None:
        parts.append(f"{hit.doc_date:%Y-%m-%d}")
    if hit.doc_status != "current" or not hit.is_latest:
        parts.append("historical")
    return f"[{group.n}] " + " — ".join(parts)


def _render(groups: Sequence[_Group], extra_blocks: Sequence[tuple[str, str]]) -> str:
    """Render the groups and the extra blocks into the final context text."""
    parts = [
        "\n".join([_header_line(group), *[snippet.text for snippet in group.snippets]])
        for group in groups
    ]
    for name, block in extra_blocks:
        heading = f"### {name.replace('_', ' ').capitalize()}"
        parts.append(f"{heading}\n{block}")
    return "\n\n".join(parts)


def _has_markdown_table(text: str) -> bool:
    """Whether the text holds a Markdown table: a ``|`` line over a ``|---|`` line."""
    lines = text.splitlines()
    for above, below in itertools.pairwise(lines):
        if "|" not in above or "|" not in below:
            continue
        if set(below) <= {"|", "-", ":", " ", "\t"} and "-" in below:
            return True
    return False


def _truncate_at_sentence(text: str, max_tokens: int, counter: TokenCounter) -> str:
    """The longest prefix of ``text`` ending at a sentence boundary that fits.

    Falls back to a word cut when even the first sentence does not fit.
    """
    if max_tokens <= 0:
        return ""
    if counter.count(text) <= max_tokens:
        return text
    stripped = text.strip()
    boundaries = [match.end() for match in _SENTENCE_BOUNDARY.finditer(stripped)]
    for end in reversed(boundaries):
        candidate = stripped[:end].rstrip()
        if candidate and counter.count(candidate) <= max_tokens:
            return candidate
    return _cut_words(stripped, max_tokens, counter)


def _cut_words(text: str, max_tokens: int, counter: TokenCounter) -> str:
    """Hard-cut ``text`` to the longest whole-word prefix within ``max_tokens``."""
    words = text.split()
    low, high = 0, len(words)
    while low < high:
        middle = (low + high + 1) // 2
        if counter.count(" ".join(words[:middle])) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return " ".join(words[:low])


class DbChunkStore:
    """:class:`ChunkStore` reading expansion rows from the ``chunk`` table."""

    def __init__(self, session: Session) -> None:
        """Bind the store to one SQLAlchemy session.

        Args:
            session: An open session; queries run inside its transaction.
        """
        self._session = session

    def section_chunks(self, document_id: int, version_id: int, section: str) -> list[ChunkRow]:
        """All chunks of ``section`` across pages, ordered by ordinal.

        Args:
            document_id: The owning document id.
            version_id: The owning document version id.
            section: The exact section label.

        Returns:
            The section's chunks, ordered by ordinal.
        """
        return self._fetch(
            Chunk.document_id == document_id,
            Chunk.document_version_id == version_id,
            Chunk.section == section,
        )

    def page_chunks(
        self, document_id: int, version_id: int, page: int | None, slide: int | None
    ) -> list[ChunkRow]:
        """All chunks of one page or slide (or the page-less ones) by ordinal.

        Args:
            document_id: The owning document id.
            version_id: The owning document version id.
            page: The PDF page, or None for page-less documents.
            slide: The PPTX slide, or None.

        Returns:
            The page's chunks, ordered by ordinal.
        """
        return self._fetch(
            Chunk.document_id == document_id,
            Chunk.document_version_id == version_id,
            Chunk.page.is_not_distinct_from(page),
            Chunk.slide.is_not_distinct_from(slide),
        )

    def _fetch(self, *criteria: object) -> list[ChunkRow]:
        """Run one projection query ordered by ordinal."""
        statement = (
            select(
                Chunk.ordinal,
                Chunk.text,
                Chunk.page,
                Chunk.slide,
                Chunk.section,
                Chunk.token_count,
            )
            .where(*criteria)
            .order_by(Chunk.ordinal)
        )
        return [ChunkRow(*row) for row in self._session.execute(statement).all()]
