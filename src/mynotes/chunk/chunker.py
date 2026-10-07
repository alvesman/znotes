"""Heading-aware, token-budgeted chunking of extraction IR (PLAN.md §7.3).

A heading-aware Markdown chunker with context headers,
switched from word counts to real token counts and driven by :class:`IRBlock`s
instead of raw Markdown: a heading stack carries across pages, tables and code
blocks become their own chunks, and a trailing short paragraph overlaps into
the next chunk.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Literal

from mynotes.chunk.tokenizer import TokenCounter
from mynotes.extract.ir import IRBlock, IRDocument
from mynotes.scan.hashing import hash_text
from mynotes.text.language import detect_language

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

MIN_CHUNK_CHARS = 15
"""Chunks whose stripped text is shorter than this are dropped, except
heading-only chunks (they matter for exact-identifier queries)."""

_OVERLAP_KINDS = frozenset({"paragraph", "list"})
"""Block kinds eligible to overlap into the next chunk."""


@dataclass
class ChunkDraft:
    """One chunk of a document version, before it is persisted.

    Attributes:
        ordinal: Position of the chunk within the version (0-based, without
            gaps even when chunks were dropped).
        text: Chunk text as the LLM and the user see it, including the
            heading line when a section starts in it.
        embed_text: Contextual header (title, path, section, tags) prefixed
            to ``text``; this is what gets embedded (PLAN.md §5.1).
        content_hash: blake3 over the whitespace-normalized text and the
            heading path; identifies reused chunks across versions.
        page: PDF page the chunk starts on (``page_kind="page"``).
        slide: PPTX slide the chunk starts on (``page_kind="slide"``).
        section: Innermost active heading text, or None.
        heading_path: Heading stack texts, outermost first.
        is_table: True when the chunk holds (part of) a table.
        token_count: Token count of ``text``.
        language: Detected language of the chunk, falling back to the
            document language.
    """

    ordinal: int
    text: str
    embed_text: str = ""
    content_hash: str = ""
    page: int | None = None
    slide: int | None = None
    section: str | None = None
    heading_path: list[str] = field(default_factory=list)
    is_table: bool = False
    token_count: int = 0
    language: str | None = None


@dataclass(frozen=True)
class ChunkingParams:
    """Token budgets for chunking (PLAN.md §7.3).

    Attributes:
        target_tokens: Chunks are flushed once adding a block would exceed
            this.
        max_tokens: Hard ceiling; only single oversized rows/lines/sentences
            may push a chunk past it.
        overlap_tokens: A trailing paragraph/list block of at most this many
            tokens is repeated at the start of the next chunk.
    """

    target_tokens: int = 350
    max_tokens: int = 512
    overlap_tokens: int = 80


def chunk_content_hash(text: str, heading_path: Sequence[str]) -> str:
    """Hash a chunk's text together with its heading path.

    Args:
        text: The chunk text; whitespace is normalized before hashing so
            formatting-only changes do not invalidate embeddings.
        heading_path: The chunk's heading stack texts.

    Returns:
        The blake3 hex digest used for chunk reuse across versions.
    """
    normalized = " ".join(text.split())
    return hash_text(normalized + "\x1f" + " > ".join(heading_path))


def build_embed_text(
    *,
    title: str | None,
    rel_path: str,
    heading_path: Sequence[str],
    tags: Sequence[str],
    text: str,
) -> str:
    """Prefix the contextual header to a chunk text.

    Args:
        title: Document title, omitted when None.
        rel_path: POSIX relative path of the document; its parent folder is
            included as ``Path:``, omitted at the corpus root.
        heading_path: Heading stack texts, joined with ``" > "``.
        tags: Document tags, joined with ``", "``.
        text: The chunk text.

    Returns:
        ``Title: … | Path: … | Section: … | Tags: …`` (empty parts skipped)
        followed by a blank line and ``text``; just ``text`` when no header
        part is present.
    """
    parts: list[str] = []
    if title:
        parts.append(f"Title: {title}")
    folder = str(PurePosixPath(rel_path).parent)
    if folder and folder != ".":
        parts.append(f"Path: {folder}")
    if heading_path:
        parts.append("Section: " + " > ".join(heading_path))
    if tags:
        parts.append("Tags: " + ", ".join(tags))
    header = " | ".join(parts)
    return f"{header}\n\n{text}" if header else text


@dataclass
class _Carry:
    """A block carried over as overlap into the next chunk."""

    text: str
    kind: str
    tokens: int
    page: int | None


class _Chunker:
    """Single-document chunking state machine."""

    def __init__(
        self,
        ir: IRDocument,
        *,
        rel_path: str,
        tags: Sequence[str],
        params: ChunkingParams,
        counter: TokenCounter,
        page_kind: Literal["page", "slide"] | None,
    ) -> None:
        self._ir = ir
        self._rel_path = rel_path
        self._tags = list(tags)
        self._params = params
        self._counter = counter
        self._page_kind = page_kind
        self._drafts: list[ChunkDraft] = []
        self._heading_stack: list[tuple[str, int]] = []
        self._acc_texts: list[str] = []
        self._acc_kinds: list[str] = []
        self._acc_tokens: list[int] = []
        self._total_tokens = 0
        self._chunk_page: int | None = None
        self._pending_heading: tuple[str, int | None] | None = None

    def run(self) -> list[ChunkDraft]:
        """Chunk every IR block and return the drafts."""
        for block in self._ir.blocks:
            if block.kind == "page_break":
                continue
            if block.kind == "heading":
                self._handle_heading(block)
            elif block.kind == "table":
                self._handle_table(block)
            elif block.kind == "code":
                self._handle_code(block)
            else:
                self._handle_content(block)
        self._flush()
        if self._pending_heading is not None:
            self._emit_heading_only()
        return self._drafts

    def _count(self, text: str) -> int:
        return self._counter.count(text)

    def _handle_heading(self, block: IRBlock) -> None:
        if self._acc_texts:
            self._flush()
        elif self._pending_heading is not None:
            self._emit_heading_only()
        level = block.level or 1
        while self._heading_stack and self._heading_stack[-1][1] >= level:
            self._heading_stack.pop()
        self._heading_stack.append((block.text, level))
        self._pending_heading = ("#" * level + " " + block.text, block.page)

    def _handle_table(self, block: IRBlock) -> None:
        self._flush()
        lines = [line for line in block.text.splitlines() if line.strip()]
        if self._count(block.text) <= self._params.max_tokens or len(lines) <= 2:
            self._emit(block.text, page=block.page, is_table=True)
            return
        header = lines[:2]
        header_tokens = self._count("\n".join(header))
        part: list[str] = list(header)
        part_tokens = header_tokens
        for row in lines[2:]:
            row_tokens = self._count(row)
            if part_tokens + row_tokens > self._params.max_tokens and len(part) > len(header):
                self._emit("\n".join(part), page=block.page, is_table=True)
                part = [*header, row]
                part_tokens = header_tokens + row_tokens
            else:
                part.append(row)
                part_tokens += row_tokens
        if part:
            self._emit("\n".join(part), page=block.page, is_table=True)

    def _handle_code(self, block: IRBlock) -> None:
        self._flush()
        if self._count(block.text) <= self._params.max_tokens:
            self._emit(block.text, page=block.page)
            return
        part: list[str] = []
        part_tokens = 0
        for line in block.text.splitlines():
            line_tokens = self._count(line)
            if part_tokens + line_tokens > self._params.max_tokens and part:
                self._emit("\n".join(part), page=block.page)
                part = [line]
                part_tokens = line_tokens
            else:
                part.append(line)
                part_tokens += line_tokens
        if part:
            self._emit("\n".join(part), page=block.page)

    def _handle_content(self, block: IRBlock) -> None:
        if self._acc_texts and block.page != self._chunk_page:
            self._flush()
        block_tokens = self._count(block.text)
        if block_tokens > self._params.max_tokens:
            pieces = self._split_oversized(block.text)
        else:
            pieces = [(block.text, block_tokens)]
        for piece, piece_tokens in pieces:
            if self._acc_texts and self._total_tokens + piece_tokens > self._params.target_tokens:
                carry = self._flush()
                if carry is not None:
                    self._acc_texts = [carry.text]
                    self._acc_kinds = [carry.kind]
                    self._acc_tokens = [carry.tokens]
                    self._total_tokens = carry.tokens
                    self._chunk_page = carry.page
            self._acc_texts.append(piece)
            self._acc_kinds.append(block.kind)
            self._acc_tokens.append(piece_tokens)
            self._total_tokens += piece_tokens
            if self._chunk_page is None:
                self._chunk_page = block.page

    def _split_oversized(self, text: str) -> list[tuple[str, int]]:
        sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        pieces: list[tuple[str, int]] = []
        group: list[str] = []
        group_tokens = 0

        def close_group() -> None:
            nonlocal group, group_tokens
            if group:
                joined = " ".join(group)
                pieces.append((joined, self._count(joined)))
                group = []
                group_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count(sentence)
            if sentence_tokens > self._params.max_tokens:
                close_group()
                for part in self._split_words(sentence):
                    pieces.append((part, self._count(part)))
                continue
            if group and group_tokens + sentence_tokens > self._params.target_tokens:
                close_group()
            group.append(sentence)
            group_tokens += sentence_tokens
        close_group()
        return pieces

    def _split_words(self, sentence: str) -> list[str]:
        parts: list[str] = []
        part: list[str] = []
        part_tokens = 0
        for word in sentence.split():
            word_tokens = self._count(word)
            if part and part_tokens + word_tokens > self._params.max_tokens:
                parts.append(" ".join(part))
                part = [word]
                part_tokens = word_tokens
            else:
                part.append(word)
                part_tokens += word_tokens
        if part:
            parts.append(" ".join(part))
        return parts

    def _flush(self) -> _Carry | None:
        if not self._acc_texts:
            return None
        text = "\n\n".join(self._acc_texts)
        if self._pending_heading is not None:
            text = self._pending_heading[0] + "\n\n" + text
            self._pending_heading = None
        carry: _Carry | None = None
        if self._params.overlap_tokens > 0:
            for kind, block_text, block_tokens in zip(
                reversed(self._acc_kinds),
                reversed(self._acc_texts),
                reversed(self._acc_tokens),
                strict=True,
            ):
                if kind in _OVERLAP_KINDS:
                    if block_tokens <= self._params.overlap_tokens:
                        carry = _Carry(
                            text=block_text,
                            kind=kind,
                            tokens=block_tokens,
                            page=self._chunk_page,
                        )
                    break
        self._emit(text, page=self._chunk_page)
        self._acc_texts = []
        self._acc_kinds = []
        self._acc_tokens = []
        self._total_tokens = 0
        self._chunk_page = None
        return carry

    def _emit_heading_only(self) -> None:
        pending = self._pending_heading
        assert pending is not None
        self._pending_heading = None
        self._emit(pending[0], page=pending[1], heading_only=True)

    def _emit(
        self,
        text: str,
        *,
        page: int | None,
        is_table: bool = False,
        heading_only: bool = False,
    ) -> None:
        if len(text.strip()) < MIN_CHUNK_CHARS and not heading_only:
            return
        heading_path = [heading for heading, _ in self._heading_stack]
        section = self._heading_stack[-1][0] if self._heading_stack else None
        if self._page_kind == "slide":
            draft_page: int | None = None
            draft_slide: int | None = page
        elif self._page_kind == "page":
            draft_page = page
            draft_slide = None
        else:
            draft_page = None
            draft_slide = None
        self._drafts.append(
            ChunkDraft(
                ordinal=len(self._drafts),
                text=text,
                embed_text=build_embed_text(
                    title=self._ir.title,
                    rel_path=self._rel_path,
                    heading_path=heading_path,
                    tags=self._tags,
                    text=text,
                ),
                content_hash=chunk_content_hash(text, heading_path),
                page=draft_page,
                slide=draft_slide,
                section=section,
                heading_path=heading_path,
                is_table=is_table,
                token_count=self._count(text),
                language=detect_language(text) or self._ir.language,
            )
        )


def chunk_document(
    ir: IRDocument,
    *,
    rel_path: str,
    tags: Sequence[str],
    params: ChunkingParams,
    counter: TokenCounter,
    page_kind: Literal["page", "slide"] | None,
) -> list[ChunkDraft]:
    """Turn an :class:`IRDocument` into token-budgeted, heading-aware chunks.

    Args:
        ir: The extraction IR of one document version.
        rel_path: POSIX relative path of the document (used for the
            ``embed_text`` folder).
        tags: Document tags for the ``embed_text`` header.
        params: Token budgets.
        counter: Token counter used for all budget decisions.
        page_kind: Where block page numbers go; None for md/txt.

    Returns:
        The chunks in document order; tiny chunks are dropped except
        heading-only chunks.
    """
    return _Chunker(
        ir,
        rel_path=rel_path,
        tags=tags,
        params=params,
        counter=counter,
        page_kind=page_kind,
    ).run()
