"""Unit tests for the heading-aware chunker and token counters (task 19)."""

from __future__ import annotations

from typing import Literal

import pytest

from mynotes.chunk import tokenizer as tokenizer_module
from mynotes.chunk.chunker import (
    ChunkDraft,
    ChunkingParams,
    build_embed_text,
    chunk_content_hash,
    chunk_document,
)
from mynotes.chunk.tokenizer import WhitespaceCounter, get_token_counter
from mynotes.extract.ir import IRBlock, IRDocument
from mynotes.scan.hashing import hash_text

PARAMS = ChunkingParams(target_tokens=40, max_tokens=60, overlap_tokens=15)


def _para(words: int, page: int | None = None, start: int = 0) -> IRBlock:
    text = " ".join(f"word{start + i}" for i in range(words))
    return IRBlock(kind="paragraph", text=text, page=page)


def _chunk(
    ir: IRDocument,
    *,
    rel_path: str = "docs/notes.md",
    tags: list[str] | None = None,
    page_kind: Literal["page", "slide"] | None = None,
    params: ChunkingParams = PARAMS,
) -> list[ChunkDraft]:
    return chunk_document(
        ir,
        rel_path=rel_path,
        tags=tags if tags is not None else [],
        params=params,
        counter=WhitespaceCounter(),
        page_kind=page_kind,
    )


class TestWhitespaceCounter:
    def test_counts_words_with_inflation(self) -> None:
        counter = WhitespaceCounter()
        assert counter.count("") == 0
        assert counter.count("one two three") == 4
        assert counter.count("one two three four five") == 7

    def test_get_token_counter_is_cached(self) -> None:
        assert get_token_counter() is get_token_counter()

    def test_set_and_reset_token_counter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Dummy:
            def count(self, text: str) -> int:
                return 42

        dummy = Dummy()
        tokenizer_module.set_token_counter(dummy)
        assert get_token_counter() is dummy
        monkeypatch.setattr(tokenizer_module, "_resolve_token_counter", WhitespaceCounter)
        tokenizer_module.reset_token_counter()
        resolved = get_token_counter()
        assert resolved is not dummy
        assert isinstance(resolved, WhitespaceCounter)


class TestSections:
    def test_headings_split_chunks_and_build_stack(self) -> None:
        ir = IRDocument(
            blocks=[
                IRBlock(kind="heading", text="A", level=1),
                _para(20),
                IRBlock(kind="heading", text="B", level=2),
                _para(20),
            ]
        )
        drafts = _chunk(ir)
        assert len(drafts) == 2
        assert drafts[0].text.startswith("# A")
        assert drafts[0].section == "A"
        assert drafts[0].heading_path == ["A"]
        assert drafts[1].section == "B"
        assert drafts[1].heading_path == ["A", "B"]

    def test_token_count_and_content_hash_match_spec(self) -> None:
        ir = IRDocument(blocks=[IRBlock(kind="heading", text="A", level=1), _para(20)])
        (draft,) = _chunk(ir)
        assert draft.token_count == WhitespaceCounter().count(draft.text)
        assert draft.content_hash == chunk_content_hash(draft.text, ["A"])

    def test_content_hash_normalizes_whitespace(self) -> None:
        assert chunk_content_hash("a\n b\t c", ["A"]) == hash_text("a b c\x1fA")


class TestPages:
    def test_heading_stack_carries_across_pages(self) -> None:
        ir = IRDocument(
            blocks=[
                IRBlock(kind="heading", text="B", level=2, page=1),
                _para(20, page=1),
                _para(20, page=2),
            ]
        )
        drafts = _chunk(ir, page_kind="page")
        assert len(drafts) == 2
        assert drafts[0].page == 1
        assert drafts[0].text.startswith("## B")
        assert drafts[1].page == 2
        assert drafts[1].section == "B"
        assert drafts[1].heading_path == ["B"]
        assert "#" not in drafts[1].text

    def test_slides_stored_in_slide_field(self) -> None:
        ir = IRDocument(blocks=[_para(20, page=1), _para(20, page=2), _para(20, page=3)])
        drafts = _chunk(ir, page_kind="slide")
        assert [draft.slide for draft in drafts] == [1, 2, 3]
        assert all(draft.page is None for draft in drafts)

    def test_no_page_kind_leaves_both_none(self) -> None:
        ir = IRDocument(blocks=[_para(20, page=1), _para(20, page=2)])
        drafts = _chunk(ir)
        assert all(draft.page is None and draft.slide is None for draft in drafts)


class TestBudgets:
    def test_long_section_splits_with_paragraph_overlap(self) -> None:
        p1, p2, p3 = _para(30), _para(10), _para(30)
        ir = IRDocument(blocks=[p1, p2, p3])
        drafts = _chunk(ir)
        assert len(drafts) == 3
        assert all(draft.token_count <= PARAMS.max_tokens for draft in drafts)
        assert drafts[1].text == p2.text
        assert drafts[2].text == f"{p2.text}\n\n{p3.text}"

    def test_oversized_block_splits_into_sentences_within_max(self) -> None:
        sentences = [f"This is sentence {i} with ten simple words total now." for i in range(9)]
        ir = IRDocument(blocks=[IRBlock(kind="paragraph", text=" ".join(sentences))])
        drafts = _chunk(ir)
        assert len(drafts) == 3
        assert all(draft.token_count <= PARAMS.max_tokens for draft in drafts)
        assert drafts[0].text == " ".join(sentences[:3])
        assert drafts[1].text == " ".join(sentences[3:6])
        assert drafts[2].text == " ".join(sentences[6:])

    def test_sentence_over_max_is_hard_split_by_words(self) -> None:
        ir = IRDocument(
            blocks=[IRBlock(kind="paragraph", text=" ".join(f"w{i}" for i in range(100)))]
        )
        drafts = _chunk(ir)
        assert len(drafts) >= 2
        assert all(draft.token_count <= PARAMS.max_tokens for draft in drafts)
        joined = " ".join(draft.text for draft in drafts)
        assert "w0" in joined
        assert "w99" in joined

    def test_table_split_by_rows_repeats_header(self) -> None:
        header = "| Item | Count |"
        separator = "| --- | --- |"
        rows = [f"| row {i} | {i} |" for i in range(50)]
        table = "\n".join([header, separator, *rows])
        ir = IRDocument(blocks=[IRBlock(kind="table", text=table)])
        drafts = _chunk(ir)
        assert len(drafts) >= 2
        assert all(draft.is_table for draft in drafts)
        assert all(draft.token_count <= PARAMS.max_tokens for draft in drafts)
        assert all(draft.text.startswith(f"{header}\n{separator}") for draft in drafts)

    def test_code_block_kept_whole_when_it_fits(self) -> None:
        code = "\n".join(f"line {i} of code" for i in range(10))
        ir = IRDocument(blocks=[IRBlock(kind="code", text=code)])
        (draft,) = _chunk(ir)
        assert draft.text == code
        assert not draft.is_table

    def test_code_block_over_max_splits_by_lines(self) -> None:
        code = "\n".join(f"line {i} of code here" for i in range(60))
        ir = IRDocument(blocks=[IRBlock(kind="code", text=code)])
        drafts = _chunk(ir)
        assert len(drafts) >= 2
        assert all(draft.token_count <= PARAMS.max_tokens for draft in drafts)
        lines = [line for draft in drafts for line in draft.text.splitlines()]
        assert lines == code.splitlines()


class TestHeadingOnlyChunks:
    def test_heading_followed_by_heading_emits_heading_only(self) -> None:
        ir = IRDocument(
            blocks=[
                IRBlock(kind="heading", text="A", level=1),
                IRBlock(kind="heading", text="B", level=2),
                _para(20),
            ]
        )
        drafts = _chunk(ir)
        assert len(drafts) == 2
        assert drafts[0].text == "# A"
        assert drafts[0].heading_path == ["A"]
        assert drafts[1].text.startswith("## B")
        assert drafts[1].heading_path == ["A", "B"]

    def test_heading_at_document_end_emits_heading_only(self) -> None:
        ir = IRDocument(blocks=[IRBlock(kind="heading", text="A", level=1)])
        drafts = _chunk(ir)
        assert [draft.text for draft in drafts] == ["# A"]
        assert drafts[0].heading_path == ["A"]

    def test_tiny_non_heading_chunk_is_dropped(self) -> None:
        ir = IRDocument(blocks=[IRBlock(kind="paragraph", text="ok")])
        assert _chunk(ir) == []

    def test_tiny_paragraph_merged_into_big_chunk_is_kept(self) -> None:
        ir = IRDocument(blocks=[IRBlock(kind="paragraph", text="ok"), _para(10)])
        drafts = _chunk(ir)
        assert [draft.text for draft in drafts] == [f"ok\n\n{_para(10).text}"]


class TestEmbedText:
    def test_embed_text_format_with_all_parts(self) -> None:
        assert (
            build_embed_text(
                title="My Doc",
                rel_path="projects/alpha/notes.md",
                heading_path=["A", "B"],
                tags=["t1", "t2"],
                text="body",
            )
            == "Title: My Doc | Path: projects/alpha | Section: A > B | Tags: t1, t2\n\nbody"
        )

    def test_embed_text_skips_empty_parts(self) -> None:
        assert (
            build_embed_text(title=None, rel_path="notes.md", heading_path=[], tags=[], text="body")
            == "body"
        )
        assert (
            build_embed_text(
                title=None, rel_path="notes.md", heading_path=["A"], tags=[], text="body"
            )
            == "Section: A\n\nbody"
        )

    def test_chunk_embed_text_carries_title_folder_section_tags(self) -> None:
        ir = IRDocument(
            blocks=[IRBlock(kind="heading", text="A", level=1), _para(20)],
            title="My Doc",
        )
        (draft,) = _chunk(ir, rel_path="projects/alpha/notes.md", tags=["t1", "t2"])
        header = "Title: My Doc | Path: projects/alpha | Section: A | Tags: t1, t2"
        assert draft.embed_text == f"{header}\n\n{draft.text}"

    def test_chunk_at_root_has_no_path_part(self) -> None:
        ir = IRDocument(blocks=[_para(20)])
        (draft,) = _chunk(ir, rel_path="notes.md")
        assert draft.embed_text == draft.text


class TestDeterminism:
    def test_chunking_twice_gives_identical_hashes(self) -> None:
        ir = IRDocument(
            blocks=[
                IRBlock(kind="heading", text="A", level=1),
                _para(30),
                _para(10),
                IRBlock(kind="heading", text="B", level=2),
                IRBlock(kind="heading", text="C", level=3),
                _para(30, page=2),
            ],
            title="Doc",
        )
        first = _chunk(ir, page_kind="page")
        second = _chunk(ir, page_kind="page")
        assert [draft.content_hash for draft in first] == [draft.content_hash for draft in second]
        assert [draft.text for draft in first] == [draft.text for draft in second]


class TestLanguageFallback:
    def test_document_language_used_when_detection_returns_none(self) -> None:
        ir = IRDocument(blocks=[IRBlock(kind="heading", text="A", level=1)], language="en")
        drafts = _chunk(ir)
        assert all(draft.language == "en" for draft in drafts)
