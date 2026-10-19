"""Unit tests for the golden-set drafting helpers (task 32, Part A).

Only the pure helpers are covered here: the LLM call itself and the DB/OS
queries need the real stack and belong to the Part A draft run.
"""

from __future__ import annotations

import random

from eval.build_golden_set import (
    MIN_EXCERPT_WORDS,
    ChunkRow,
    DocInfo,
    DraftedQuestion,
    answer_match,
    assign_cross_lingual,
    find_alternates,
    question_row,
    select_excerpts,
    stratified_pick,
)

from mynotes.text.normalize import normalize_key


def make_doc(doc_id: int, folder: str, fmt: str, bucket: str) -> DocInfo:
    return DocInfo(
        doc_id=doc_id,
        version_id=doc_id,
        path=f"{folder}/file-{doc_id}.{fmt}",
        title=None,
        format=fmt,
        language=bucket if bucket in ("pt", "en") else None,
        folder=folder,
        chunk_langs={bucket} if bucket in ("pt", "en") else {"pt", "en"},
    )


def make_chunk(ordinal: int, words: int) -> ChunkRow:
    return ChunkRow(
        ordinal=ordinal,
        text=" ".join(f"word{ordinal}_{i}" for i in range(words)),
        page=None,
        slide=None,
        section=None,
        heading_path=[],
        is_table=False,
    )


def test_stratified_pick_round_robin_over_buckets() -> None:
    docs = [
        make_doc(1, "A", "md", "en"),
        make_doc(2, "A", "md", "en"),
        make_doc(3, "A", "md", "en"),
        make_doc(4, "B", "pdf", "pt"),
        make_doc(5, "B", "pdf", "pt"),
    ]
    picked = stratified_pick(docs, 3, random.Random(0), key=lambda d: (d.folder, d.lang_bucket))
    buckets = {(d.folder, d.lang_bucket) for d in picked}
    assert len(picked) == 3
    # Round-robin: both buckets are represented even though A has more docs.
    assert buckets == {("A", "en"), ("B", "pt")}


def test_stratified_pick_is_seed_deterministic_and_capped() -> None:
    docs = [make_doc(i, f"F{i % 3}", "md", "en") for i in range(12)]
    first = stratified_pick(docs, 4, random.Random(7), key=lambda d: d.folder)
    second = stratified_pick(docs, 4, random.Random(7), key=lambda d: d.folder)
    assert [d.doc_id for d in first] == [d.doc_id for d in second]
    assert len(first) == 4
    assert len(stratified_pick(docs, 100, random.Random(7), key=lambda d: d.folder)) == 12


def test_assign_cross_lingual_prefers_en_to_pt() -> None:
    picked = [make_doc(1, "A", "md", "en"), make_doc(2, "B", "md", "pt")]
    assignment = assign_cross_lingual(picked, n=2)
    assert assignment == {1: "pt", 2: "en"}

    only_en = [make_doc(i, "A", "md", "en") for i in range(3)]
    assert assign_cross_lingual(only_en, n=2) == {0: "pt", 1: "pt"}


def test_select_excerpts_spreads_and_tops_up() -> None:
    chunks = [make_chunk(i, MIN_EXCERPT_WORDS + 1) for i in range(20)]
    excerpts = select_excerpts(chunks, max_n=4)
    assert len(excerpts) == 4
    # Evenly spaced: first and last chunk are both sampled.
    assert "word0_0" in excerpts[0]
    assert "word19_" in excerpts[3]

    tiny = [make_chunk(0, MIN_EXCERPT_WORDS - 1)]
    assert select_excerpts(tiny) == []


def test_question_row_cross_lingual_and_pages() -> None:
    info = make_doc(1, "work", "pdf", "en")
    info.paged = True
    drafted = DraftedQuestion(
        question="Qual e o prazo?",
        language="pt",
        category="factual",
        expected_pages=[3, 1, 3],
        reference_answer="2 semanas",
    )
    row = question_row(info, drafted, cross_lang="pt", tag_pick=False)
    assert row["language"] == "pt"
    assert row["category"] == "language"
    assert row["tags"] == ["draft", "cross-lingual"]
    assert row["expected_docs"] == [info.path]
    assert row["expected_pages"] == [1, 3]
    assert row["filters"] == {"tags": [], "folders": []}

    info.paged = False
    row = question_row(info, drafted, cross_lang=None, tag_pick=False)
    assert row["category"] == "factual"
    assert "expected_pages" not in row


def test_question_row_tag_filter_uses_first_tag() -> None:
    info = make_doc(1, "work", "md", "en")
    info.tags = ["daily", "journal"]
    drafted = DraftedQuestion(
        question="When do I review the sprint?",
        language="en",
        category="factual",
        reference_answer="Every Monday",
    )
    row = question_row(info, drafted, cross_lang=None, tag_pick=True)
    assert row["category"] == "tag-filtered"
    assert row["filters"] == {"tags": ["daily"], "folders": []}


def _chunks(*texts: str) -> list[tuple[str, int | None, int | None]]:
    """Chunk tuples as ``load_chunk_index`` builds them (normalized text)."""
    return [(normalize_key(text), page, None) for page, text in enumerate(texts, start=1)]


def test_answer_match_list_answer_matches_table_cells() -> None:
    table = (
        "| Database | Port |\n|---|---|\n| Orders | 31001 |\n| OrdersJobs | 31001 |\n"
        "| BillingJobs | 31001 |\n| Reports | 31001 |\n| ReportsJobs | 31001 |"
    )
    answer = "Orders, OrdersJobs, BillingJobs, Reports, ReportsJobs"
    score, page = answer_match(answer, _chunks("intro text", table))
    assert score == 1.0
    assert page == 2


def test_answer_match_verbatim_ignores_punctuation() -> None:
    score, page = answer_match("`[!hint]`, `[!important]`.", _chunks("aliases: hint; important"))
    assert score == 1.0
    assert page == 1


def test_answer_match_partial_containment_uses_words() -> None:
    score, _ = answer_match("alpha, beta, gamma, delta", _chunks("alpha and beta only"))
    assert score == 0.5


def test_answer_match_empty_answer() -> None:
    assert answer_match("", _chunks("anything")) == (0.0, None)
    assert answer_match("!!", _chunks("anything")) == (0.0, None)


def test_answer_match_punctuation_only_answer_matches_verbatim() -> None:
    table = "| Left | Center |\n|:---|:------------:|\n| a | b |"
    assert answer_match(":------------:", _chunks("no table here", table)) == (1.0, 2)


def _plain_chunks(*texts: str) -> list[tuple[str, int | None, int | None]]:
    return [(normalize_key(text), None, None) for text in texts]


def test_find_alternates_lists_copies_of_the_answer() -> None:
    answer = "Ana + Rui + Marta Lopes"
    index = {
        "meetings/2025-06-02.md": _plain_chunks(f"- {answer} -> energia. Outros pontos."),
        "meetings/2025-07-01.md": _plain_chunks(f"- {answer} -> energia."),
        "meetings/2025-08-04.md": _plain_chunks(
            "Ana, Rui e Marta continuam."
        ),  # partial: 3 of 4 words
        "other.md": _plain_chunks("Nada a ver."),
    }
    questions = [
        {"id": "q1", "expected_docs": ["meetings/2025-07-01.md"], "reference_answer": answer},
        {"id": "q2", "expected_docs": ["other.md"], "reference_answer": "8080"},
        {"id": "q3", "expected_docs": [], "reference_answer": answer},
    ]

    found = find_alternates(questions, index)

    assert set(found) == {"q1", "q2"}  # q3 has no expected document
    assert found["q1"].verbatim == ["meetings/2025-06-02.md"]  # the expected doc is excluded
    assert found["q1"].partial == [("meetings/2025-08-04.md", 0.75)]
    assert found["q2"].skipped == "answer too short to audit"
