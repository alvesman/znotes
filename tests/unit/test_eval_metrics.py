"""Tests for eval harness metrics (pure logic, no infra)."""

from __future__ import annotations

from eval import metrics as m
from mynotes.retrieval.prompts import NOT_FOUND

QUESTION = {
    "id": "q001",
    "question": "Qual a temperatura?",
    "language": "pt",
    "category": "factual",
    "expected_docs": ["docs/manual-x.md"],
    "expected_pages": [12, 13],
    "tags": ["table"],
}

HIT = {"path": "docs/manual-x.md", "page": 12}
WRONG_PAGE = {"path": "docs/manual-x.md", "page": 3}
WRONG_DOC = {"path": "docs/other.md", "page": 12}


def test_doc_hit_at_k():
    assert m.doc_hit_at_k(QUESTION, [WRONG_DOC, HIT], 2)
    assert not m.doc_hit_at_k(QUESTION, [WRONG_DOC, HIT], 1)


def test_doc_hit_any_expected_doc_counts():
    question = dict(QUESTION, expected_docs=["docs/other.md", "docs/manual-x.md"])
    assert m.doc_hit_at_k(question, [WRONG_DOC], 1)


def test_page_hit_requires_doc_and_page():
    assert m.page_hit_at_k(QUESTION, [HIT], 1)
    assert not m.page_hit_at_k(QUESTION, [WRONG_PAGE], 1)
    assert not m.page_hit_at_k(QUESTION, [WRONG_DOC], 1)


def test_page_hit_none_without_expected_pages():
    question = dict(QUESTION, expected_pages=[])
    assert m.page_hit_at_k(question, [HIT], 1) is None


def test_reciprocal_rank_first_page_match():
    assert m.reciprocal_rank(QUESTION, [WRONG_DOC, WRONG_PAGE, HIT]) == 1 / 3
    assert m.reciprocal_rank(QUESTION, []) == 0.0


def test_reciprocal_rank_doc_only_when_no_expected_pages():
    question = dict(QUESTION, expected_pages=[])
    assert m.reciprocal_rank(question, [WRONG_DOC, WRONG_PAGE]) == 1 / 2


def test_no_expected_docs_metrics_are_none():
    question = dict(QUESTION, id="q003", category="no-answer", expected_docs=[])
    metrics = m.question_retrieval_metrics(question, [HIT])
    assert metrics["mrr"] is None
    assert metrics["doc_hit@5"] is None
    assert metrics["page_hit@5"] is None
    assert set(metrics) == set(m._RETRIEVAL_KEYS)


def test_source_attribution_by_path():
    sources = [{"path": "docs/manual-x.md", "page": 12}]
    assert m.source_attribution_correct(QUESTION, sources)
    assert not m.source_attribution_correct(QUESTION, [{"path": "docs/other.md", "page": 12}])


def test_current_version_precision_two_cited_one_stale():
    question = dict(QUESTION, stale_docs=["docs/manual-x-v1.md"])
    citations = [
        {"path": "docs/manual-x.md", "is_current": True},
        {"path": "docs/manual-x-v1.md", "is_current": False},
    ]
    assert m.current_version_precision(question, citations) == 0.5
    assert m.current_version_precision(question, []) is None


def test_current_version_precision_ignores_stale_path_of_current_doc():
    question = dict(QUESTION, stale_docs=["docs/manual-x-v1.md"])
    citations = [{"path": "docs/manual-x-v1.md", "is_current": True}]
    assert m.current_version_precision(question, citations) == 0.0


def test_stale_answer_matches_normalized_value():
    question = dict(QUESTION, stale_values=["15k€", "Ação Técnica"])
    assert m.stale_answer(question, "O orçamento era 15K€ para a ACAO TECNICA.") is True
    assert m.stale_answer(question, "O orçamento é 20k€ aprovado.") is False
    assert m.stale_answer(QUESTION, "15k€") is None


def test_no_answer_correct():
    question = dict(QUESTION, category="no-answer")
    not_found = {"answer": NOT_FOUND["pt"], "relevant": True}
    assert m.no_answer_correct(question, not_found) is True
    assert m.no_answer_correct(question, {"answer": "Orçamento: 20k€", "relevant": True}) is False
    assert m.no_answer_correct(question, {"answer": "qualquer", "relevant": False}) is True
    assert m.no_answer_correct(QUESTION, not_found) is None


def test_strip_accents_keeps_case():
    assert m.strip_accents("Ação Técnica") == "Acao Tecnica"
    assert m.strip_accents("CORAÇÃO") == "CORACAO"
    assert m.strip_accents("plain") == "plain"


def test_aggregate_slices_by_category_language_and_tag():
    rows = [
        {"question": QUESTION, "metrics": {"mrr": 1.0, "doc_hit@1": True}},
        {
            "question": dict(QUESTION, id="q002", language="en", category="no-answer", tags=[]),
            "metrics": {"mrr": None, "doc_hit@1": None},
        },
    ]
    aggregate = m.aggregate(rows)
    assert aggregate["overall"]["mrr"] == 1.0
    assert aggregate["overall"]["n"] == 2
    assert aggregate["slices"]["language:pt"]["mrr"] == 1.0
    assert aggregate["slices"]["language:en"] == {"n": 1}
    assert aggregate["slices"]["category:factual"]["n"] == 1
    assert aggregate["slices"]["category:no-answer"]["n"] == 1
    assert aggregate["slices"]["tag:table"]["n"] == 1


def test_parity_block_means_per_question_delta():
    original = [{"question": QUESTION, "metrics": {"mrr": 1.0, "doc_hit@5": True}}]
    variant = [{"question": QUESTION, "metrics": {"mrr": 0.5, "doc_hit@5": True}}]
    assert m.parity_block(original, variant) == {"doc_hit@5": 0.0, "mrr": -0.5}


def test_parity_block_none_without_measurable_pairs():
    original = [{"question": QUESTION, "metrics": {"mrr": None, "doc_hit@5": None}}]
    variant = [{"question": QUESTION, "metrics": {"mrr": None, "doc_hit@5": None}}]
    assert m.parity_block(original, variant) == {"doc_hit@5": None, "mrr": None}


def test_is_decline_recognizes_model_phrasing() -> None:
    declines = [
        "The provided notes do not contain information about the number of pull requests merged.",
        "A data de expiração do certificado TLS não consta nas notas fornecidas.",
        "The phone number of the building reception is not mentioned in the provided notes.",
        "Um livro do autor 'J.K. Rowling' não se encontra nas notas fornecidas.",
        "As notas fornecidas não contêm informação sobre uma data limite definida.",
        "NÃO ENCONTREI ESSA INFORMAÇÃO NAS TUAS NOTAS.",
        NOT_FOUND["en"],
    ]
    for text in declines:
        assert m.is_decline(text), text
    for text in ["O orçamento aprovado é 20k€ [1].", "The deadline is 3 March 2026 [2].", ""]:
        assert not m.is_decline(text), text


def test_no_answer_correct_counts_model_declines() -> None:
    question = {"id": "q070", "category": "no-answer"}
    declined = {"answer": "Essa data não consta nas notas fornecidas.", "relevant": True}
    assert m.no_answer_correct(question, declined) is True
