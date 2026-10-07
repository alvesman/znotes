"""Unit tests for the pure temporal rules (task 42)."""

from __future__ import annotations

from datetime import UTC, datetime

from mynotes.graph.temporal import (
    AttributeFact,
    EvidenceFact,
    pick_current_attribute,
    relation_status,
)

_D1 = datetime(2026, 3, 10, tzinfo=UTC)
_D2 = datetime(2026, 9, 1, tzinfo=UTC)
_D3 = datetime(2026, 10, 5, tzinfo=UTC)


def _fact(
    doc_date: datetime | None = None, is_current: bool = True, deleted: bool = False
) -> EvidenceFact:
    return EvidenceFact(doc_date=doc_date, is_current=is_current, deleted=deleted)


def test_current_evidence_is_active_with_earliest_valid_from() -> None:
    status, valid_from, valid_to = relation_status(
        [_fact(_D2), _fact(_D1, is_current=False)], [_D3]
    )
    assert (status, valid_from, valid_to) == ("active", _D1, None)


def test_current_evidence_without_dates_is_active() -> None:
    status, valid_from, valid_to = relation_status([_fact(None)], [])
    assert (status, valid_from, valid_to) == ("active", None, None)


def test_current_evidence_on_deleted_document_does_not_count() -> None:
    status, valid_from, valid_to = relation_status([_fact(_D1, deleted=True)], [])
    assert (status, valid_from, valid_to) == ("stale", _D1, None)


def test_historical_uses_earliest_successor_date() -> None:
    status, valid_from, valid_to = relation_status(
        [_fact(_D1, is_current=False), _fact(_D2, is_current=False)], [_D3, _D2]
    )
    assert (status, valid_from, valid_to) == ("historical", _D1, _D2)


def test_historical_falls_back_to_latest_evidence_date() -> None:
    status, valid_from, valid_to = relation_status(
        [_fact(_D1, is_current=False), _fact(_D3, is_current=False)], []
    )
    assert (status, valid_from, valid_to) == ("historical", _D1, _D3)


def test_historical_without_any_dates_has_no_valid_to() -> None:
    status, valid_from, valid_to = relation_status([_fact(None, is_current=False)], [])
    assert (status, valid_from, valid_to) == ("historical", None, None)


def test_mixed_deleted_and_living_evidence_is_historical() -> None:
    status, valid_from, valid_to = relation_status(
        [_fact(_D1, deleted=True), _fact(_D2, is_current=False)], []
    )
    assert (status, valid_from, valid_to) == ("historical", _D1, _D2)


def test_all_evidence_deleted_is_stale() -> None:
    status, valid_from, valid_to = relation_status(
        [_fact(_D1, deleted=True), _fact(_D2, is_current=False, deleted=True)], []
    )
    assert (status, valid_from, valid_to) == ("stale", _D1, None)


def test_no_evidence_is_stale() -> None:
    assert relation_status([], []) == ("stale", None, None)


def test_pick_current_attribute_newest_date_wins() -> None:
    older = AttributeFact(relation_id=1, value="15k", doc_date=_D1, confidence=0.99)
    newer = AttributeFact(relation_id=2, value="20k", doc_date=_D2, confidence=0.5)
    assert pick_current_attribute([older, newer]) is newer


def test_pick_current_attribute_ties_go_to_higher_confidence() -> None:
    low = AttributeFact(relation_id=1, value="15k", doc_date=_D1, confidence=0.5)
    high = AttributeFact(relation_id=2, value="20k", doc_date=_D1, confidence=0.9)
    assert pick_current_attribute([low, high]) is high


def test_pick_current_attribute_undated_loses() -> None:
    stale = AttributeFact(relation_id=1, value="15k", doc_date=None, confidence=1.0)
    dated = AttributeFact(relation_id=2, value="20k", doc_date=_D1, confidence=0.1)
    assert pick_current_attribute([stale, dated]) is dated


def test_pick_current_attribute_without_current_evidence_is_none() -> None:
    stale = AttributeFact(relation_id=1, value="15k", doc_date=None, confidence=1.0)
    assert pick_current_attribute([stale]) is None


def test_pick_current_attribute_full_tie_is_deterministic() -> None:
    first = AttributeFact(relation_id=1, value="15k", doc_date=_D1, confidence=0.9)
    second = AttributeFact(relation_id=2, value="20k", doc_date=_D1, confidence=0.9)
    assert pick_current_attribute([second, first]) is first


def test_pick_current_attribute_empty_is_none() -> None:
    assert pick_current_attribute([]) is None
