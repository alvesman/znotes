"""Unit tests for bare-date folding (improvement plan step 6)."""

from __future__ import annotations

import pytest

from mynotes.graph.dates import DATE_ATTRIBUTE, fold_dates, is_bare_date
from mynotes.graph.extract import Extraction


@pytest.mark.parametrize(
    "name",
    [
        "2023-05-20",
        "2022-07",
        "2018",
        "20/05/2023",
        "20.05.23",
        "2023/05/20",
        "August 2022",
        "17 de junho de 2025",
        "Junho de 2025",
        "março",
        "Q3 2025",
        "Dec 2024",
    ],
)
def test_bare_dates(name: str) -> None:
    assert is_bare_date(name)


@pytest.mark.parametrize(
    "name",
    [
        "Release 2.0",
        "Daily Note 2022-01-18",
        "Project Start Date",
        "1999 Honda",
        "Season 1",
        "SonarQube end date",
        "Sábado às 23h",
        "v2023.1",
    ],
)
def test_names_that_merely_contain_a_date_stay(name: str) -> None:
    assert not is_bare_date(name)


def _extraction() -> Extraction:
    return Extraction.model_validate(
        {
            "entities": [
                {"name": "Update OS", "type": "Project", "description": "", "aliases": []},
                {"name": "2024-10-15", "type": "Milestone", "description": "", "aliases": []},
                {"name": "Ep 2", "type": "Milestone", "description": "", "aliases": []},
                {"name": "2022-10-17", "type": "Milestone", "description": "", "aliases": []},
            ],
            "relations": [
                {"source": "Update OS", "target": "2024-10-15", "type": "scheduled_for"},
                {
                    "source": "Ep 2",
                    "target": "2022-10-17",
                    "type": "scheduled_for",
                    "attribute": "release_date",
                    "value": "2022-10-17",
                },
                {"source": "2024-10-15", "target": "Update OS", "type": "related_to"},
                {"source": "Update OS", "target": "Ep 2", "type": "related_to"},
            ],
        }
    )


def test_fold_dates_rewrites_relations_and_drops_date_entities() -> None:
    folded = fold_dates(_extraction())
    assert [e.name for e in folded.entities] == ["Update OS", "Ep 2"]
    rels = [(r.source, r.target, r.type, r.attribute, r.value) for r in folded.relations]
    assert rels == [
        ("Update OS", "Update OS", "scheduled_for", DATE_ATTRIBUTE, "2024-10-15"),
        ("Ep 2", "Ep 2", "scheduled_for", "release_date", "2022-10-17"),
        ("Update OS", "Ep 2", "related_to", None, None),
    ]


def test_fold_dates_is_a_no_op_without_dates() -> None:
    extraction = Extraction.model_validate(
        {
            "entities": [{"name": "Ana", "type": "Person", "description": "", "aliases": []}],
            "relations": [],
        }
    )
    assert fold_dates(extraction) is extraction


def test_fold_dates_dedupes_identical_folded_relations() -> None:
    extraction = Extraction.model_validate(
        {
            "entities": [
                {"name": "X", "type": "Project", "description": "", "aliases": []},
                {"name": "2024-01-01", "type": "Milestone", "description": "", "aliases": []},
            ],
            "relations": [
                {"source": "X", "target": "2024-01-01", "type": "scheduled_for"},
                {
                    "source": "X",
                    "target": "2024-01-01",
                    "type": "scheduled_for",
                    "attribute": "date",
                    "value": "2024-01-01",
                },
            ],
        }
    )
    assert len(fold_dates(extraction).relations) == 1
