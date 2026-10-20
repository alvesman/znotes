"""Unit tests for graph service helpers: entity references and edge filtering."""

from __future__ import annotations

from mynotes.core.service import entity_id_ref


def test_entity_id_ref_parses_hash_id_only() -> None:
    assert entity_id_ref("#519") == 519
    assert entity_id_ref("  #7 ") == 7
    assert entity_id_ref("519") is None  # a plain number is still a name first
    assert entity_id_ref("#") is None
    assert entity_id_ref("#abc") is None
    assert entity_id_ref("Keycloak") is None


def test_edges_among_keeps_only_edges_between_shown_nodes() -> None:
    from mynotes.core.service import edges_among

    edges = [(1, 2, 10, 1.0), (2, 3, 11, 0.5), (1, 9, 12, 0.9), (9, 8, 13, 0.1)]
    assert edges_among(edges, {1, 2, 3}) == [(1, 2, 10, 1.0), (2, 3, 11, 0.5)]
    assert edges_among(edges, {1}) == []
