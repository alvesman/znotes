"""KG ontology: entity types, relation types and the extractor version (PLAN.md §8.2).

The lists are the vocabulary the extraction prompt offers and the normalization
target for LLM answers: an unknown entity type becomes ``"Concept"``, an
unknown relation type becomes ``"related_to"`` (the raw label is kept on the
relation). Config overrides may come later; the constants are enough now.

``EXTRACTOR_VERSION`` is part of every extraction cache key
(:class:`mynotes.db.models.KgExtractionCache`): bump it whenever the prompt or
the ontology changes so a deliberate full rebuild happens on the next
``mynotes graph backfill``.
"""

from __future__ import annotations

ENTITY_TYPES = [
    "Project",
    "Idea",
    "Person",
    "Organization",
    "Technology",
    "Concept",
    "Decision",
    "Requirement",
    "Milestone",
    "Risk",
    "Metric",
]
"""The ontology's entity types, in prompt order."""

RELATION_TYPES = [
    "part_of",
    "depends_on",
    "uses",
    "proposes",
    "decided",
    "owned_by",
    "works_on",
    "targets",
    "has_requirement",
    "has_risk",
    "measured_by",
    "scheduled_for",
    "replaces",
    "contradicts",
    "related_to",
]
"""The ontology's normalized relation types (PLAN.md §8.2)."""

EXTRACTOR_VERSION = "kg-v1"
"""Bumped when the prompt or ontology changes, forcing a full rebuild."""
