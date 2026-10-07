"""Knowledge graph endpoints: entity cards, neighbours, timeline (PLAN.md §10.2)."""

from __future__ import annotations

from fastapi import APIRouter, Query

from mynotes.api import schemas
from mynotes.api.auth import PrincipalDep
from mynotes.core import service

router = APIRouter()


@router.get("/v1/graph/entities")
def graph_entity_card(
    principal: PrincipalDep,
    q: str = Query(..., description="Entity name (exact, normalized or fuzzy)."),
) -> dict:
    """One entity card; 409 with candidates when the name is ambiguous."""
    return service.entity_card(q, principal=principal)


@router.get("/v1/graph/entities/{ref}/neighbors")
def graph_entity_neighbors(
    principal: PrincipalDep,
    ref: str,
    hops: int = Query(default=1, ge=1),
) -> dict:
    """The scored neighbours of one entity, up to ``hops`` relation steps."""
    return service.entity_neighbors(ref, hops, principal=principal)


@router.get("/v1/graph/timeline")
def graph_timeline(
    principal: PrincipalDep,
    entity: str = Query(..., description="Entity name, or 'project:NAME'."),
) -> list[schemas.TimelineItemModel]:
    """The temporal timeline of one entity or project tag."""
    items = service.timeline(entity, principal=principal)
    return [schemas.TimelineItemModel.model_validate(item, from_attributes=True) for item in items]
