"""The search endpoint (PLAN.md §10.2)."""

from __future__ import annotations

from fastapi import APIRouter

from mynotes.api import schemas
from mynotes.api.auth import PrincipalDep
from mynotes.core import service
from mynotes.retrieval.types import SearchFilters

router = APIRouter()


def _filters_model(filters: SearchFilters) -> schemas.FiltersModel:
    """The applied filters as a response model (``history`` folded back in)."""
    return schemas.FiltersModel(
        tags=list(filters.tags),
        folders=list(filters.folders),
        formats=list(filters.formats),
        since=filters.since,
        until=filters.until,
        history=filters.include_superseded or filters.include_old_versions,
    )


@router.post("/v1/search")
def search(payload: schemas.SearchRequest, principal: PrincipalDep) -> schemas.SearchResponse:
    """Run the retrieval pipeline and return the ranked hits."""
    base_filters = payload.filters.to_search_filters() if payload.filters is not None else None
    result = service.search(payload.query, base_filters, principal=principal, rerank=payload.rerank)
    return schemas.SearchResponse(
        hits=[
            schemas.HitModel.model_validate(hit, from_attributes=True)
            for hit in result.hits[: payload.limit]
        ],
        analysis=schemas.AnalysisModel(
            intent=result.analysis.intent,
            text=result.analysis.text,
            filters=_filters_model(result.analysis.filters),
        ),
    )
