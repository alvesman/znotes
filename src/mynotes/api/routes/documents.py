"""Document endpoints: lookup, view and version history (PLAN.md §10.2)."""

from __future__ import annotations

from fastapi import APIRouter, Query

from mynotes.api import schemas
from mynotes.api.auth import PrincipalDep
from mynotes.core import service

router = APIRouter()


@router.get("/v1/documents/lookup")
def lookup_document(
    principal: PrincipalDep,
    ref: str = Query(..., description="Document id, path or title."),
) -> schemas.DocumentModel:
    """Resolve a document reference; 409 with candidates when ambiguous."""
    return schemas.DocumentModel.model_validate(
        service.get_document(ref, principal=principal), from_attributes=True
    )


@router.get("/v1/documents/{document_id}")
def get_document(document_id: int, principal: PrincipalDep) -> schemas.DocumentModel:
    """One document with its extracted content, tags and section map."""
    return schemas.DocumentModel.model_validate(
        service.get_document(document_id, principal=principal), from_attributes=True
    )


@router.get("/v1/documents/{document_id}/history")
def document_history(document_id: int, principal: PrincipalDep) -> list[schemas.VersionInfoModel]:
    """A document's retained version snapshots, newest first."""
    infos = service.document_history(document_id, principal=principal)
    return [schemas.VersionInfoModel.model_validate(info, from_attributes=True) for info in infos]
