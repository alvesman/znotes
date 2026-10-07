"""Version relation endpoints: list and review actions (PLAN.md §10.2).

Confirm/reject/swap responses carry an ``HX-Trigger: mynotes:versions-changed``
header (task 48) so the web UI re-renders its review tab; the JSON bodies are
unchanged.
"""

from __future__ import annotations

from fastapi import APIRouter
from starlette.responses import JSONResponse

from mynotes.api import schemas
from mynotes.api.auth import PrincipalDep
from mynotes.core import service

router = APIRouter()

_HX_VERSIONS_CHANGED = {"HX-Trigger": "mynotes:versions-changed"}


@router.get("/v1/versions")
def list_versions(
    principal: PrincipalDep, status: str | None = None
) -> list[schemas.RelationModel]:
    """The supersedes relations, optionally filtered by status."""
    relations = service.list_version_relations(status, principal=principal)
    return [
        schemas.RelationModel.model_validate(relation, from_attributes=True)
        for relation in relations
    ]


@router.post("/v1/versions/{relation_id}/confirm")
def confirm_version(relation_id: int, principal: PrincipalDep) -> dict[str, bool]:
    """Confirm a proposed supersedes relation (applies the chain effects)."""
    service.version_action(relation_id, "confirm", principal=principal)
    return JSONResponse(content={"ok": True}, headers=_HX_VERSIONS_CHANGED)


@router.post("/v1/versions/{relation_id}/reject")
def reject_version(relation_id: int, principal: PrincipalDep) -> dict[str, bool]:
    """Reject a proposed supersedes relation (undoes its effects)."""
    service.version_action(relation_id, "reject", principal=principal)
    return JSONResponse(content={"ok": True}, headers=_HX_VERSIONS_CHANGED)


@router.post("/v1/versions/{relation_id}/swap")
def swap_version(relation_id: int, principal: PrincipalDep) -> dict[str, bool]:
    """Reverse a supersedes relation's direction and confirm it."""
    service.version_action(relation_id, "swap", principal=principal)
    return JSONResponse(content={"ok": True}, headers=_HX_VERSIONS_CHANGED)
