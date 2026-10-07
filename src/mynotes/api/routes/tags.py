"""Tag endpoints: vocabulary, merge and proposal review (PLAN.md §10.2).

Accept, reject and merge responses carry an ``HX-Trigger: mynotes:tags-changed``
header (task 48) so the web UI re-renders its tag fragments; the JSON bodies
are unchanged.
"""

from __future__ import annotations

from fastapi import APIRouter
from starlette.responses import JSONResponse

from mynotes.api import schemas
from mynotes.api.auth import PrincipalDep
from mynotes.core import service

router = APIRouter()

_HX_TAGS_CHANGED = {"HX-Trigger": "mynotes:tags-changed"}


@router.get("/v1/tags")
def list_tags(principal: PrincipalDep, prefix: str | None = None) -> list[schemas.TagModel]:
    """The tag vocabulary with accepted-usage counts."""
    tags = service.list_tags(prefix, principal=principal)
    return [schemas.TagModel.model_validate(tag, from_attributes=True) for tag in tags]


@router.post("/v1/tags/merge")
def merge_tags(payload: schemas.MergeTagsRequest, principal: PrincipalDep) -> dict[str, object]:
    """Merge one tag into another and report the moved row count."""
    moved = service.merge_tags(payload.from_name, payload.into, principal=principal)
    return JSONResponse(content={"ok": True, "moved": moved}, headers=_HX_TAGS_CHANGED)


@router.post("/v1/tags/{document_id}/{tag}/accept")
def accept_tag(document_id: int, tag: str, principal: PrincipalDep) -> dict[str, bool]:
    """Accept one proposed LLM tag of a document."""
    service.tag_action(document_id, tag, "accept", principal=principal)
    return JSONResponse(content={"ok": True}, headers=_HX_TAGS_CHANGED)


@router.post("/v1/tags/{document_id}/{tag}/reject")
def reject_tag(document_id: int, tag: str, principal: PrincipalDep) -> dict[str, bool]:
    """Reject one proposed LLM tag (it will not be proposed again)."""
    service.tag_action(document_id, tag, "reject", principal=principal)
    return JSONResponse(content={"ok": True}, headers=_HX_TAGS_CHANGED)
