"""Raw file serving and open-in-app endpoints (PLAN.md §10.2, §10.4).

Files are addressed by ``document_id`` only; the path comes from the database
via :func:`mynotes.core.service.resolve_file` and is verified to stay inside
its source root, so no request input ever reaches the filesystem.
"""

from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter
from fastapi.responses import FileResponse, Response

from mynotes.api.auth import PrincipalDep
from mynotes.core import service

router = APIRouter()

_CSP_SANDBOX = "sandbox; script-src 'none'"
"""Sandbox CSP for inline HTML: no scripts, unique origin, no same-origin access."""


@router.get("/v1/files/{document_id}/raw")
def raw_file(document_id: int, principal: PrincipalDep) -> FileResponse:
    """Serve the document's original file (inline for PDF/text/HTML, else download)."""
    resolved = service.resolve_file(document_id, principal=principal)
    disposition = "inline" if resolved.inline else "attachment"
    headers = {
        "Content-Disposition": f"{disposition}; filename*=UTF-8''{quote(resolved.filename)}",
    }
    if resolved.media_type.startswith("text/html"):
        headers["Content-Security-Policy"] = _CSP_SANDBOX
    return FileResponse(resolved.path, media_type=resolved.media_type, headers=headers)


@router.post("/v1/files/{document_id}/open", status_code=204)
def open_file(document_id: int, principal: PrincipalDep) -> Response:
    """Open the original file in the OS default app (local mode only)."""
    service.open_original(document_id, principal=principal)
    return Response(status_code=204)
