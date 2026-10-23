"""The ingest API for agents (server mode): manifests, uploads, deletes, moves, heartbeats.

Every route works on one remote root, ``/v1/roots/{root}/...``, and needs a
token whose ``ingest:<root>`` scope names it (403 otherwise). Paths in
requests and responses are relative to the agent's folder; the server stores
them prefixed with the root's name (``scan.remote``).
"""

from __future__ import annotations

import gzip
from typing import Annotated, Any

import blake3
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from mynotes.api import schemas
from mynotes.api.auth import PrincipalDep
from mynotes.config import get_settings
from mynotes.core.principal import Principal
from mynotes.db.models import SourceRoot, Workspace
from mynotes.db.session import session_scope
from mynotes.scan import remote
from mynotes.scan.formats import format_for
from mynotes.storage.blobs import HashMismatch, StorageNotConfigured, get_blob_store

router = APIRouter(prefix="/v1/roots")


def _load_root(name: str) -> SourceRoot | None:
    """The workspace's root of that name (detached, with its columns loaded)."""
    with session_scope() as session:
        row = session.execute(
            select(SourceRoot)
            .join(Workspace, Workspace.id == SourceRoot.workspace_id)
            .where(Workspace.name == get_settings().workspace, SourceRoot.name == name)
        ).scalar_one_or_none()
        if row is not None:
            session.expunge(row)
        return row


def remote_root(root: str, principal: PrincipalDep) -> SourceRoot:
    """The remote root the token may feed; 403/404/409 otherwise."""
    if root not in principal.ingest_roots:
        raise HTTPException(status_code=403, detail=f"this token may not feed root {root!r}")
    row = _load_root(root)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no root named {root!r}")
    if row.kind != "remote":
        raise HTTPException(
            status_code=409, detail=f"root {root!r} is a local folder, not fed by agents"
        )
    if not row.enabled:
        raise HTTPException(status_code=409, detail=f"root {root!r} is disabled")
    return row


RootDep = Annotated[SourceRoot, Depends(remote_root)]


def _store():
    try:
        return get_blob_store()
    except StorageNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _entry(model: schemas.ManifestEntryModel) -> remote.ManifestEntry:
    return remote.ManifestEntry(
        rel_path=model.rel_path,
        size=model.size,
        mtime_ns=model.mtime_ns,
        blake3=model.blake3.lower() if model.blake3 else None,
        placeholder=model.placeholder,
    )


@router.post("/{root}/manifest")
async def manifest(root: RootDep, request: Request) -> schemas.ManifestResponse:
    """Compare the agent's listing with the server's rows; answer what to upload."""
    body = await request.body()
    if request.headers.get("content-encoding", "").lower() == "gzip":
        try:
            body = gzip.decompress(body)
        except OSError as exc:
            raise HTTPException(status_code=400, detail="invalid gzip body") from exc
    payload = schemas.ManifestRequest.model_validate_json(body)
    settings = get_settings()
    if len(payload.entries) > settings.storage.max_manifest_entries:
        raise HTTPException(status_code=413, detail="manifest has too many entries")
    try:
        for path in payload.paths:
            remote.valid_agent_path(path)
        entries = [_entry(model) for model in payload.entries]
        for entry in entries:
            remote.valid_agent_path(entry.rel_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    store = _store()

    def work() -> remote.ManifestResult:
        with session_scope() as session:
            return remote.apply_manifest(
                session,
                root,
                entries,
                scope=payload.scope,
                paths=payload.paths or None,
                settings=settings,
                store=store,
            )

    try:
        result = await run_in_threadpool(work)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return schemas.ManifestResponse(
        scan_run_id=result.scan_run_id,
        counts=result.counts,
        upload=result.upload,
        actions=result.actions,
        skipped=result.skipped,
    )


@router.put("/{root}/files/{rel_path:path}")
async def upload(
    root: RootDep,
    rel_path: str,
    request: Request,
    x_content_blake3: Annotated[str, Header()],
    x_mtime_ns: Annotated[int, Header()],
    x_size: Annotated[int | None, Header()] = None,
) -> schemas.UploadResponse:
    """Store the file's bytes, then record the file exactly as a scan would."""
    try:
        rel_path = remote.valid_agent_path(rel_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    fmt = format_for(rel_path)
    if fmt is None or fmt == "unsupported":
        raise HTTPException(status_code=415, detail="not a format mynotes extracts")
    settings = get_settings()
    max_bytes = settings.storage.max_upload_mb * 1024 * 1024
    declared = request.headers.get("content-length")
    if declared is not None and int(declared) > max_bytes:
        raise HTTPException(status_code=413, detail="file larger than storage.max_upload_mb")
    store = _store()
    # The bytes go straight to a temp file in the store, hashed on the way, never into memory.
    tmp = store.new_temp()
    digest = blake3.blake3()
    size = 0
    try:
        with open(tmp, "wb") as handle:
            async for chunk in request.stream():
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413, detail="file larger than storage.max_upload_mb"
                    )
                digest.update(chunk)
                handle.write(chunk)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    content_hash = digest.hexdigest()

    def work() -> schemas.UploadResponse:
        try:
            stored = store.adopt(
                tmp,
                int(root.workspace_id),
                int(root.id),
                fmt,
                content_hash,
                expected_hash=x_content_blake3,
            )
        except HashMismatch as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        entry = remote.ManifestEntry(
            rel_path=rel_path,
            size=x_size if x_size is not None else stored.size,
            mtime_ns=x_mtime_ns,
            blake3=stored.content_hash,
        )
        with session_scope() as session:
            action = remote.record_upload(session, root, entry, settings=settings, store=store)
        return schemas.UploadResponse(
            action=action, content_hash=stored.content_hash, size=stored.size
        )

    return await run_in_threadpool(work)


@router.delete("/{root}/files/{rel_path:path}")
def delete_file(root: RootDep, rel_path: str, force: bool = False) -> dict[str, str]:
    """The file is gone on the agent: mark it missing (or delete it at once with ``force``)."""
    try:
        rel_path = remote.valid_agent_path(rel_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    settings = get_settings()
    store = _store()
    with session_scope() as session:
        action = remote.delete_path(
            session, root, rel_path, force=force, settings=settings, store=store
        )
    return {"action": action}


@router.post("/{root}/files/{rel_path:path}/move")
def move_file(root: RootDep, rel_path: str, payload: schemas.MoveRequest) -> dict[str, Any]:
    """A rename on the agent: pair the old and new paths (no re-upload when the hash is known)."""
    try:
        old = remote.valid_agent_path(rel_path)
        new = remote.valid_agent_path(payload.to)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    settings = get_settings()
    store = _store()
    entry = remote.ManifestEntry(
        rel_path=new, size=payload.size, mtime_ns=payload.mtime_ns, blake3=payload.blake3.lower()
    )
    with session_scope() as session:
        result = remote.move_path(session, root, old, entry, settings=settings, store=store)
    return {"action": result.actions.get(new, "unchanged"), "upload": result.upload}


@router.post("/{root}/heartbeat", status_code=204)
def heartbeat(root: RootDep, payload: schemas.HeartbeatRequest) -> None:
    """The agent is alive; shown on the status page."""
    with session_scope() as session:
        remote.heartbeat(session, root, payload.model_dump())


@router.get("/{root}/status")
def root_status(root: RootDep) -> dict[str, Any]:
    """Counts for this root: files by status, extraction state, the last manifest, the limits."""
    with session_scope() as session:
        data = remote.root_status(session, root)
    data["max_upload_mb"] = get_settings().storage.max_upload_mb
    return data


__all__ = ["Principal", "router"]
