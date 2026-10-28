"""Admin endpoints: scan trigger, status and health (PLAN.md §10.2).

The scan response carries an ``HX-Trigger: mynotes:status-changed`` header
(task 48) so the web status dashboard refreshes immediately; its JSON body is
unchanged, and the request body is optional so the UI button can post empty.
"""

from __future__ import annotations

import logging
import threading

from fastapi import APIRouter
from starlette.responses import JSONResponse

from mynotes.api.auth import PrincipalDep
from mynotes.api.schemas import ScanRequest
from mynotes.core import service
from mynotes.core.errors import Forbidden
from mynotes.core.principal import Principal

logger = logging.getLogger(__name__)

router = APIRouter()


def _run_scan(root: str | None, full: bool, principal: Principal) -> None:
    """Run the scan off-thread and log the outcome (the caller already got 202)."""
    try:
        summaries = service.trigger_scan(root, full, principal=principal)
        logger.info("api scan finished: %s", [summary.counts for summary in summaries])
    except Exception:
        logger.exception("api scan of root %r failed", root)


@router.post("/v1/scan", status_code=202)
def trigger_scan(
    payload: ScanRequest | None = None, principal: PrincipalDep = None
) -> dict[str, object]:
    """Start a scan of the enabled roots (or one named root) in the background (admin only)."""
    if not principal.is_admin:
        raise Forbidden("this action requires an admin principal")
    thread = threading.Thread(
        target=_run_scan,
        args=(payload.root if payload else None, payload.full if payload else False, principal),
        name="api-scan",
        daemon=True,
    )
    thread.start()
    return JSONResponse(
        status_code=202,
        content={"accepted": True, "detail": "scan started in the background"},
        headers={"HX-Trigger": "mynotes:status-changed"},
    )


@router.get("/v1/status")
def status(principal: PrincipalDep) -> dict:
    """The current state of config, database, indices and endpoints."""
    return service.status(principal=principal)


@router.get("/healthz")
def healthz() -> dict[str, object]:
    """Liveness plus Postgres/OpenSearch reachability flags."""
    return {"status": "ok", "postgres": _ping_postgres(), "opensearch": _ping_opensearch()}


def _ping_postgres() -> bool:
    """Whether Postgres answers a trivial query."""
    import sqlalchemy as sa

    from mynotes.db.session import get_engine

    try:
        with get_engine().connect() as connection:
            connection.execute(sa.text("SELECT 1"))
    except Exception:
        logger.info("healthz: postgres ping failed", exc_info=True)
        return False
    return True


def _ping_opensearch() -> bool:
    """Whether the OpenSearch cluster answers a ping."""
    from mynotes.index.opensearch import get_client

    try:
        return bool(get_client().ping())
    except Exception:
        logger.info("healthz: opensearch ping failed", exc_info=True)
        return False
