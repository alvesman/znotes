"""HTTP client for the ingest API (the agent side of ``api/routes/ingest.py``)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from mynotes import __version__

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(30.0, read=300.0, write=300.0)


class AgentAuthError(RuntimeError):
    """The server refused the token (401) or the root (403): no retry will help."""


class ServerError(RuntimeError):
    """The server answered with an error the agent should retry later."""


class ServerClient:
    """Calls to ``/v1/roots/{root}/...`` with a bearer token."""

    def __init__(
        self,
        server_url: str,
        token: str,
        root: str,
        *,
        verify_tls: bool = True,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.root = root
        self._http = httpx.Client(
            base_url=server_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": f"mynotes-agent/{__version__}",
            },
            timeout=_TIMEOUT,
            verify=verify_tls,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def _check(self, response: httpx.Response) -> dict[str, Any]:
        if response.status_code in (401, 403):
            raise AgentAuthError(
                f"{response.status_code} from the server: {_detail(response)}; "
                "check the token and the root name"
            )
        if response.status_code >= 400:
            raise ServerError(f"{response.status_code} from the server: {_detail(response)}")
        if response.status_code == 204 or not response.content:
            return {}
        return response.json()

    def status(self) -> dict[str, Any]:
        return self._check(self._http.get(f"/v1/roots/{self.root}/status"))

    def manifest(
        self, entries: Sequence[dict[str, Any]], *, scope: str = "full", paths: Sequence[str] = ()
    ) -> dict[str, Any]:
        body = {"scope": scope, "paths": list(paths), "entries": list(entries)}
        return self._check(self._http.post(f"/v1/roots/{self.root}/manifest", json=body))

    def upload(
        self, rel_path: str, file: Path, *, content_hash: str, mtime_ns: int, size: int
    ) -> dict:
        with open(file, "rb") as handle:
            response = self._http.put(
                f"/v1/roots/{self.root}/files/{rel_path}",
                content=handle,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(size),
                    "X-Content-Blake3": content_hash,
                    "X-Mtime-Ns": str(mtime_ns),
                    "X-Size": str(size),
                },
            )
        return self._check(response)

    def delete(self, rel_path: str, *, force: bool = False) -> dict[str, Any]:
        params = {"force": "true"} if force else None
        return self._check(
            self._http.delete(f"/v1/roots/{self.root}/files/{rel_path}", params=params)
        )

    def move(self, old_rel_path: str, new_entry: dict[str, Any]) -> dict[str, Any]:
        body = {
            "to": new_entry["rel_path"],
            "size": new_entry["size"],
            "mtime_ns": new_entry["mtime_ns"],
            "blake3": new_entry["blake3"],
        }
        return self._check(
            self._http.post(f"/v1/roots/{self.root}/files/{old_rel_path}/move", json=body)
        )

    def heartbeat(self, info: dict[str, Any]) -> None:
        self._check(self._http.post(f"/v1/roots/{self.root}/heartbeat", json=info))


def _detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(payload, dict):
        return str(payload.get("detail") or payload.get("error") or payload)[:200]
    return str(payload)[:200]
