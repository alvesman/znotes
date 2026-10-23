"""The agent against a fake server: listing, hashing, manifests, uploads, batches."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import blake3
import httpx
import pytest

from mynotes.agent import sync as sync_mod
from mynotes.agent.client import AgentAuthError, ServerClient
from mynotes.agent.hashcache import HashCache
from mynotes.agent.sync import AgentSync
from mynotes.config import AgentConfig, ScanConfig
from mynotes.scan.watch import HintBatch


class FakeServer:
    """Answers the ingest routes; remembers manifests and uploads; asks for unknown hashes."""

    def __init__(self) -> None:
        self.known: dict[str, str] = {}  # rel_path -> blake3
        self.manifests: list[dict[str, Any]] = []
        self.uploads: list[tuple[str, bytes, dict[str, str]]] = []
        self.status_code = 200
        self.max_upload_mb = 200

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.headers.get("authorization") != "Bearer mn_secret":
            return httpx.Response(401, json={"detail": "invalid token"})
        if path.endswith("/status"):
            return httpx.Response(
                self.status_code,
                json={"root": "alice-notes", "files": {}, "max_upload_mb": self.max_upload_mb},
            )
        if path.endswith("/manifest"):
            payload = json.loads(request.content)
            self.manifests.append(payload)
            upload = [
                e["rel_path"]
                for e in payload["entries"]
                if e["blake3"] and self.known.get(e["rel_path"]) != e["blake3"]
            ]
            if payload["scope"] == "paths":
                listed = {e["rel_path"] for e in payload["entries"]}
                for gone in set(payload["paths"]) - listed:
                    self.known.pop(gone, None)
            return httpx.Response(
                200,
                json={
                    "scan_run_id": 1,
                    "counts": {"unchanged": len(payload["entries"]) - len(upload)},
                    "upload": upload,
                    "actions": {},
                    "skipped": [],
                },
            )
        if "/files/" in path and request.method == "PUT":
            rel = path.split("/files/", 1)[1]
            body = request.read()
            digest = request.headers["x-content-blake3"]
            if blake3.blake3(body).hexdigest() != digest:
                return httpx.Response(422, json={"detail": "hash_mismatch"})
            self.known[rel] = digest
            self.uploads.append((rel, body, dict(request.headers)))
            return httpx.Response(
                200, json={"action": "new", "content_hash": digest, "size": len(body)}
            )
        if path.endswith("/heartbeat"):
            return httpx.Response(204)
        return httpx.Response(404, json={"detail": f"no route {path}"})


def _write(folder: Path, rel: str, text: str) -> None:
    file = folder / rel
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf-8")
    old = 1_600_000_000  # well settled
    os.utime(file, (old, old))


@pytest.fixture
def world(tmp_path: Path) -> tuple[AgentSync, FakeServer, Path]:
    folder = tmp_path / "notes"
    _write(folder, "ideas/a.md", "# A")
    _write(folder, "plans/b.md", "# B")
    _write(folder, "photo.jpg", "not a note")
    server = FakeServer()
    client = ServerClient(
        "http://server", "mn_secret", "alice-notes", transport=httpx.MockTransport(server)
    )
    agent = AgentConfig(
        server_url="http://server", token="mn_secret", root="alice-notes", path=folder
    )
    scan = ScanConfig()
    return AgentSync(agent, scan, client, HashCache(tmp_path / "hashes.json")), server, folder


def test_full_sync_lists_hashes_and_uploads_what_the_server_asks(world) -> None:
    agent, server, folder = world
    report = agent.full_sync()
    assert report.listed == 2 and report.uploaded == 2 and report.failed == []
    assert sorted(e["rel_path"] for e in server.manifests[0]["entries"]) == [
        "ideas/a.md",
        "plans/b.md",
    ]
    assert server.manifests[0]["scope"] == "full"
    assert sorted(rel for rel, _body, _h in server.uploads) == ["ideas/a.md", "plans/b.md"]
    rel, body, headers = server.uploads[0]
    assert body == (folder / rel).read_bytes()
    assert headers["x-mtime-ns"] == str(1_600_000_000 * 10**9)

    second = agent.full_sync()  # nothing new: no uploads
    assert second.uploaded == 0 and second.counts == {"unchanged": 2}


def test_hash_cache_avoids_rehashing_unchanged_files(
    world, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _server, _folder = world
    calls: list[Path] = []
    real = sync_mod.hash_file
    monkeypatch.setattr(sync_mod, "hash_file", lambda path: calls.append(path) or real(path))
    agent.full_sync()
    agent.full_sync()
    assert len(calls) == 2  # once per file, the second sync came from the cache
    assert agent.cache.file is not None and agent.cache.file.exists()


def test_batches_sync_only_the_changed_paths(world) -> None:
    agent, server, folder = world
    agent.full_sync()
    _write(folder, "ideas/a.md", "# A changed")
    (folder / "plans" / "b.md").unlink()
    agent.handle_batch(HintBatch(0, {"ideas/a.md", "plans/b.md"}))
    last = server.manifests[-1]
    assert last["scope"] == "paths" and last["paths"] == ["ideas/a.md", "plans/b.md"]
    assert [e["rel_path"] for e in last["entries"]] == ["ideas/a.md"]
    assert server.uploads[-1][0] == "ideas/a.md"
    assert "plans/b.md" not in server.known

    agent.handle_batch(HintBatch(0, set(), overflow=True))
    assert server.manifests[-1]["scope"] == "full"


def test_bad_token_is_a_clear_error(tmp_path: Path) -> None:
    server = FakeServer()
    client = ServerClient(
        "http://server", "mn_wrong", "alice-notes", transport=httpx.MockTransport(server)
    )
    with pytest.raises(AgentAuthError):
        client.status()


def test_files_over_the_server_limit_are_skipped_and_reported_once(
    world, caplog: pytest.LogCaptureFixture
) -> None:
    agent, server, folder = world
    server.max_upload_mb = 1
    big = folder / "big.pdf"
    big.write_bytes(b"%PDF" + b"\0" * (1024 * 1024 + 10))
    os.utime(big, (1_600_000_000, 1_600_000_000))

    agent.learn_limits()
    with caplog.at_level("WARNING"):
        first = agent.full_sync()
        second = agent.full_sync()
    assert first.skipped_large == 1 and second.skipped_large == 1
    assert first.failed == [] and "big.pdf" not in {rel for rel, _b, _h in server.uploads}
    assert all("big.pdf" not in e["rel_path"] for e in server.manifests[-1]["entries"])
    assert sum("larger than the server accepts" in r.message for r in caplog.records) == 1
