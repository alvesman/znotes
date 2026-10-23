"""The ingest routes with the manifest logic faked: auth, validation and response shapes."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import blake3
import pytest
from fastapi.testclient import TestClient

from mynotes.api.app import create_app
from mynotes.api.auth import get_principal
from mynotes.api.routes import ingest
from mynotes.core.principal import Principal
from mynotes.scan import remote
from mynotes.storage.blobs import BlobStore

AGENT = Principal("alice", ["alice-notes"], ingest_roots=frozenset({"alice-notes"}))
ROOT = SimpleNamespace(id=7, workspace_id=1, name="alice-notes", kind="remote", enabled=True)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_principal] = lambda: AGENT
    monkeypatch.setattr(ingest, "_load_root", lambda name: ROOT if name == "alice-notes" else None)
    monkeypatch.setattr(ingest, "get_blob_store", lambda: BlobStore(tmp_path / "blobs"))
    monkeypatch.setattr(ingest, "session_scope", lambda: nullcontext(object()))
    return TestClient(app)


def test_other_roots_are_forbidden_or_unknown(client: TestClient) -> None:
    assert client.get("/v1/roots/bob-notes/status").status_code == 403
    client.app.dependency_overrides[get_principal] = lambda: Principal(
        "alice", ["x"], ingest_roots=frozenset({"alice-notes", "ghost"})
    )
    assert client.get("/v1/roots/ghost/status").status_code == 404


def test_manifest_returns_the_upload_list(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake_apply(session, root, entries, *, scope, paths, settings, store, now=None):
        seen.update(root=root.name, scope=scope, paths=paths, entries=entries)
        return remote.ManifestResult(
            scan_run_id=5,
            counts={"new": 1},
            upload=["ideas/a.md"],
            actions={"ideas/a.md": "pending_upload"},
        )

    monkeypatch.setattr(remote, "apply_manifest", fake_apply)
    response = client.post(
        "/v1/roots/alice-notes/manifest",
        json={
            "scope": "full",
            "entries": [{"rel_path": "ideas\\a.md", "size": 3, "mtime_ns": 1, "blake3": "ab" * 32}],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["upload"] == ["ideas/a.md"]
    assert seen["scope"] == "full" and seen["paths"] is None
    assert seen["entries"][0].rel_path == "ideas\\a.md"  # normalized by apply_manifest

    bad = client.post(
        "/v1/roots/alice-notes/manifest",
        json={"scope": "full", "entries": [{"rel_path": "../x.md", "size": 1, "mtime_ns": 1}]},
    )
    assert bad.status_code == 422


def test_upload_checks_the_hash_and_records_the_file(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: list[remote.ManifestEntry] = []
    monkeypatch.setattr(
        remote,
        "record_upload",
        lambda session, root, entry, *, settings, store: recorded.append(entry) or "new",
    )
    body = b"# hello"
    digest = blake3.blake3(body).hexdigest()
    ok = client.put(
        "/v1/roots/alice-notes/files/ideas/a.md",
        content=body,
        headers={"X-Content-Blake3": digest, "X-Mtime-Ns": "123"},
    )
    assert ok.status_code == 200, ok.text
    assert ok.json() == {"action": "new", "content_hash": digest, "size": 7}
    assert recorded[0].rel_path == "ideas/a.md" and recorded[0].mtime_ns == 123

    mismatch = client.put(
        "/v1/roots/alice-notes/files/ideas/b.md",
        content=body,
        headers={"X-Content-Blake3": "00" * 32, "X-Mtime-Ns": "1"},
    )
    assert mismatch.status_code == 422
    unsupported = client.put(
        "/v1/roots/alice-notes/files/photo.jpg",
        content=body,
        headers={"X-Content-Blake3": digest, "X-Mtime-Ns": "1"},
    )
    assert unsupported.status_code == 415


def test_delete_move_heartbeat_and_status(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        remote,
        "delete_path",
        lambda session, root, rel, *, force, settings, store: "deleted" if force else "missing",
    )
    monkeypatch.setattr(
        remote,
        "move_path",
        lambda session, root, old, entry, *, settings, store: remote.ManifestResult(
            scan_run_id=1, actions={entry.rel_path: "moved"}
        ),
    )
    beats: list[dict] = []
    monkeypatch.setattr(remote, "heartbeat", lambda session, root, info: beats.append(info))
    monkeypatch.setattr(
        remote, "root_status", lambda session, root: {"root": root.name, "files": {}}
    )

    assert client.delete("/v1/roots/alice-notes/files/ideas/a.md").json() == {"action": "missing"}
    assert client.delete("/v1/roots/alice-notes/files/ideas/a.md?force=true").json() == {
        "action": "deleted"
    }
    moved = client.post(
        "/v1/roots/alice-notes/files/ideas/a.md/move",
        json={"to": "ideas/b.md", "size": 1, "mtime_ns": 1, "blake3": "ab" * 32},
    )
    assert moved.json() == {"action": "moved", "upload": []}
    assert (
        client.post("/v1/roots/alice-notes/heartbeat", json={"hostname": "laptop"}).status_code
        == 204
    )
    assert beats[0]["hostname"] == "laptop"
    assert client.get("/v1/roots/alice-notes/status").json()["root"] == "alice-notes"


def test_upload_over_the_limit_is_refused_while_streaming(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from mynotes.config import Settings

    monkeypatch.setattr(ingest, "get_settings", lambda: Settings(storage={"max_upload_mb": 1}))
    monkeypatch.setattr(
        remote, "record_upload", lambda session, root, entry, *, settings, store: "new"
    )
    big = b"x" * (1024 * 1024 + 1)
    digest = blake3.blake3(big).hexdigest()

    declared = client.put(
        "/v1/roots/alice-notes/files/big.md",
        content=big,
        headers={"X-Content-Blake3": digest, "X-Mtime-Ns": "1"},
    )
    assert declared.status_code == 413

    def chunks():  # no Content-Length: the limit is enforced as the bytes arrive
        yield big[: 1024 * 1024]
        yield big[1024 * 1024 :]

    streamed = client.put(
        "/v1/roots/alice-notes/files/big.md",
        content=chunks(),
        headers={"X-Content-Blake3": digest, "X-Mtime-Ns": "1"},
    )
    assert streamed.status_code == 413
    assert list((tmp_path / "blobs" / "tmp").glob("*.part")) == []  # nothing left behind


def test_status_tells_the_agent_the_upload_limit(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mynotes.config import Settings

    monkeypatch.setattr(ingest, "get_settings", lambda: Settings(storage={"max_upload_mb": 50}))
    monkeypatch.setattr(remote, "root_status", lambda session, root: {"root": root.name})
    assert client.get("/v1/roots/alice-notes/status").json()["max_upload_mb"] == 50
