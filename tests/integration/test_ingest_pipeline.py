"""End to end through the ingest API: manifest, upload, extract, chunk, embed, search, rename,
delete.

The server side only: an agent's requests are replayed with the test client
against a remote root fed by a token, with the blob store in a temp dir.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import blake3
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select
from testcontainers.community.opensearch import OpenSearchContainer

from fakes import FakeEmbedder
from mynotes.api.app import create_app
from mynotes.config import Settings
from mynotes.core import roots, service, tokens
from mynotes.db.models import (
    AclGrant,
    ApiToken,
    Document,
    EmbeddingCache,
    File,
    IndexOutbox,
    Job,
    ScanRun,
    SourceRoot,
    Tag,
    TagAlias,
    Workspace,
)
from mynotes.db.session import reset_engine, session_scope
from mynotes.index.opensearch import ensure_indices, ensure_pipeline, get_client
from mynotes.index.sync import sync_once
from mynotes.jobs import pipeline
from mynotes.jobs.registry import JobContext
from mynotes.jobs.worker import Worker
from mynotes.retrieval.hybrid import hybrid_search
from mynotes.retrieval.types import SearchFilters

pytestmark = pytest.mark.integration

_NOTE_A = "# Frigorífico\n\nO frigorífico da cozinha está a 4 graus.\n"
_NOTE_B = "# Roadmap\n\nA primeira entrega é em março.\n"


@dataclass
class Env:
    settings: Settings
    fake: FakeEmbedder
    client: TestClient
    headers: dict[str, str]
    workspace_id: int
    root_id: int
    chunks_alias: str


@pytest.fixture
def env(
    migrated_dsn: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    os_container: OpenSearchContainer,
) -> Iterator[Env]:
    suffix = uuid4().hex[:8]
    settings = Settings(
        workspace=f"ingest-{suffix}",
        postgres={"dsn": migrated_dsn},
        storage={"dir": str(tmp_path / "blobs")},
        auth={"mode": "token"},
    )
    config = os_container.get_config()
    settings.opensearch.url = f"http://{config['host']}:{config['port']}"
    settings.opensearch.chunks_alias = f"t{suffix}_chunks"
    settings.opensearch.entities_alias = f"t{suffix}_entities"
    settings.models.embedding.dim = 8
    settings.scan.delete_grace_seconds = 0
    for target in (
        "mynotes.config.get_settings",
        "mynotes.db.session.get_settings",
        "mynotes.scan.scanner.get_settings",
        "mynotes.jobs.worker.get_settings",
        "mynotes.index.sync.get_settings",
        "mynotes.index.opensearch.get_settings",
        "mynotes.retrieval.hybrid.get_settings",
        "mynotes.storage.blobs.get_settings",
        "mynotes.api.auth.get_settings",
        "mynotes.api.routes.ingest.get_settings",
        "mynotes.core.tokens.get_settings",
        "mynotes.core.roots.get_settings",
        "mynotes.core.acl.get_settings",
        "mynotes.core.service.get_settings",
        "mynotes.web.routes.get_settings",
    ):
        monkeypatch.setattr(target, lambda: settings)
    fake = FakeEmbedder(dim=8)
    settings.models.embedding.model = fake.model
    monkeypatch.setattr("mynotes.retrieval.hybrid.get_embedder", lambda: fake)
    reset_engine()
    with session_scope() as session:
        workspace = Workspace(name=settings.workspace)
        session.add(workspace)
        session.flush()
        workspace_id = int(workspace.id)
    root_id = roots.add_remote_root("alice-notes", "alice")
    secret, _info = tokens.create_token("alice", "laptop", ["ingest:alice-notes", "web"])
    ensure_indices(get_client(), settings)
    ensure_pipeline(get_client(), settings)
    client = TestClient(create_app())
    state = Env(
        settings=settings,
        fake=fake,
        client=client,
        headers={"Authorization": f"Bearer {secret}"},
        workspace_id=workspace_id,
        root_id=root_id,
        chunks_alias=settings.opensearch.chunks_alias,
    )
    yield state
    os_client = get_client()
    for alias in (settings.opensearch.chunks_alias, settings.opensearch.entities_alias):
        try:
            physical = list(os_client.indices.get(index=f"{alias}_v*"))
        except Exception:
            physical = []
        for name in physical:
            os_client.indices.delete(index=name)
    with session_scope() as session:
        session.execute(
            delete(Job).where(
                Job.type.in_(
                    (
                        "extract",
                        "chunk",
                        "embed",
                        "version_detect",
                        "tag_llm",
                        "link_resolve",
                        "kg_extract",
                        "reindex_metadata",
                        "delete_document",
                        "apply_retention",
                    )
                )
            )
        )
        session.execute(delete(IndexOutbox).where(IndexOutbox.doc_kind == "chunk"))
        session.execute(delete(EmbeddingCache).where(EmbeddingCache.model == fake.model))
        session.execute(delete(File).where(File.source_root_id == root_id))
        session.execute(delete(Document).where(Document.source_root_id == root_id))
        session.execute(delete(ScanRun).where(ScanRun.source_root_id == root_id))
        session.execute(delete(ApiToken).where(ApiToken.workspace_id == workspace_id))
        session.execute(delete(AclGrant).where(AclGrant.workspace_id == workspace_id))
        session.execute(delete(SourceRoot).where(SourceRoot.id == root_id))
        session.execute(delete(TagAlias).where(TagAlias.workspace_id == workspace_id))
        session.execute(delete(Tag).where(Tag.workspace_id == workspace_id))
        session.execute(delete(Workspace).where(Workspace.id == workspace_id))
    reset_engine()


def _entry(rel: str, text: str, mtime_ns: int = 1_000) -> dict:
    data = text.encode("utf-8")
    return {
        "rel_path": rel,
        "size": len(data),
        "mtime_ns": mtime_ns,
        "blake3": blake3.blake3(data).hexdigest(),
    }


def _upload(env: Env, rel: str, text: str, mtime_ns: int = 1_000) -> dict:
    data = text.encode("utf-8")
    response = env.client.put(
        f"/v1/roots/alice-notes/files/{rel}",
        content=data,
        headers={
            **env.headers,
            "X-Content-Blake3": blake3.blake3(data).hexdigest(),
            "X-Mtime-Ns": str(mtime_ns),
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _drain(env: Env) -> None:
    worker = Worker(concurrency=1, poll_interval_s=0.05, types=["extract", "chunk"])
    while worker.run_once():
        pass
    with session_scope() as session:
        version_ids = list(
            session.execute(
                select(Document.latest_version_id).where(
                    Document.source_root_id == env.root_id, Document.status == "current"
                )
            ).scalars()
        )
    for version_id in version_ids:
        pipeline.handle_embed(
            {"version_id": version_id},
            JobContext(worker_id="test", settings=env.settings, embedder=env.fake),
        )
    sync_once()
    get_client().indices.refresh(index=env.chunks_alias)


def test_manifest_upload_search_rename_delete(env: Env) -> None:
    client, headers = env.client, env.headers

    # 1. A full manifest of a new folder: everything must be uploaded.
    manifest = client.post(
        "/v1/roots/alice-notes/manifest",
        json={
            "scope": "full",
            "entries": [_entry("ideas/a.md", _NOTE_A), _entry("plans/b.md", _NOTE_B)],
        },
        headers=headers,
    )
    assert manifest.status_code == 200, manifest.text
    assert manifest.json()["upload"] == ["ideas/a.md", "plans/b.md"]
    with session_scope() as session:
        assert (
            session.execute(select(File).where(File.source_root_id == env.root_id)).first() is None
        )

    # 2. Uploads create the rows (paths prefixed with the root's name) and queue extraction.
    assert _upload(env, "ideas/a.md", _NOTE_A)["action"] == "new"
    assert _upload(env, "plans/b.md", _NOTE_B)["action"] == "new"
    assert _upload(env, "plans/b.md", _NOTE_B)["action"] in ("unchanged", "touched")  # idempotent
    with session_scope() as session:
        paths = sorted(
            session.execute(
                select(File.rel_path).where(File.source_root_id == env.root_id)
            ).scalars()
        )
    assert paths == ["alice-notes/ideas/a.md", "alice-notes/plans/b.md"]

    # 3. The worker extracts from the blob store; the note becomes searchable.
    _drain(env)
    hits = hybrid_search("frigorífico cozinha graus", SearchFilters())
    assert hits and hits[0].path == "alice-notes/ideas/a.md"

    # 4. The original is served from the server's copy; opening it in an app is refused.
    document_id = hits[0].document_id
    raw = client.get(f"/v1/files/{document_id}/raw", headers=headers)
    assert raw.status_code == 200 and raw.content.decode("utf-8") == _NOTE_A
    assert service.get_document(str(document_id)).openable is False

    # 5. A second manifest with the same listing changes nothing.
    again = client.post(
        "/v1/roots/alice-notes/manifest",
        json={
            "scope": "full",
            "entries": [_entry("ideas/a.md", _NOTE_A), _entry("plans/b.md", _NOTE_B)],
        },
        headers=headers,
    ).json()
    assert again["upload"] == [] and again["counts"].get("unchanged") == 2

    # 6. A rename pairs the paths without a new upload.
    moved = client.post(
        "/v1/roots/alice-notes/files/plans/b.md/move",
        json={**_entry("plans/roadmap.md", _NOTE_B), "to": "plans/roadmap.md"},
        headers=headers,
    )
    assert moved.status_code == 200, moved.text
    assert moved.json() == {"action": "moved", "upload": []}
    with session_scope() as session:
        paths = sorted(
            session.execute(
                select(File.rel_path).where(
                    File.source_root_id == env.root_id, File.status == "active"
                )
            ).scalars()
        )
    assert paths == ["alice-notes/ideas/a.md", "alice-notes/plans/roadmap.md"]

    # 7. A deleted file goes missing first (delete grace); force deletes it at once.
    gone = client.delete("/v1/roots/alice-notes/files/plans/roadmap.md", headers=headers).json()
    assert gone["action"] == "missing"
    gone = client.delete(
        "/v1/roots/alice-notes/files/plans/roadmap.md?force=true", headers=headers
    ).json()
    assert gone["action"] == "deleted"
    with session_scope() as session:
        statuses = dict(
            session.execute(
                select(File.rel_path, File.status).where(File.source_root_id == env.root_id)
            ).all()
        )
    assert statuses["alice-notes/plans/roadmap.md"] == "deleted"

    # 8. Once the tombstone is purged, the orphan blob is garbage collected; the live one stays.
    from mynotes.jobs.maintenance import purge_tombstones
    from mynotes.storage.blobs import BlobStore
    from mynotes.storage.gc import collect_garbage

    store = BlobStore(Path(env.settings.storage.dir))
    hash_a = blake3.blake3(_NOTE_A.encode()).hexdigest()
    hash_b = blake3.blake3(_NOTE_B.encode()).hexdigest()
    assert store.exists(env.workspace_id, env.root_id, hash_a, "md")
    assert store.exists(env.workspace_id, env.root_id, hash_b, "md")
    env.settings.retention.deleted_days = 0
    with session_scope() as session:
        purge_tombstones(session, env.settings)
        counts = collect_garbage(session, store)
    assert counts["blobs_removed"] == 1
    assert store.exists(env.workspace_id, env.root_id, hash_a, "md")
    assert not store.exists(env.workspace_id, env.root_id, hash_b, "md")

    # 9. Status and heartbeat.
    assert (
        client.post(
            "/v1/roots/alice-notes/heartbeat", json={"hostname": "laptop"}, headers=headers
        ).status_code
        == 204
    )
    status = client.get("/v1/roots/alice-notes/status", headers=headers).json()
    assert status["files"]["active"] == 1 and status["agent_info"]["hostname"] == "laptop"
    assert status["agent_last_seen_at"] is not None


def test_a_token_for_another_root_is_refused(env: Env) -> None:
    response = env.client.get("/v1/roots/bob-notes/status", headers=env.headers)
    assert response.status_code == 403
    assert env.client.get("/v1/roots/alice-notes/status").status_code == 401  # no token at all
