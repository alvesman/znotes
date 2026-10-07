"""Integration tests for the embed job, embedding cache and outbox writers (task 21).

Each test registers a temporary directory as a ``source_root``, scans it and
drains only the ``extract``/``chunk`` jobs with a ``Worker``; the ``embed``
handler then runs directly with a ``JobContext`` carrying a ``FakeEmbedder``
(dim 8), so no model endpoint is ever contacted. Handlers commit through
``session_scope``, so setup and assertions also go through ``session_scope``
and every test cleans up its rows afterwards.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from fakes import FakeEmbedder
from mynotes.config import Settings
from mynotes.db.models import (
    Chunk,
    Document,
    DocumentVersion,
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
from mynotes.db.session import session_scope
from mynotes.embed.cache import embed_key, get_cached
from mynotes.jobs import pipeline
from mynotes.jobs.registry import JobContext
from mynotes.jobs.worker import Worker
from mynotes.scan.scanner import scan_root

pytestmark = pytest.mark.integration

_MTIME = 1_000_000_000_000_000_000

_NOTE = """# Relatório Trimestral

Este documento reúne os números do trimestre fechado e as decisões aprovadas pela diretoria.

## Introdução
O trimestre trouxe crescimento moderado e uma demanda estável dos clientes corporativos.

## Metodologia
Os dados foram coletados semanalmente e validados pela controladoria antes da consolidação.

## Resultados
A receita cresceu doze por cento e a margem bruta manteve-se estável apesar do câmbio.

## Riscos
O principal risco é a dependência de um único fornecedor estrangeiro de componentes.

## Próximos passos
A próxima etapa é automatizar a conciliação contábil e contratar dois analistas novos.
"""

_EDITED_PARAGRAPH = (
    "A receita encolheu três por cento no período e a margem bruta caiu com o custo dos insumos."
)
_EDITED_NOTE = _NOTE.replace(
    "A receita cresceu doze por cento e a margem bruta manteve-se estável apesar do câmbio.",
    _EDITED_PARAGRAPH,
)


@dataclass
class EmbedEnv:
    """The registered test root plus the fake embedder the handler will use."""

    settings: Settings
    fake: FakeEmbedder
    root_id: int
    workspace_id: int
    root_path: Path


@pytest.fixture
def embed_env(
    settings_with_db: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[EmbedEnv]:
    """Container settings with a fast settle time, a temp root and a fake embedder."""
    settings_with_db.scan.settle_seconds = 0
    monkeypatch.setattr("mynotes.scan.scanner.get_settings", lambda: settings_with_db)
    monkeypatch.setattr("mynotes.jobs.worker.get_settings", lambda: settings_with_db)
    fake = FakeEmbedder(dim=8)
    with session_scope() as session:
        workspace = Workspace(name=f"embed-{uuid4().hex[:8]}")
        session.add(workspace)
        session.flush()
        root = SourceRoot(workspace_id=workspace.id, name="root", abs_path=str(tmp_path))
        session.add(root)
        session.flush()
        env = EmbedEnv(
            settings=settings_with_db,
            fake=fake,
            root_id=root.id,
            workspace_id=workspace.id,
            root_path=tmp_path,
        )
    yield env
    with session_scope() as session:
        session.execute(
            delete(Job).where(
                Job.type.in_(
                    (
                        "extract",
                        "chunk",
                        "embed",
                        "apply_retention",
                        "version_detect",
                        "tag_llm",
                        "link_resolve",
                        "kg_extract",
                    )
                )
            )
        )
        session.execute(delete(IndexOutbox).where(IndexOutbox.doc_kind == "chunk"))
        session.execute(delete(EmbeddingCache).where(EmbeddingCache.model == fake.model))
        session.execute(delete(File).where(File.source_root_id == env.root_id))
        session.execute(delete(Document).where(Document.source_root_id == env.root_id))
        session.execute(delete(ScanRun).where(ScanRun.source_root_id == env.root_id))
        session.execute(delete(SourceRoot).where(SourceRoot.id == env.root_id))
        session.execute(delete(TagAlias).where(TagAlias.workspace_id == env.workspace_id))
        session.execute(delete(Tag).where(Tag.workspace_id == env.workspace_id))
        session.execute(delete(Workspace).where(Workspace.id == env.workspace_id))


def write_file(path: Path, content: str, mtime_ns: int = _MTIME) -> None:
    """Create a file with a fixed mtime so scans are deterministic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))


def drain_extract_chunk(settings: Settings) -> None:
    """Run the worker until no extract/chunk job is claimable."""
    worker = Worker(concurrency=1, poll_interval_s=0.05, types=["extract", "chunk"])
    while worker.run_once():
        pass


def run_embed(env: EmbedEnv, version_id: int) -> None:
    """Run the embed handler with the fake embedder, like the worker would."""
    pipeline.handle_embed(
        {"version_id": version_id},
        JobContext(worker_id="test", settings=env.settings, embedder=env.fake),
    )


def document_of(env: EmbedEnv, session: Session) -> Document:
    """The only document created under the test root."""
    return session.execute(
        select(Document).where(Document.source_root_id == env.root_id)
    ).scalar_one()


def versions_of(session: Session, document_id: int) -> list[DocumentVersion]:
    """The document's versions ordered by version number."""
    return list(
        session.execute(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.version_no)
        ).scalars()
    )


def chunks_of(session: Session, version_id: int) -> list[Chunk]:
    """The version's chunks ordered by ordinal."""
    return list(
        session.execute(
            select(Chunk).where(Chunk.document_version_id == version_id).order_by(Chunk.ordinal)
        ).scalars()
    )


def outbox_rows(session: Session, op: str) -> list[IndexOutbox]:
    """The chunk outbox rows for one op, ordered by id."""
    return list(
        session.execute(
            select(IndexOutbox)
            .where(IndexOutbox.doc_kind == "chunk", IndexOutbox.op == op)
            .order_by(IndexOutbox.id)
        ).scalars()
    )


def cache_count(session: Session, model: str) -> int:
    """The number of cached vectors for one model."""
    return session.execute(
        select(func.count()).select_from(EmbeddingCache).where(EmbeddingCache.model == model)
    ).scalar_one()


def scan_and_prepare(env: EmbedEnv, content: str, mtime_ns: int = _MTIME) -> int:
    """Write the note, scan, drain extract/chunk and return the latest version id."""
    write_file(env.root_path / "relatorio.md", content, mtime_ns)
    scan_root(env.root_id)
    drain_extract_chunk(env.settings)
    with session_scope() as session:
        document = document_of(env, session)
        versions = versions_of(session, document.id)
        return versions[-1].id


def test_embed_marks_version_fills_cache_and_queues_upserts(embed_env: EmbedEnv) -> None:
    """Extract → chunk → embed leaves the version embedded with cache and outbox rows."""
    version_id = scan_and_prepare(embed_env, _NOTE)

    run_embed(embed_env, version_id)

    with session_scope() as session:
        version = session.get(DocumentVersion, version_id)
        assert version is not None
        assert version.status == "embedded"
        chunks = chunks_of(session, version_id)
        unique_texts = {chunk.embed_text for chunk in chunks}
        assert len(unique_texts) == len(chunks) == 6
        assert cache_count(session, embed_env.fake.model) == len(unique_texts)
        upserts = outbox_rows(session, "upsert")
        assert [row.doc_id for row in upserts] == [str(chunk.id) for chunk in chunks]
        assert all(row.index_name == "chunks" for row in upserts)
        assert outbox_rows(session, "delete") == []
        ordered_texts = [chunk.embed_text for chunk in chunks]
    assert embed_env.fake.calls == [ordered_texts]


def test_embed_rerun_skips_embedder_and_reenqueues(embed_env: EmbedEnv) -> None:
    """A second embed run makes no embedder calls but appends outbox rows again."""
    version_id = scan_and_prepare(embed_env, _NOTE)
    run_embed(embed_env, version_id)
    with session_scope() as session:
        chunk_count = len(chunks_of(session, version_id))
        first_upserts = len(outbox_rows(session, "upsert"))
        calls_before = [list(batch) for batch in embed_env.fake.calls]
        assert first_upserts == chunk_count

    run_embed(embed_env, version_id)

    with session_scope() as session:
        version = session.get(DocumentVersion, version_id)
        assert version is not None
        assert version.status == "embedded"
        assert len(outbox_rows(session, "upsert")) == 2 * first_upserts
        assert cache_count(session, embed_env.fake.model) == first_upserts
    assert embed_env.fake.calls == calls_before


def test_edit_reembeds_only_changed_chunk_and_deletes_old(embed_env: EmbedEnv) -> None:
    """After an edit only the changed chunk is embedded; the old chunks get deletes."""
    first_id = scan_and_prepare(embed_env, _NOTE)
    run_embed(embed_env, first_id)
    with session_scope() as session:
        old_chunks = chunks_of(session, first_id)
        old_ids = [chunk.id for chunk in old_chunks]
        old_texts = {chunk.embed_text for chunk in old_chunks}

    write_file(embed_env.root_path / "relatorio.md", _EDITED_NOTE, _MTIME + 5_000_000_000)
    scan_root(embed_env.root_id)
    drain_extract_chunk(embed_env.settings)

    with session_scope() as session:
        versions = versions_of(session, document_of(embed_env, session).id)
        assert [item.version_no for item in versions] == [1, 2]
        second_id = versions[1].id
        new_chunks = chunks_of(session, second_id)
        changed = [chunk for chunk in new_chunks if chunk.embed_text not in old_texts]
        assert len(new_chunks) == 6
        assert len(changed) == 1
        changed_text = changed[0].embed_text

    run_embed(embed_env, second_id)

    with session_scope() as session:
        version = session.get(DocumentVersion, second_id)
        assert version is not None
        assert version.status == "embedded"
        deletes = outbox_rows(session, "delete")
        assert {row.doc_id for row in deletes} == {str(chunk_id) for chunk_id in old_ids}
        upserts = outbox_rows(session, "upsert")
        assert {row.doc_id for row in upserts} == {str(chunk_id) for chunk_id in old_ids} | {
            str(chunk.id) for chunk in chunks_of(session, second_id)
        }
    assert len(embed_env.fake.calls) == 2
    assert embed_env.fake.calls[1] == [changed_text]


def test_identical_embed_texts_are_embedded_once(embed_env: EmbedEnv) -> None:
    """Two chunks sharing an embed_text produce one embedder call and one cache row."""
    text = "conteúdo idêntico em dois chunks do mesmo documento de teste improvisado."
    with session_scope() as session:
        document = Document(
            workspace_id=embed_env.workspace_id,
            source_root_id=embed_env.root_id,
            format="md",
            status="current",
        )
        session.add(document)
        session.flush()
        version = DocumentVersion(
            document_id=document.id,
            version_no=1,
            content_hash="hash-dup",
            rel_path_at_version="dup.md",
            mtime_ns=_MTIME,
            status="chunked",
            is_latest=True,
        )
        session.add(version)
        session.flush()
        document.latest_version_id = version.id
        for ordinal in (0, 1):
            session.add(
                Chunk(
                    document_version_id=version.id,
                    document_id=document.id,
                    ordinal=ordinal,
                    content_hash=f"hash-dup-{ordinal}",
                    text=text,
                    embed_text=text,
                    token_count=len(text.split()),
                )
            )
        version_id = version.id

    run_embed(embed_env, version_id)

    with session_scope() as session:
        version = session.get(DocumentVersion, version_id)
        assert version is not None
        assert version.status == "embedded"
        assert cache_count(session, embed_env.fake.model) == 1
        upserts = outbox_rows(session, "upsert")
        assert len(upserts) == 2
    assert embed_env.fake.calls == [[text]]


def test_cache_roundtrip_keeps_float16_precision(embed_env: EmbedEnv) -> None:
    """Vectors read back from the cache match the fake's output within 1e-3."""
    version_id = scan_and_prepare(embed_env, _NOTE)

    run_embed(embed_env, version_id)

    with session_scope() as session:
        chunks = chunks_of(session, version_id)
        keys = [embed_key(embed_env.fake, chunk.embed_text) for chunk in chunks]
        stored = get_cached(session, embed_env.fake.model, keys)
        texts = [chunk.embed_text for chunk in chunks]
    assert set(stored) == set(keys)
    expected = FakeEmbedder(dim=8).embed_documents(texts)
    for key, want in zip(keys, expected, strict=True):
        got = stored[key]
        assert len(got) == len(want) == 8
        assert max(abs(a - b) for a, b in zip(got, want, strict=True)) <= 1e-3


def test_embed_ignores_non_chunked_versions(embed_env: EmbedEnv) -> None:
    """A version that was never chunked ends the job quietly, without outbox rows."""
    with session_scope() as session:
        document = Document(
            workspace_id=embed_env.workspace_id,
            source_root_id=embed_env.root_id,
            format="md",
            status="current",
        )
        session.add(document)
        session.flush()
        version = DocumentVersion(
            document_id=document.id,
            version_no=1,
            content_hash="hash-pending",
            rel_path_at_version="pending.md",
            mtime_ns=_MTIME,
            status="pending",
            is_latest=True,
        )
        session.add(version)
        session.flush()
        version_id = version.id

    run_embed(embed_env, version_id)

    with session_scope() as session:
        version = session.get(DocumentVersion, version_id)
        assert version is not None
        assert version.status == "pending"
        assert outbox_rows(session, "upsert") == []
        assert outbox_rows(session, "delete") == []
    assert embed_env.fake.calls == []
