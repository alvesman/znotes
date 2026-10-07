"""Integration tests for ``DbChunkStore`` (task 29).

A temporary source root with one long markdown note and a two-page PDF is
scanned and drained through the real ``extract``/``chunk`` handlers; the tests
then read the committed ``chunk`` rows through :class:`DbChunkStore` and check
that sections come back across pages in ordinal order and page queries filter
by page.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pymupdf
import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from mynotes.config import Settings
from mynotes.db.models import (
    Chunk,
    Document,
    DocumentVersion,
    File,
    Job,
    ScanRun,
    SourceRoot,
    Tag,
    TagAlias,
    Workspace,
)
from mynotes.db.session import session_scope
from mynotes.jobs.worker import Worker
from mynotes.retrieval.assemble import DbChunkStore
from mynotes.scan.scanner import scan_root

pytestmark = pytest.mark.integration

_MTIME = 1_000_000_000_000_000_000
_JOB_TYPES = (
    "extract",
    "chunk",
    "embed",
    "version_detect",
    "tag_llm",
    "link_resolve",
    "kg_extract",
)

_LONG_NOTE = (
    "---\ntitle: Manual da Maquina\ntags: [Trabalho]\ndate: 2026-09-01\n---\n\n"
    "# Manual\n\n"
    + "\n\n".join(
        f"Paragrafo {number} da manutencao descreve o procedimento completo de verificacao "
        "semanal da maquina com todos os passos necessarios anotados com cuidado."
        for number in range(1, 61)
    )
    + "\n"
)


@dataclass
class AssembleEnv:
    """The registered test root plus the settings the pipeline reads."""

    settings: Settings
    root_id: int
    workspace_id: int
    root_path: Path


@pytest.fixture
def env(
    settings_with_db: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[AssembleEnv]:
    """Container settings with a fast settle time and a temporary source root."""
    settings_with_db.scan.settle_seconds = 0
    monkeypatch.setattr("mynotes.scan.scanner.get_settings", lambda: settings_with_db)
    monkeypatch.setattr("mynotes.jobs.worker.get_settings", lambda: settings_with_db)
    with session_scope() as session:
        workspace = Workspace(name=f"assemble-{uuid4().hex[:8]}")
        session.add(workspace)
        session.flush()
        root = SourceRoot(workspace_id=workspace.id, name="root", abs_path=str(tmp_path))
        session.add(root)
        session.flush()
        state = AssembleEnv(
            settings=settings_with_db,
            root_id=root.id,
            workspace_id=workspace.id,
            root_path=tmp_path,
        )
    yield state
    with session_scope() as session:
        session.execute(delete(Job).where(Job.type.in_(_JOB_TYPES)))
        session.execute(delete(File).where(File.source_root_id == state.root_id))
        session.execute(delete(Document).where(Document.source_root_id == state.root_id))
        session.execute(delete(ScanRun).where(ScanRun.source_root_id == state.root_id))
        session.execute(delete(SourceRoot).where(SourceRoot.id == state.root_id))
        session.execute(delete(TagAlias).where(TagAlias.workspace_id == state.workspace_id))
        session.execute(delete(Tag).where(Tag.workspace_id == state.workspace_id))
        session.execute(delete(Workspace).where(Workspace.id == state.workspace_id))


def write_file(path: Path, content: str, mtime_ns: int = _MTIME) -> None:
    """Create a file with a fixed mtime so scans are deterministic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))


def write_pdf(path: Path, pages: list[list[tuple[str, float]]], mtime_ns: int = _MTIME) -> None:
    """Write a small PDF with one ``insert_text`` line per tuple (text, size)."""
    document = pymupdf.open()
    try:
        for lines in pages:
            page = document.new_page()
            y = 100.0
            for text, size in lines:
                page.insert_text((72, y), text, fontsize=size, fontname="helv")
                y += size + 38.0
        document.save(path)
    finally:
        document.close()
    os.utime(path, ns=(mtime_ns, mtime_ns))


def ingest(env: AssembleEnv) -> None:
    """Scan the root and drain the extract/chunk queue."""
    scan_root(env.root_id)
    worker = Worker(concurrency=1, poll_interval_s=0.05, types=["extract", "chunk"])
    while worker.run_once():
        pass


def _single_document_version(
    session: Session, env: AssembleEnv
) -> tuple[Document, DocumentVersion]:
    """The only document and version of the one-file test root."""
    document = session.execute(
        select(Document).where(Document.source_root_id == env.root_id)
    ).scalar_one()
    version = session.execute(
        select(DocumentVersion).where(DocumentVersion.document_id == document.id)
    ).scalar_one()
    return document, version


def _committed_chunks(
    session: Session, document: Document, version: DocumentVersion
) -> list[Chunk]:
    """All chunks of the version, ordered by ordinal."""
    return list(
        session.execute(
            select(Chunk)
            .where(
                Chunk.document_id == document.id,
                Chunk.document_version_id == version.id,
            )
            .order_by(Chunk.ordinal)
        )
        .scalars()
        .all()
    )


def test_section_chunks_returns_markdown_section_in_ordinal_order(env: AssembleEnv) -> None:
    """The ingested note's section chunks match the DB rows, ordered by ordinal."""
    write_file(env.root_path / "manual.md", _LONG_NOTE)
    ingest(env)

    with session_scope() as session:
        document, version = _single_document_version(session, env)
        chunks = _committed_chunks(session, document, version)
        section = next(chunk.section for chunk in chunks if chunk.section)
        expected = [(chunk.ordinal, chunk.text) for chunk in chunks if chunk.section == section]

        store = DbChunkStore(session)
        rows = store.section_chunks(document.id, version.id, section)
        whole_document = store.page_chunks(document.id, version.id, None, None)

        assert len(rows) >= 2
        assert [(row.ordinal, row.text) for row in rows] == expected
        assert [row.ordinal for row in rows] == sorted(row.ordinal for row in rows)
        assert all(row.page is None and row.slide is None for row in rows)
        assert all(row.token_count > 0 for row in rows)
        assert [(row.ordinal, row.text) for row in whole_document] == [
            (chunk.ordinal, chunk.text) for chunk in chunks
        ]


def test_section_chunks_spans_pdf_pages_in_ordinal_order(env: AssembleEnv) -> None:
    """One heading's section is returned across pages 1 and 2 in ordinal order."""
    write_pdf(
        env.root_path / "guia.pdf",
        [
            [
                ("1. INSTALACAO E PRIMEIROS PASSOS", 11.0),
                ("Ligue o aparelho na tomada e aguarde o parametro de aquecimento inicial.", 11.0),
                ("Selecione o modo desejado no painel e confirme o programa escolhido.", 11.0),
                ("Verifique a vedacao da porta antes de iniciar qualquer ciclo completo.", 11.0),
            ],
            [
                ("Ajuste a temperatura conforme a tabela do fabricante do equipamento.", 11.0),
                ("Ao final do ciclo deixe a porta entreaberta para secar por completo.", 11.0),
                ("Registre a manutencao no caderno de ocorrencias do equipamento usado.", 11.0),
                ("Repita o procedimento a cada trinta dias de uso continuo da maquina.", 11.0),
            ],
        ],
    )
    ingest(env)

    with session_scope() as session:
        document, version = _single_document_version(session, env)
        chunks = _committed_chunks(session, document, version)
        section = chunks[-1].section
        assert section, "expected the page-2 continuation chunks to carry a section"
        assert {chunk.page for chunk in chunks if chunk.section == section} >= {1, 2}

        store = DbChunkStore(session)
        rows = store.section_chunks(document.id, version.id, section)
        page_two = store.page_chunks(document.id, version.id, 2, None)

        assert {row.page for row in rows} >= {1, 2}
        assert [(row.ordinal, row.text, row.page) for row in rows] == [
            (chunk.ordinal, chunk.text, chunk.page) for chunk in chunks if chunk.section == section
        ]
        assert [row.ordinal for row in rows] == sorted(row.ordinal for row in rows)
        assert [row.ordinal for row in page_two] == [
            chunk.ordinal for chunk in chunks if chunk.page == 2
        ]
        assert all(row.page == 2 and row.slide is None for row in page_two)
