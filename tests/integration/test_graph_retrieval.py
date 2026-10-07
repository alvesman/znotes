"""Integration tests for graph expansion in retrieval (task 43).

The full stack runs: a temporary source root with two notes is ingested with a
``FakeEmbedder`` (dim 8), the KG is seeded through the real ``kg_extract`` →
``resolve_entities`` → ``kg_temporal_refresh`` path with a ``FakeChat``
returning fixed extraction JSON, everything is synced into per-test OpenSearch
aliases and searched through :func:`~mynotes.retrieval.pipeline.retrieve` with
a ``FakeReranker``. Note A says that the Projeto Alfa uses Rust; note B says
Rust was idealized by Ana and shares **no word** with the question, so only the
graph can surface it.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from testcontainers.community.opensearch import OpenSearchContainer

from fakes import FakeChat, FakeEmbedder, FakeReranker
from mynotes.config import Settings
from mynotes.db.models import (
    Chunk,
    Document,
    DocumentVersion,
    EmbeddingCache,
    File,
    IndexOutbox,
    Job,
    KgEntity,
    KgEntityAlias,
    KgEntityDistinct,
    KgExtractionCache,
    KgMention,
    KgRelation,
    KgRelationEvidence,
    ScanRun,
    SourceRoot,
    Tag,
    TagAlias,
    Workspace,
)
from mynotes.db.session import reset_engine, session_scope
from mynotes.graph.expand import fetch_subgraph
from mynotes.graph.ontology import EXTRACTOR_VERSION
from mynotes.index.opensearch import ensure_indices, ensure_pipeline, get_client
from mynotes.index.sync import sync_once
from mynotes.jobs import pipeline
from mynotes.jobs.queue import complete
from mynotes.jobs.registry import JobContext
from mynotes.llm.chat import ChatMessage
from mynotes.retrieval.graph_expand import register as register_graph
from mynotes.retrieval.pipeline import retrieve
from mynotes.scan.scanner import scan_root

pytestmark = pytest.mark.integration

_MTIME = 1_000_000_000_000_000_000
_JOB_TYPES = (
    "extract",
    "chunk",
    "embed",
    "apply_retention",
    "reindex_metadata",
    "delete_document",
    "version_detect",
    "tag_llm",
    "link_resolve",
    "kg_extract",
    "resolve_entities",
    "adjudicate_entities",
    "kg_temporal_refresh",
    "refresh_summaries",
)

_PATCH_TARGETS = (
    "mynotes.config.get_settings",
    "mynotes.db.session.get_settings",
    "mynotes.scan.scanner.get_settings",
    "mynotes.jobs.worker.get_settings",
    "mynotes.llm.limits.get_settings",
    "mynotes.index.sync.get_settings",
    "mynotes.index.opensearch.get_settings",
    "mynotes.index.documents.get_settings",
    "mynotes.graph.resolve.get_settings",
    "mynotes.graph.store.get_settings",
    "mynotes.embed.client.get_settings",
    "mynotes.retrieval.hybrid.get_settings",
    "mynotes.retrieval.analyze.get_settings",
    "mynotes.retrieval.pipeline.get_settings",
    "mynotes.retrieval.entity_link.get_settings",
    "mynotes.retrieval.graph_expand.get_settings",
)

_QUESTION = "Quem propôs a tecnologia usada no Projeto Alfa?"

_NOTE_ALFA = """---
title: Projeto Alfa
date: 2026-01-10
---

# Contexto

O Projeto Alfa moderniza o controle de acesso do laboratorio com crachas
digitais e auditoria automatica das portas de todas as salas. A tecnologia
escolhida para o novo servico de indexacao e usada em producao desde março e o
desempenho convenceu o comite. A primeira fase cobre o cadastro dos visitantes
externos e o Congresso de outubro já tem data marcada na agenda da equipe.
"""

_NOTE_HISTORIA = """---
title: Historia Rust
date: 2026-06-15
---

# Origem

Linguagem Rust foi idealizada pela engenheira Ana quando o laboratorio ainda
testava alternativas ao leitor legado de crachas. O desenho inicial ganhou uma
versao interna em 2015 e o compilador atual manteve as garantias de memoria
que convenceram o comite tecnico. Ana também escreveu uma ata da reuniao de
aprovacao com medidas de desempenho que justificaram essa escolha final.
"""

# Filler note without words in common with the question and without KG
# entities: it sits between the two bridge-note chunks in the k-NN order so
# the comparison with the flag off has something to rank note B behind.
_NOTE_ROTINA = """---
title: Rotina do laboratorio
date: 2026-03-02
---

# Rotina

Laboratorio trocou fechaduras: vigias noturnas nas portas, revistas do
arquivo com etiquetas coladas, crachas novos para equipe tecnica.
"""

_ALFA_PATH = "projetos/alfa.md"
_HISTORIA_PATH = "privado/historia.md"
_ROTINA_PATH = "geral/rotina.md"


def _extraction_alfa() -> str:
    """The Alfa group: Projeto Alfa uses Rust."""
    import json

    return json.dumps(
        {
            "entities": [
                {
                    "name": "Projeto Alfa",
                    "type": "Project",
                    "description": "controle de acesso do laboratorio",
                    "aliases": [],
                },
                {
                    "name": "Rust",
                    "type": "Technology",
                    "description": "linguagem do servico de indexacao",
                    "aliases": [],
                },
            ],
            "relations": [
                {
                    "source": "Projeto Alfa",
                    "target": "Rust",
                    "type": "uses",
                    "raw_label": "usada",
                    "description": "",
                    "attribute": None,
                    "value": None,
                    "quote": "usada em producao",
                    "confidence": 0.9,
                }
            ],
        }
    )


def _extraction_historia() -> str:
    """The history group: Ana proposed Rust."""
    import json

    return json.dumps(
        {
            "entities": [
                {
                    "name": "Rust",
                    "type": "Technology",
                    "description": "linguagem do servico de indexacao",
                    "aliases": [],
                },
                {
                    "name": "Ana",
                    "type": "Person",
                    "description": "engenheira",
                    "aliases": [],
                },
            ],
            "relations": [
                {
                    "source": "Ana",
                    "target": "Rust",
                    "type": "proposes",
                    "raw_label": "idealizada pela",
                    "description": "",
                    "attribute": None,
                    "value": None,
                    "quote": "idealizada pela engenheira Ana",
                    "confidence": 0.9,
                }
            ],
        }
    )


def _extraction_rotina() -> str:
    """The filler note contributes no entities or relations."""
    import json

    return json.dumps({"entities": [], "relations": []})


def _kg_chat_answers() -> Callable[[Sequence[ChatMessage]], str]:
    """Extractions in ingestion order: Alfa, Historia, then empty fillers."""
    answers = [_extraction_alfa(), _extraction_historia()]

    def answer(_messages: Sequence[ChatMessage]) -> str:
        return answers.pop(0) if answers else _extraction_rotina()

    return answer


class _SteeredEmbedder:
    """Embedder with per-text vector overrides (query texts only).

    The 8-dim :class:`FakeEmbedder` maps every token onto 8 buckets, so a
    question shares buckets with unrelated notes and the k-NN leg surfaces
    them regardless of content. Pinning the question vector to the filler
    note's chunk vector emulates a well-behaved embedder: the filler is the
    question's nearest neighbour and the bridge note (no words in common with
    the question) drops out of the k-NN top ranks, so only the graph can
    surface it.
    """

    def __init__(self, inner: FakeEmbedder, overrides: dict[str, list[float]]) -> None:
        self._inner = inner
        self._overrides = overrides
        self.model = inner.model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str, task: str = "retrieval") -> list[float]:
        if text in self._overrides:
            return list(self._overrides[text])
        return self._inner.embed_query(text, task=task)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine of two equal-length vectors."""
    return sum(x * y for x, y in zip(left, right, strict=True))


@dataclass
class Env:
    """The registered test root, the fakes and the test's aliases."""

    settings: Settings
    fake: FakeEmbedder
    reranker: FakeReranker
    root_id: int
    workspace_id: int
    root_path: Path
    chunks_alias: str


@pytest.fixture
def env(
    migrated_dsn: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    os_container: OpenSearchContainer,
) -> Iterator[Env]:
    """Container settings (dim 8, fast settle, grace 0) and per-test aliases."""
    settings = Settings(postgres={"dsn": migrated_dsn})
    config = os_container.get_config()
    suffix = uuid4().hex[:8]
    settings.opensearch.url = f"http://{config['host']}:{config['port']}"
    settings.opensearch.chunks_alias = f"t{suffix}_chunks"
    settings.opensearch.entities_alias = f"t{suffix}_entities"
    settings.models.embedding.dim = 8
    settings.scan.settle_seconds = 0
    settings.scan.delete_grace_seconds = 0
    settings.retrieval.min_rerank_score = 0.005
    settings.retrieval.graph_expansion = True
    for target in _PATCH_TARGETS:
        monkeypatch.setattr(target, lambda: settings)
    fake = FakeEmbedder(dim=8)
    settings.models.embedding.model = fake.model
    monkeypatch.setattr("mynotes.embed.client.get_embedder", lambda: fake)
    monkeypatch.setattr("mynotes.retrieval.hybrid.get_embedder", lambda: fake)
    reranker = FakeReranker()
    monkeypatch.setattr("mynotes.retrieval.pipeline.get_reranker", lambda: reranker)
    reset_engine()
    with session_scope() as session:
        workspace = Workspace(name=f"graph-{suffix}")
        session.add(workspace)
        session.flush()
        root = SourceRoot(workspace_id=workspace.id, name="root", abs_path=str(tmp_path))
        session.add(root)
        session.flush()
        settings.workspace = workspace.name
        state = Env(
            settings=settings,
            fake=fake,
            reranker=reranker,
            root_id=root.id,
            workspace_id=workspace.id,
            root_path=tmp_path,
            chunks_alias=settings.opensearch.chunks_alias,
        )
    ensure_indices(get_client(), settings)
    ensure_pipeline(get_client(), settings)
    register_graph()
    yield state
    from mynotes.retrieval.assemble import unregister_extra_block
    from mynotes.retrieval.pipeline import unregister_expander

    unregister_expander("graph")
    unregister_extra_block("graph_facts")
    client = get_client()
    for alias in (settings.opensearch.chunks_alias, settings.opensearch.entities_alias):
        try:
            physical = list(client.indices.get(index=f"{alias}_v*"))
        except Exception:
            physical = []
        for name in physical:
            client.indices.delete(index=name)
    with session_scope() as session:
        session.execute(delete(KgEntityDistinct))
        session.execute(delete(KgRelationEvidence))
        session.execute(delete(KgRelation))
        session.execute(delete(KgMention))
        session.execute(delete(KgEntityAlias))
        session.execute(delete(KgEntity))
        session.execute(delete(Job).where(Job.type.in_(_JOB_TYPES)))
        session.execute(delete(IndexOutbox))
        session.execute(
            delete(KgExtractionCache).where(
                KgExtractionCache.extractor_version == EXTRACTOR_VERSION
            )
        )
        session.execute(delete(EmbeddingCache).where(EmbeddingCache.model == fake.model))
        session.execute(delete(TagAlias).where(TagAlias.workspace_id == state.workspace_id))
        session.execute(delete(Tag).where(Tag.workspace_id == state.workspace_id))
        session.execute(delete(File).where(File.source_root_id == state.root_id))
        session.execute(delete(Document).where(Document.source_root_id == state.root_id))
        session.execute(delete(ScanRun).where(ScanRun.source_root_id == state.root_id))
        session.execute(delete(SourceRoot).where(SourceRoot.id == state.root_id))
        session.execute(delete(Workspace).where(Workspace.id == state.workspace_id))
    reset_engine()


def write_file(path: Path, content: str, mtime_ns: int = _MTIME) -> None:
    """Create a file with a fixed mtime so scans are deterministic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))


def drain_extract_chunk() -> None:
    """Run the worker until no extract/chunk job is claimable."""
    from mynotes.jobs.worker import Worker

    worker = Worker(concurrency=1, poll_interval_s=0.05, types=["extract", "chunk"])
    while worker.run_once():
        pass


def run_queued(env: Env, job_type: str) -> int:
    """Run every queued job of one type through its handler; return the count."""
    handled = 0
    while True:
        with session_scope() as session:
            job = (
                session.execute(
                    select(Job)
                    .where(Job.type == job_type, Job.status == "queued")
                    .order_by(Job.id)
                    .limit(1)
                )
                .scalars()
                .first()
            )
            if job is None:
                return handled
            job_id = job.id
            payload = dict(job.payload)
        context = JobContext(worker_id="test", settings=env.settings, embedder=env.fake)
        getattr(pipeline, f"handle_{job_type}")(payload, context)
        # Mark the job done; otherwise the same queued job is picked again forever.
        with session_scope() as session:
            complete(session, job_id)
        handled += 1


def patch_chat(monkeypatch: pytest.MonkeyPatch, chat: FakeChat) -> None:
    """Point the pipeline's ``get_chat`` at the fake."""
    monkeypatch.setattr("mynotes.jobs.pipeline.get_chat", lambda purpose="extract": chat)


def refresh(env: Env) -> None:
    """Make the synced chunks and entities visible to search."""
    client = get_client()
    client.indices.refresh(index=env.chunks_alias)
    client.indices.refresh(index=env.settings.opensearch.entities_alias)


@pytest.fixture
def graph_env(env: Env, monkeypatch: pytest.MonkeyPatch) -> Env:
    """Ingest both notes, extract the KG through the real handlers and sync."""
    chat = FakeChat(_kg_chat_answers())
    patch_chat(monkeypatch, chat)
    for rel_path, content in (
        (_ALFA_PATH, _NOTE_ALFA),
        (_HISTORIA_PATH, _NOTE_HISTORIA),
        (_ROTINA_PATH, _NOTE_ROTINA),
    ):
        write_file(env.root_path / rel_path, content)
        scan_root(env.root_id)
        drain_extract_chunk()
        with session_scope() as session:
            file_row = session.execute(
                select(File).where(File.source_root_id == env.root_id, File.rel_path == rel_path)
            ).scalar_one()
            document = session.get(Document, file_row.document_id)
            assert document is not None and document.latest_version_id is not None
            version_id = document.latest_version_id
        pipeline.handle_embed(
            {"version_id": version_id},
            JobContext(worker_id="test", settings=env.settings, embedder=env.fake),
        )
        run_queued(env, "kg_extract")
        run_queued(env, "resolve_entities")
        run_queued(env, "kg_temporal_refresh")
    with session_scope() as session:
        chunk_texts_by_path = {
            path: list(
                session.execute(
                    select(Chunk.text)
                    .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
                    .join(Document, Document.id == DocumentVersion.document_id)
                    .join(File, File.id == Document.current_file_id)
                    .where(
                        File.source_root_id == env.root_id,
                        File.rel_path == path,
                        DocumentVersion.is_latest.is_(True),
                    )
                ).scalars()
            )
            for path in (_HISTORIA_PATH, _ROTINA_PATH)
        }
    bridge_texts = chunk_texts_by_path[_HISTORIA_PATH]
    filler_texts = chunk_texts_by_path[_ROTINA_PATH]
    assert bridge_texts and filler_texts, "bridge or filler note has no chunks"
    steered = env.fake.embed_query(filler_texts[0])
    bridge_cos = max(_cosine(steered, env.fake.embed_query(text)) for text in bridge_texts)
    filler_cos = max(_cosine(steered, env.fake.embed_query(text)) for text in filler_texts)
    assert filler_cos > bridge_cos, (
        f"filler chunk must outrank the bridge note in k-NN: {filler_cos} <= {bridge_cos}"
    )
    embedder = _SteeredEmbedder(env.fake, {_QUESTION: list(steered)})
    monkeypatch.setattr("mynotes.retrieval.hybrid.get_embedder", lambda: embedder)
    monkeypatch.setattr("mynotes.embed.client.get_embedder", lambda: embedder)
    sync_once()
    refresh(env)
    return env


def entity_id_by_name(env: Env, name: str) -> int:
    """The active entity id with exactly this canonical name."""
    with session_scope() as session:
        row = (
            session.execute(select(KgEntity.id).where(KgEntity.canonical_name == name))
            .scalars()
            .all()
        )
    assert len(row) == 1, f"expected exactly one {name!r} entity, got {row}"
    return int(row[0])


def test_graph_expansion_surfaces_the_bridge_note(graph_env: Env) -> None:
    """Test 1: with graph expansion on, note B is a graph-sourced hit."""
    result = retrieve(_QUESTION)

    bridge = [hit for hit in result.hits if hit.path == _HISTORIA_PATH]
    assert bridge, f"note B missing from the hits: {[hit.path for hit in result.hits]}"
    assert any("graph" in hit.sources for hit in bridge)


def test_without_graph_expansion_the_bridge_note_stays_behind(graph_env: Env) -> None:
    """Test 1 (negative): with the flag off, note B is absent or ranked lower."""
    with_expansion = retrieve(_QUESTION)
    on_rank = min(
        (hit.rank for hit in with_expansion.hits if hit.path == _HISTORIA_PATH),
        default=None,
    )
    assert on_rank is not None, (
        f"note B missing with expansion on: {[hit.path for hit in with_expansion.hits]}"
    )
    assert any("graph" in hit.sources for hit in with_expansion.hits if hit.path == _HISTORIA_PATH)

    graph_env.settings.retrieval.graph_expansion = False
    without_expansion = retrieve(_QUESTION)
    off_rank = min(
        (hit.rank for hit in without_expansion.hits if hit.path == _HISTORIA_PATH),
        default=None,
    )

    assert off_rank is None or off_rank > on_rank
    assert all(
        "graph" not in hit.sources for hit in without_expansion.hits if hit.path == _HISTORIA_PATH
    )


def test_context_contains_the_graph_facts_block(graph_env: Env) -> None:
    """Test 2: the assembled context carries 'Projeto Alfa —uses→ Rust'."""
    from mynotes.retrieval.assemble import assemble_context

    result = retrieve(_QUESTION)
    bundle = assemble_context(result)

    assert "Graph facts" in bundle.text
    assert "Projeto Alfa —uses→ Rust" in bundle.text


def test_acl_filters_remove_the_bridge_note(graph_env: Env) -> None:
    """Test 3: an ACL excluding note B's folder wins over graph expansion."""
    from mynotes.retrieval.types import SearchFilters

    result = retrieve(_QUESTION, SearchFilters(acl_prefixes=["projetos"]))

    paths = [hit.path for hit in result.hits]
    assert _HISTORIA_PATH not in paths
    assert _ALFA_PATH in paths


def test_freshness_decays_the_older_evidence_edge(graph_env: Env) -> None:
    """With a half-life, the edge evidenced by the older note weighs less than the newer one."""
    rust = entity_id_by_name(graph_env, "Rust")
    ana = entity_id_by_name(graph_env, "Ana")
    alfa = entity_id_by_name(graph_env, "Projeto Alfa")
    now = datetime(2026, 7, 1, tzinfo=UTC)

    with session_scope() as session:
        plain = fetch_subgraph(session, [rust], hops=1, now=now)
        decayed = fetch_subgraph(session, [rust], hops=1, half_life_days=90, now=now)

    def weight(sub, src: int, dst: int) -> float:
        return next(w for s, d, _r, w in sub.edges if (s, d) == (src, dst))

    # Note A (Alfa uses Rust) is dated 2026-01-10, note B (Ana proposes Rust) 2026-06-15.
    alfa_plain, ana_plain = weight(plain, alfa, rust), weight(plain, ana, rust)
    alfa_decayed, ana_decayed = weight(decayed, alfa, rust), weight(decayed, ana, rust)
    assert alfa_decayed < alfa_plain and ana_decayed < ana_plain
    # 172 days old at a 90-day half-life vs 16 days: the older edge loses far more.
    assert alfa_decayed / alfa_plain == pytest.approx(0.5 ** (172 / 90), rel=1e-3)
    assert ana_decayed / ana_plain == pytest.approx(0.5 ** (16 / 90), rel=1e-3)


def test_as_of_before_valid_from_excludes_the_relation(graph_env: Env) -> None:
    """Test 4: history intent with as-of before a relation's valid_from excludes it."""
    rust = entity_id_by_name(graph_env, "Rust")
    ana = entity_id_by_name(graph_env, "Ana")

    with session_scope() as session:
        current = fetch_subgraph(session, [rust], hops=1)
        before = fetch_subgraph(session, [rust], hops=1, as_of=datetime(2026, 6, 1, tzinfo=UTC))
        after = fetch_subgraph(session, [rust], hops=1, as_of=datetime(2026, 7, 1, tzinfo=UTC))

    # The Ana —proposes→ Rust relation is evidenced in note B (2026-06-15), so
    # it only exists for as-of dates from that day on.
    assert (ana, rust) in [(src, dst) for src, dst, _r, _w in current.edges]
    assert (ana, rust) not in [(src, dst) for src, dst, _r, _w in before.edges]
    assert (ana, rust) in [(src, dst) for src, dst, _r, _w in after.edges]
