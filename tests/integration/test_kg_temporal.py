"""Integration tests for KG temporal maintenance and summaries (task 42).

The setup mirrors ``test_entity_resolution.py``: a temporary source root is
ingested with a ``FakeEmbedder`` (dim 8), ``kg_extract`` runs with a
``FakeChat`` returning fixed extraction JSON, and the ``resolve_entities``,
``kg_temporal_refresh``, ``delete_document`` and ``refresh_summaries``
handlers run as the worker would. No OpenSearch container is needed: the
k-NN signal in resolution degrades to None without it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select
from typer.testing import CliRunner

from fakes import FakeChat, FakeEmbedder
from mynotes.cli import app
from mynotes.config import Settings
from mynotes.db.models import (
    DocRelation,
    Document,
    EmbeddingCache,
    File,
    IndexOutbox,
    Job,
    KgEntity,
    KgEntityAlias,
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
from mynotes.graph.ontology import EXTRACTOR_VERSION
from mynotes.graph.summaries import enqueue_refresh_summaries
from mynotes.graph.temporal import current_attribute_values, enqueue_temporal_refresh, timeline
from mynotes.jobs import pipeline
from mynotes.jobs.registry import JobContext
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

# Targets that bind get_settings at import time and must see the test settings.
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
)

_ALFA = {
    "name": "Projeto Alfa",
    "type": "Project",
    "description": "controle de acesso",
    "aliases": [],
}
_ANA = {"name": "Ana", "type": "Person", "description": "gestora de dados", "aliases": []}
_RELATORIOS = {
    "name": "Sistema Relatorios",
    "type": "Project",
    "description": "controle de metas",
    "aliases": [],
}

_BUDGET_PATH = "projects/alfa/budget.md"
_BUDGET_V2_PATH = "projects/alfa/budget_v2.md"
_ALFA_PATH = "projects/alpha/plano.md"
_ALFA_LOWER_PATH = "projects/alpha/reuniao.md"
_SYSTEM_PATH = "projects/system/relatorios.md"

_BUDGET_BODY = """O Projeto Alfa tem um orcamento de {value} aprovado para a fase inicial do
controle de acesso do laboratorio. A Ana coordena as compras e o Bruno acompanha
as notas fiscais dos leitores digitais. O valor cobre equipamentos, licencas e
horas de integracao com o leitor existente. A previsao e rever o orcamento no
proximo trimestre com base nos resultados do piloto das salas de reuniao."""

_BUDGET_NOTE = f"""---
title: Orcamento Alfa
date: 2026-03-10
---

# Orcamento

{_BUDGET_BODY.format(value="15k EUR")}"""

_BUDGET_V2_NOTE = f"""---
title: Orcamento Alfa
date: 2026-09-01
---

# Orcamento

{_BUDGET_BODY.format(value="20k EUR")}"""

_BUDGET_EDITED_NOTE = """---
title: Orcamento Alfa
date: 2026-05-20
---

# Andamento

O Projeto Alfa segue em piloto nas salas de reuniao e o cadastro dos visitantes
externos entrou em teste. A Ana consolida os apontamentos das equipes e o Bruno
ajusta a sensibilidade dos leitores digitais. A decisao sobre a expansao fica
para o proximo comite, depois da avaliacao dos primeiros resultados."""

_NOTE_ALFA = """---
title: Plano Alfa
date: 2026-09-01
---

# Contexto

O Projeto Alfa moderniza o controle de acesso do laboratorio com crachas
digitais e auditoria automatica das portas. A Ana coordena as frentes e o
Bruno cuida da integracao com o leitor existente. A primeira fase cobre as
salas de reuniao e o cadastro dos visitantes externos antes do Congresso."""

_NOTE_ALFA_LOWER = """---
title: Reuniao Alfa
date: 2026-09-02
---

# Ata

Na reuniao de hoje o projeto alfa avancou para a fase de piloto e o comite
aprovou a compra dos leitores digitais. A Ana apresentou o cronograma revisado
e o Bruno demonstrou a leitura em serie dos crachas de teste. O piloto comeca
na proxima segunda-feira e os resultados serao revisados na reuniao seguinte."""

_NOTE_SYSTEM = """---
title: Sistema Relatorios
date: 2026-09-01
---

# Contexto

O Sistema Relatorios substitui a planilha mensal da equipe de dados por um
servico proprio com filas, paineis diarios e exportacao para o time financeiro.
A Ana ficou responsavel pelo desenho dos indicadores e pelo treinamento das
pessoas que vao usar os paineis. O primeiro ciclo de entregas cobre o
inventario de relatorios e a revisao dos numeros com as areas de negocio."""


def _budget_relation(value: str) -> dict:
    return {
        "source": "Projeto Alfa",
        "target": "Projeto Alfa",
        "type": "related_to",
        "raw_label": "orcamento",
        "description": "",
        "attribute": "budget",
        "value": value,
        "quote": f"orcamento de {value}",
        "confidence": 0.9,
    }


def _relation(source: str, target: str, quote: str) -> dict:
    return {
        "source": source,
        "target": target,
        "type": "works_on",
        "raw_label": "ficou responsavel",
        "description": "",
        "attribute": None,
        "value": None,
        "quote": quote,
        "confidence": 0.9,
    }


def _extraction(entities: list[dict], relations: list[dict]) -> str:
    return json.dumps({"entities": entities, "relations": relations})


@dataclass
class Env:
    """The registered test root, the fake embedder and the test's settings."""

    settings: Settings
    fake: FakeEmbedder
    root_id: int
    workspace_id: int
    root_path: Path


@pytest.fixture
def env(migrated_dsn: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Env]:
    """Container settings (dim 8, fast settle, grace 0) and a disposable workspace."""
    settings = Settings(postgres={"dsn": migrated_dsn})
    fake = FakeEmbedder(dim=8)
    settings.models.embedding.dim = 8
    settings.models.embedding.model = fake.model
    settings.scan.settle_seconds = 0
    settings.scan.delete_grace_seconds = 0
    suffix = uuid4().hex[:8]
    settings.opensearch.chunks_alias = f"t{suffix}_chunks"
    settings.opensearch.entities_alias = f"t{suffix}_entities"
    for target in _PATCH_TARGETS:
        monkeypatch.setattr(target, lambda: settings)
    reset_engine()
    with session_scope() as session:
        workspace = Workspace(name=f"kgt-{suffix}")
        session.add(workspace)
        session.flush()
        root = SourceRoot(workspace_id=workspace.id, name="root", abs_path=str(tmp_path))
        session.add(root)
        session.flush()
        settings.workspace = workspace.name
        state = Env(
            settings=settings,
            fake=fake,
            root_id=root.id,
            workspace_id=workspace.id,
            root_path=tmp_path,
        )
    yield state
    with session_scope() as session:
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
        session.execute(delete(EmbeddingCache).where(EmbeddingCache.model == state.fake.model))
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


def ingest(env: Env, rel_path: str, content: str, mtime_ns: int = _MTIME) -> tuple[int, int]:
    """Write a file, scan, drain extract/chunk and embed its latest version."""
    write_file(env.root_path / rel_path, content, mtime_ns)
    scan_root(env.root_id)
    drain_extract_chunk()
    document_id, version_id = document_and_version(env, rel_path)
    pipeline.handle_embed(
        {"version_id": version_id},
        JobContext(worker_id="test", settings=env.settings, embedder=env.fake),
    )
    return document_id, version_id


def document_and_version(env: Env, rel_path: str) -> tuple[int, int]:
    """The document and latest version ids currently stored for ``rel_path``."""
    with session_scope() as session:
        file_row = session.execute(
            select(File).where(File.source_root_id == env.root_id, File.rel_path == rel_path)
        ).scalar_one()
        assert file_row.document_id is not None
        document = session.get(Document, file_row.document_id)
        assert document is not None and document.latest_version_id is not None
        return document.id, document.latest_version_id


def patch_chat(monkeypatch: pytest.MonkeyPatch, chat: FakeChat) -> None:
    """Point the pipeline's ``get_chat`` at the fake."""
    monkeypatch.setattr("mynotes.jobs.pipeline.get_chat", lambda purpose="extract": chat)


def run_handler(env: Env, job_type: str, payload: dict) -> None:
    """Run one pipeline handler directly, like the worker would."""
    context = JobContext(worker_id="test", settings=env.settings, embedder=env.fake)
    getattr(pipeline, f"handle_{job_type}")(payload, context)


def queued_payloads(job_type: str) -> list[dict]:
    """The payloads of the queued jobs of one type, in id order."""
    with session_scope() as session:
        jobs = (
            session.execute(select(Job).where(Job.type == job_type).order_by(Job.id))
            .scalars()
            .all()
        )
        return [dict(job.payload) for job in jobs]


def run_kg_pipeline(env: Env, version_id: int) -> None:
    """Run kg_extract, the version's resolve_entities and every temporal refresh."""
    run_handler(env, "kg_extract", {"version_id": version_id})
    for payload in queued_payloads("resolve_entities"):
        if payload.get("version_id") == version_id:
            run_handler(env, "resolve_entities", payload)
    for payload in queued_payloads("kg_temporal_refresh"):
        run_handler(env, "kg_temporal_refresh", payload)


def document_status(env: Env, document_id: int) -> str:
    """The current status of one document."""
    with session_scope() as session:
        document = session.get(Document, document_id)
        assert document is not None
        return str(document.status)


def entity_by_name(name: str) -> KgEntity | None:
    """The entity whose canonical name matches exactly, or None."""
    with session_scope() as session:
        entity = session.execute(
            select(KgEntity).where(KgEntity.canonical_name == name)
        ).scalar_one_or_none()
        if entity is None:
            return None
        session.expunge(entity)
        return entity


def relation_rows() -> list[tuple[str, str, str | None, str]]:
    """Detached ``(value, status, valid_to_iso, type)`` per attribute relation."""
    with session_scope() as session:
        relations = session.execute(select(KgRelation).order_by(KgRelation.id)).scalars().all()
        rows = [
            (
                relation.value,
                relation.status,
                relation.valid_to.date().isoformat() if relation.valid_to else None,
                relation.type,
            )
            for relation in relations
        ]
    return rows


def setup_superseded_budgets(env: Env, monkeypatch: pytest.MonkeyPatch) -> tuple[int, int]:
    """Ingest budget.md, then budget_v2.md superseding it, and refresh both.

    Returns:
        ``(old_doc_id, new_doc_id)``.
    """
    patch_chat(
        monkeypatch,
        FakeChat(
            [
                _extraction([_ALFA], [_budget_relation("15k EUR")]),
                _extraction([_ALFA], [_budget_relation("20k EUR")]),
            ]
        ),
    )
    old_doc, old_version = ingest(env, _BUDGET_PATH, _BUDGET_NOTE)
    run_kg_pipeline(env, old_version)
    new_doc, new_version = ingest(env, _BUDGET_V2_PATH, _BUDGET_V2_NOTE, _MTIME + 1_000_000_000)
    run_kg_pipeline(env, new_version)
    run_handler(env, "version_detect", {"document_id": new_doc})
    for payload in queued_payloads("kg_temporal_refresh"):
        run_handler(env, "kg_temporal_refresh", payload)
    return old_doc, new_doc


def test_superseded_budget_becomes_historical_and_new_value_wins(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test 1: the 20k relation is active, the 15k one historical, 20k is current."""
    old_doc, new_doc = setup_superseded_budgets(env, monkeypatch)
    assert document_status(env, old_doc) == "superseded"
    assert document_status(env, new_doc) == "current"

    assert sorted(relation_rows()) == [
        ("15k EUR", "historical", "2026-09-01", "related_to"),
        ("20k EUR", "active", None, "related_to"),
    ]

    alfa = entity_by_name("Projeto Alfa")
    assert alfa is not None
    with session_scope() as session:
        values = current_attribute_values(session, alfa.id)
        evidence = sorted(
            (row.doc_date.date().isoformat(), row.is_current)
            for row in session.execute(select(KgRelationEvidence)).scalars()
        )
    assert set(values) == {"budget"}
    assert values["budget"].value == "20k EUR"
    assert values["budget"].doc_date is not None
    assert values["budget"].doc_date.date().isoformat() == "2026-09-01"
    assert evidence == [("2026-03-10", False), ("2026-09-01", True)]


def test_deleting_the_successor_restores_the_older_document(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test 2: deleting budget_v2 removes the supersedes, the 15k value is active again."""
    old_doc, new_doc = setup_superseded_budgets(env, monkeypatch)

    (env.root_path / _BUDGET_V2_PATH).unlink()
    scan_root(env.root_id)
    scan_root(env.root_id)
    assert document_status(env, new_doc) == "deleted"
    run_handler(env, "delete_document", {"document_id": new_doc})
    for payload in queued_payloads("kg_temporal_refresh"):
        run_handler(env, "kg_temporal_refresh", payload)

    assert document_status(env, old_doc) == "current"
    with session_scope() as session:
        supersedes = session.execute(
            select(func.count())
            .select_from(DocRelation)
            .where(
                DocRelation.type == "supersedes",
                DocRelation.src_document_id == new_doc,
            )
        ).scalar_one()
    assert supersedes == 0, "the deleted successor's supersedes relation is gone"
    assert sorted(relation_rows()) == [
        ("15k EUR", "active", None, "related_to"),
        ("20k EUR", "stale", None, "related_to"),
    ]

    alfa = entity_by_name("Projeto Alfa")
    assert alfa is not None
    with session_scope() as session:
        values = current_attribute_values(session, alfa.id)
    assert values["budget"].value == "15k EUR"


def test_editing_in_place_makes_the_old_evidence_historical(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test 3: a new version without the fact demotes the relation to historical."""
    patch_chat(
        monkeypatch,
        FakeChat(
            [
                _extraction([_ALFA], [_budget_relation("15k EUR")]),
                _extraction([_ALFA], []),
            ]
        ),
    )
    document_id, version_id = ingest(env, _BUDGET_PATH, _BUDGET_NOTE)
    run_kg_pipeline(env, version_id)
    assert relation_rows() == [("15k EUR", "active", None, "related_to")]

    _doc, version2 = ingest(env, _BUDGET_PATH, _BUDGET_EDITED_NOTE, _MTIME + 1_000_000_000)
    assert version2 != version_id, "the edit must create a second version"
    assert {"document_id": document_id} in queued_payloads("kg_temporal_refresh"), (
        "the chunk hook enqueued a temporal refresh"
    )
    run_kg_pipeline(env, version2)

    assert relation_rows() == [("15k EUR", "historical", "2026-05-20", "related_to")]
    with session_scope() as session:
        is_current = (
            session.execute(
                select(KgRelationEvidence.is_current).where(
                    KgRelationEvidence.document_version_id == version_id
                )
            )
            .scalars()
            .all()
        )
    assert is_current == [False]


def test_refresh_summaries_skips_single_mention_entities(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test 4: two mentions get a background summary, a single mention none."""
    patch_chat(
        monkeypatch,
        FakeChat(
            [
                _extraction(
                    [_ALFA, _ANA],
                    [_relation("Ana", "Projeto Alfa", "A Ana coordena as frentes")],
                ),
                _extraction(
                    [_ALFA, _ANA],
                    [
                        _relation(
                            "Ana", "Projeto Alfa", "o projeto alfa avancou para a fase de piloto"
                        )
                    ],
                ),
                _extraction([_RELATORIOS], []),
            ]
        ),
    )
    notes = [
        (_ALFA_PATH, _NOTE_ALFA),
        (_ALFA_LOWER_PATH, _NOTE_ALFA_LOWER),
        (_SYSTEM_PATH, _NOTE_SYSTEM),
    ]
    for index, (rel_path, content) in enumerate(notes):
        _doc, version_id = ingest(env, rel_path, content, _MTIME + index)
        run_kg_pipeline(env, version_id)

    alfa = entity_by_name("Projeto Alfa")
    ana = entity_by_name("Ana")
    system = entity_by_name("Sistema Relatorios")
    assert alfa is not None and ana is not None and system is not None
    with session_scope() as session:
        counts = {
            entity.id: entity.mention_count_current
            for entity in session.execute(select(KgEntity)).scalars()
        }
    assert counts[alfa.id] >= 2 and counts[ana.id] >= 2 and counts[system.id] == 1

    chat = FakeChat(
        ["O Projeto Alfa moderniza o controle de acesso em piloto.", "A Ana coordena as frentes."]
    )
    patch_chat(monkeypatch, chat)
    run_handler(env, "refresh_summaries", {"entity_ids": [alfa.id, ana.id, system.id]})

    assert len(chat.calls) == 2, "the single-mention entity is skipped"
    assert chat.kwargs[0]["priority"] == "background"
    assert chat.kwargs[0]["max_tokens"] == 300
    first_prompt = chat.calls[0][1].content
    assert "Projeto Alfa" in first_prompt and "works_on" in first_prompt

    with session_scope() as session:
        refreshed = {
            entity.id: (entity.summary, entity.summary_stale)
            for entity in session.execute(select(KgEntity)).scalars()
        }
        entity_upserts = session.execute(
            select(func.count())
            .select_from(IndexOutbox)
            .where(IndexOutbox.doc_kind == "entity", IndexOutbox.op == "upsert")
        ).scalar_one()
    assert refreshed[alfa.id] == ("O Projeto Alfa moderniza o controle de acesso em piloto.", False)
    assert refreshed[ana.id][1] is False
    assert refreshed[system.id] == (None, True)
    assert entity_upserts >= 2


def test_timeline_lists_budget_values_in_date_order(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test 5: timeline shows both budget values (historical first) and documents."""
    setup_superseded_budgets(env, monkeypatch)
    alfa = entity_by_name("Projeto Alfa")
    assert alfa is not None

    with session_scope() as session:
        items = timeline(session, alfa.id)
    attributes = [item for item in items if item.kind == "attribute"]
    assert [(item.text, item.status) for item in attributes] == [
        ("budget=15k EUR", "historical"),
        ("budget=20k EUR", "active"),
    ]
    assert [item.date.date().isoformat() for item in attributes] == ["2026-03-10", "2026-09-01"]
    documents = [item for item in items if item.kind == "document"]
    assert sorted(item.document_path for item in documents) == [_BUDGET_PATH, _BUDGET_V2_PATH]

    runner = CliRunner()
    rendered = runner.invoke(app, ["graph", "timeline", "Projeto Alfa"])
    assert rendered.exit_code == 0, rendered.output
    assert "budget=15k EUR" in rendered.output and "budget=20k EUR" in rendered.output

    as_json = runner.invoke(app, ["graph", "timeline", "Projeto Alfa", "--json"])
    assert as_json.exit_code == 0, as_json.output
    rows = json.loads(as_json.output)
    assert [row["text"] for row in rows if row["kind"] == "attribute"] == [
        "budget=15k EUR",
        "budget=20k EUR",
    ]

    project_ref = runner.invoke(app, ["graph", "timeline", "project:Projeto Alfa"])
    assert project_ref.exit_code == 0, project_ref.output
    assert "budget=15k EUR" in project_ref.output

    unknown = runner.invoke(app, ["graph", "timeline", "project:Inexistente"])
    assert unknown.exit_code == 0
    assert "No timeline entries found" in unknown.output

    missing = runner.invoke(app, ["graph", "timeline", "Ninguem"])
    assert missing.exit_code == 1


def test_temporal_and_summary_jobs_dedupe_and_debounce(env: Env) -> None:
    """kg_temporal_refresh dedupes per document; refresh_summaries by batch key."""
    with session_scope() as session:
        enqueue_temporal_refresh(session, 4242)
        enqueue_temporal_refresh(session, 4242)
        assert (
            session.execute(
                select(func.count())
                .select_from(Job)
                .where(Job.type == "kg_temporal_refresh", Job.payload == {"document_id": 4242})
            ).scalar_one()
            == 1
        )
        job_id = enqueue_refresh_summaries(session, [7, 3])
        assert job_id is not None
        job = session.get(Job, job_id)
        assert job is not None
        assert job.payload == {"entity_ids": [3, 7]}
        assert job.run_after is not None
        assert job.run_after > datetime.now(UTC) + timedelta(minutes=9)
        assert enqueue_refresh_summaries(session, [3, 7]) is None, "the batch key dedupes"


def test_attribute_ties_go_to_confidence(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two current documents with the same date: the higher-confidence value wins."""
    patch_chat(
        monkeypatch,
        FakeChat(
            [
                _extraction([_ALFA], [_budget_relation("15k EUR")]),
                _extraction(
                    [_ALFA],
                    [{**_budget_relation("12k EUR"), "confidence": 0.95}],
                ),
            ]
        ),
    )
    _doc_a, version_a = ingest(env, _ALFA_PATH, _NOTE_ALFA)
    run_kg_pipeline(env, version_a)
    same_date_note = _NOTE_ALFA_LOWER.replace("date: 2026-09-02", "date: 2026-09-01")
    _doc_b, version_b = ingest(env, _ALFA_LOWER_PATH, same_date_note)
    run_kg_pipeline(env, version_b)

    alfa = entity_by_name("Projeto Alfa")
    assert alfa is not None
    with session_scope() as session:
        values = current_attribute_values(session, alfa.id)
    assert values["budget"].value == "12k EUR"
    assert values["budget"].history == [("15k EUR", values["budget"].doc_date)]
    assert {row[1] for row in relation_rows()} == {"active", "historical"}
