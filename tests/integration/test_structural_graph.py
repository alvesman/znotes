"""Integration tests for the structural graph (task 39).

A temporary source root holds three linked notes (A —wikilink→ B,
B —relative link→ C, C —wikilink→ A), a superseded version pair
(``idea.md``/``idea_v2.md`` from the fixtures tree) and an identical-content
duplicate. Files are scanned and drained through the pipeline with a
``FakeEmbedder`` (dim 8); the ``link_resolve`` handler writes the ``links_to``
relations and the structural queries run against real Postgres. OpenSearch is
not involved anywhere.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from typer.testing import CliRunner

from fakes import FakeEmbedder
from mynotes.cli import app
from mynotes.config import Settings
from mynotes.db.models import (
    DocRelation,
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
from mynotes.graph.links import documents_with_resolvable_links
from mynotes.graph.structural import find_document, linked_documents, neighbours
from mynotes.jobs import pipeline
from mynotes.jobs.registry import JobContext
from mynotes.jobs.worker import Worker
from mynotes.scan.scanner import scan_root

pytestmark = pytest.mark.integration

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "trees" / "versions"
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
)

runner = CliRunner()


@dataclass
class GraphEnv:
    """The registered test root, the fake embedder and the settings."""

    settings: Settings
    fake: FakeEmbedder
    root_id: int
    workspace_id: int
    root_path: Path


@pytest.fixture
def graph_env(
    migrated_dsn: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[GraphEnv]:
    """Container settings (dim 8, fast settle) and a fresh temporary root."""
    settings = Settings(postgres={"dsn": migrated_dsn})
    settings.models.embedding.dim = 8
    settings.scan.settle_seconds = 0
    settings.scan.delete_grace_seconds = 0
    settings.versions.min_interval_hours = 0.0
    settings.tags.llm_autotag = False
    for target in (
        "mynotes.config.get_settings",
        "mynotes.db.session.get_settings",
        "mynotes.scan.scanner.get_settings",
        "mynotes.jobs.worker.get_settings",
    ):
        monkeypatch.setattr(target, lambda: settings)
    fake = FakeEmbedder(dim=8)
    settings.models.embedding.model = fake.model
    reset_engine()
    with session_scope() as session:
        workspace = Workspace(name=f"graph-{uuid4().hex[:8]}")
        session.add(workspace)
        session.flush()
        root = SourceRoot(workspace_id=workspace.id, name="root", abs_path=str(tmp_path))
        session.add(root)
        session.flush()
        env = GraphEnv(
            settings=settings,
            fake=fake,
            root_id=root.id,
            workspace_id=workspace.id,
            root_path=tmp_path,
        )
    yield env
    with session_scope() as session:
        session.execute(delete(Job).where(Job.type.in_(_JOB_TYPES)))
        session.execute(delete(IndexOutbox).where(IndexOutbox.doc_kind == "chunk"))
        session.execute(delete(EmbeddingCache).where(EmbeddingCache.model == fake.model))
        session.execute(delete(File).where(File.source_root_id == env.root_id))
        session.execute(delete(Document).where(Document.source_root_id == env.root_id))
        session.execute(delete(ScanRun).where(ScanRun.source_root_id == env.root_id))
        session.execute(delete(SourceRoot).where(SourceRoot.id == env.root_id))
        session.execute(delete(TagAlias).where(TagAlias.workspace_id == env.workspace_id))
        session.execute(delete(Tag).where(Tag.workspace_id == env.workspace_id))
        session.execute(delete(Workspace).where(Workspace.id == env.workspace_id))
    reset_engine()


def install(
    env: GraphEnv,
    rel_path: str,
    *,
    content: str | None = None,
    source: str | None = None,
    mtime_ns: int = _MTIME,
) -> None:
    """Write a file into the root with a fixed mtime.

    Args:
        env: The test environment (provides the root path).
        rel_path: POSIX path relative to the root.
        content: Literal file content; when None, ``source`` names a fixture
            file under ``tests/fixtures/trees/versions`` to copy.
        source: Fixture file name (mutually exclusive with ``content``).
        mtime_ns: The mtime to set, so version ordering is deterministic.
    """
    path = env.root_path / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    text = content if content is not None else (_FIXTURES / source).read_text(encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))


def drain(*job_types: str) -> None:
    """Run the worker until no job of the given types is claimable."""
    worker = Worker(concurrency=1, poll_interval_s=0.05, types=list(job_types))
    while worker.run_once():
        pass


def process(env: GraphEnv) -> None:
    """Scan the root, extract/chunk everything, embed the latest versions."""
    scan_root(env.root_id)
    drain("extract", "chunk")
    with session_scope() as session:
        version_ids = [
            int(document.latest_version_id)
            for document in session.execute(
                select(Document).where(
                    Document.source_root_id == env.root_id,
                    Document.latest_version_id.is_not(None),
                )
            ).scalars()
        ]
    for version_id in version_ids:
        pipeline.handle_embed(
            {"version_id": version_id},
            JobContext(worker_id="test", settings=env.settings, embedder=env.fake),
        )
    drain("apply_retention", "version_detect", "link_resolve", "reindex_metadata")


def document_id_of(env: GraphEnv, rel_path: str) -> int:
    """The document id stored for ``rel_path``."""
    with session_scope() as session:
        file_row = session.execute(
            select(File).where(File.source_root_id == env.root_id, File.rel_path == rel_path)
        ).scalar_one()
        assert file_row.document_id is not None
        return int(file_row.document_id)


def install_linked_corpus(env: GraphEnv) -> dict[str, int]:
    """Install the three linked notes, a version pair and a duplicate; process.

    Returns:
        The document ids by short name (``a``, ``b``, ``c``, ``plano``,
        ``idea_v2``, ``dup``).
    """
    install(
        env,
        "notes/a.md",
        content="# Doc A\n\nVejamos [[b]] para continuar.\n",
    )
    install(env, "notes/b.md", content="# Doc B\n\nO plano segue em [plano](c.md).\n")
    install(env, "notes/c.md", content="# Doc C\n\nVoltamos ao início com [[a]].\n")
    install(env, "idea.md", source="idea.md")
    install(env, "idea_v2.md", source="idea_v2.md", mtime_ns=_MTIME + 1_000_000_000)
    process(env)
    install(env, "zz-copia.md", source="idea.md")
    process(env)
    return {
        name: document_id_of(env, rel_path)
        for name, rel_path in (
            ("a", "notes/a.md"),
            ("b", "notes/b.md"),
            ("c", "notes/c.md"),
            ("plano", "idea.md"),
            ("plano_v2", "idea_v2.md"),
            ("dup", "zz-copia.md"),
        )
    }


def test_links_to_relations_neighbours_and_expansion(graph_env: GraphEnv) -> None:
    """The resolved relations exist, neighbours() reads them and the CTE cycles."""
    env = graph_env
    ids = install_linked_corpus(env)
    a, b, c = ids["a"], ids["b"], ids["c"]

    with session_scope() as session:
        relations = (
            session.execute(
                select(DocRelation).where(
                    DocRelation.type == "links_to",
                    DocRelation.src_document_id.in_([a, b, c]),
                )
            )
            .scalars()
            .all()
        )
        by_pair = {
            (r.src_document_id, r.dst_document_id): (r.method, r.status, r.confidence, r.evidence)
            for r in relations
        }
    assert set(by_pair) == {(a, b), (b, c), (c, a)}
    for method, status, confidence, _evidence in by_pair.values():
        assert method == "link_parse"
        assert status == "auto"
        assert confidence == pytest.approx(1.0)
    assert by_pair[(a, b)][3] == {"count": 1, "texts": ["b"]}
    assert by_pair[(b, c)][3] == {"count": 1, "texts": ["plano"]}
    assert by_pair[(c, a)][3] == {"count": 1, "texts": ["a"]}

    with session_scope() as session:
        around_a = neighbours(session, a)
        assert [node.document_id for node in around_a.links_out] == [b]
        assert [node.document_id for node in around_a.links_in] == [c]
        assert sorted(node.document_id for node in around_a.same_folder) == sorted([b, c])
        assert around_a.same_project == []
        assert around_a.tags == ["folder:notes"]
        assert around_a.document.path == "notes/a.md"
        assert around_a.document.status == "current"

        around_plano = neighbours(session, ids["plano"])
        assert ids["dup"] in [node.document_id for node in around_plano.duplicates]
        assert ids["plano_v2"] in [node.document_id for node in around_plano.superseded_by]
        around_v2 = neighbours(session, ids["plano_v2"])
        assert ids["plano"] in [node.document_id for node in around_v2.supersedes]
        assert around_v2.links_out == [] and around_v2.links_in == []

        assert linked_documents(session, [a], hops=2) == {b: 1, c: 1}
        assert linked_documents(session, [a], hops=1) == {b: 1, c: 1}
        assert linked_documents(session, [ids["plano"]], hops=1) == {}
        assert linked_documents(session, [], hops=2) == {}

    pipeline.handle_link_resolve(
        {"document_id": a}, JobContext(worker_id="test", settings=env.settings)
    )
    with session_scope() as session:
        count = len(
            session.execute(
                select(DocRelation).where(
                    DocRelation.src_document_id == a, DocRelation.type == "links_to"
                )
            )
            .scalars()
            .all()
        )
    assert count == 1


def test_cli_graph_doc_prints_json_and_lists_ambiguities(graph_env: GraphEnv) -> None:
    """`graph doc` resolves one match as JSON/sections and exits 1 otherwise."""
    env = graph_env
    ids = install_linked_corpus(env)

    result = runner.invoke(app, ["graph", "doc", "notes/a.md", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["document"]["document_id"] == ids["a"]
    assert [node["path"] for node in data["links_out"]] == ["notes/b.md"]
    assert [node["path"] for node in data["links_in"]] == ["notes/c.md"]

    result = runner.invoke(app, ["graph", "doc", "notes/a.md"])
    assert result.exit_code == 0, result.output
    assert "Links out (1)" in result.output
    assert "Links in (1)" in result.output

    result = runner.invoke(app, ["graph", "doc", "Projeto Farol: ideia inicial"])
    assert result.exit_code == 1
    assert "3 documents match" in result.output

    result = runner.invoke(app, ["graph", "doc", "missing.md"])
    assert result.exit_code == 1
    assert "No document matches" in result.output


def test_removing_link_deletes_relation(graph_env: GraphEnv) -> None:
    """Editing the link out of B's text deletes the stale B→C relation."""
    env = graph_env
    ids = install_linked_corpus(env)
    b = ids["b"]
    with session_scope() as session:
        assert (
            session.execute(
                select(DocRelation.id).where(
                    DocRelation.src_document_id == b,
                    DocRelation.dst_document_id == ids["c"],
                    DocRelation.type == "links_to",
                )
            ).scalar_one_or_none()
            is not None
        )

    install(
        env,
        "notes/b.md",
        content="# Doc B\n\nO link para C desapareceu deste texto.\n",
        mtime_ns=_MTIME + 2_000_000_000,
    )
    process(env)

    with session_scope() as session:
        assert (
            session.execute(
                select(DocRelation.id).where(
                    DocRelation.src_document_id == b, DocRelation.type == "links_to"
                )
            ).scalar_one_or_none()
            is None
        )


def test_sweep_resolves_link_to_later_created_document(graph_env: GraphEnv) -> None:
    """A wikilink to a note created later resolves when link_resolve reruns."""
    env = graph_env
    install(env, "notes/a.md", content="# Doc A\n\nVejamos [[b]] e [[d]].\n")
    install(env, "notes/b.md", content="# Doc B\n\nSem links aqui.\n")
    process(env)
    a, b = document_id_of(env, "notes/a.md"), document_id_of(env, "notes/b.md")

    with session_scope() as session:
        assert documents_with_resolvable_links(session) == [a]
        targets = set(
            session.execute(
                select(DocRelation.dst_document_id).where(
                    DocRelation.src_document_id == a, DocRelation.type == "links_to"
                )
            ).scalars()
        )
    assert targets == {b}

    install(env, "notes/d.md", content="# Doc D\n\nConteúdo do documento D.\n")
    process(env)
    d = document_id_of(env, "notes/d.md")
    with session_scope() as session:
        targets = set(
            session.execute(
                select(DocRelation.dst_document_id).where(
                    DocRelation.src_document_id == a, DocRelation.type == "links_to"
                )
            ).scalars()
        )
    assert targets == {b}, "the sweep has not run yet"

    pipeline.handle_link_resolve(
        {"document_id": a}, JobContext(worker_id="test", settings=env.settings)
    )
    with session_scope() as session:
        targets = set(
            session.execute(
                select(DocRelation.dst_document_id).where(
                    DocRelation.src_document_id == a, DocRelation.type == "links_to"
                )
            ).scalars()
        )
        assert targets == {b, d}
        assert linked_documents(session, [d], hops=2) == {a: 1, b: 2}

    with session_scope() as session:
        assert [node.document_id for node in find_document(session, "notes/d.md")] == [d]
