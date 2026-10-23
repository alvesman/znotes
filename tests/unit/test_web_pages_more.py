"""Unit tests for the remaining web UI pages (task 48).

The facade is monkeypatched with canned results, so the tests never touch
Postgres, OpenSearch or a model endpoint. They pin the document view (heading
ids, escaping, chunk highlighting, tag review, version chain), the review
queues, the graph explorer shell and the status dashboard.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from mynotes.api.app import create_app
from mynotes.core import service
from mynotes.core.errors import Ambiguous, NotFound
from mynotes.tags.autotag import DocumentProposals, ProposalView
from mynotes.tags.store import TagDetail
from mynotes.versioning.review import DocRef, RelationView
from mynotes.web.routes import STATIC_DIR


def _view(**overrides: object) -> service.DocumentView:
    values: dict[str, object] = {
        "document_id": 5,
        "title": "Plano Alfa",
        "path": "work/plano-alfa.md",
        "format": "md",
        "status": "current",
        "doc_date": datetime(2026, 3, 1, tzinfo=UTC),
        "language": "pt",
        "tags": ["project:alfa"],
        "latest_version_no": 3,
        "markdown": "# Plano\n\n## Metas\n\nTexto das metas.\n",
        "sections": [],
    }
    values.update(overrides)
    return service.DocumentView(**values)  # type: ignore[arg-type]


def _client() -> TestClient:
    return TestClient(create_app())


def _patch_document(monkeypatch: pytest.MonkeyPatch, view: service.DocumentView) -> None:
    monkeypatch.setattr(service, "get_document", lambda *args, **kwargs: view)
    monkeypatch.setattr(service, "document_tags", lambda *args, **kwargs: [])
    monkeypatch.setattr(service, "version_chain", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(service, "document_history", lambda *args, **kwargs: [])
    monkeypatch.setattr(service, "document_chunk_text", lambda *args, **kwargs: None)


def test_document_page_renders_heading_ids_and_escapes_html(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    markdown = "# Configuração\n\n<script>alert(1)</script>\n\n## Metas & Co\n\nTexto.\n"
    _patch_document(monkeypatch, _view(markdown=markdown, format="md"))
    response = _client().get("/ui/documents/5")
    assert response.status_code == 200, response.text
    assert '<h1 id="sec-configuracao">Configuração</h1>' in response.text
    assert '<h2 id="sec-metas--co">Metas &amp; Co</h2>' in response.text
    assert "<script>alert(1)</script>" not in response.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text


def test_document_page_wraps_fenced_code_for_copy_buttons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    markdown = "# Notas\n\n```bash\necho <ok> && ls\n```\n\n    indentado\n"
    _patch_document(monkeypatch, _view(markdown=markdown, format="md"))
    response = _client().get("/ui/documents/5")
    assert response.status_code == 200, response.text
    assert (
        '<div class="code-block" data-copy-code><pre><code class="language-bash">'
        "echo &lt;ok&gt; &amp;&amp; ls\n</code></pre>\n</div>"
    ) in response.text
    assert "<pre><code>indentado\n</code></pre>" in response.text  # indented: no button
    assert response.text.count("data-copy-code") == 1
    assert "markdown.js" in response.text


def test_document_page_shows_accept_reject_only_for_proposed_llm_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tags = [
        TagDetail(
            name="project:alfa",
            kind="project",
            source="frontmatter",
            status="accepted",
            confidence=1.0,
        ),
        TagDetail(name="topic:kg", kind="topic", source="llm", status="proposed", confidence=0.7),
        TagDetail(name="topic:old", kind="topic", source="llm", status="rejected", confidence=0.9),
        TagDetail(
            name="folder:work", kind="folder", source="folder", status="accepted", confidence=1.0
        ),
    ]
    _patch_document(monkeypatch, _view())
    monkeypatch.setattr(service, "document_tags", lambda *args, **kwargs: tags)
    response = _client().get("/ui/documents/5")
    assert response.status_code == 200, response.text
    assert response.text.count(">Accept</button>") == 1
    assert response.text.count(">Reject</button>") == 1
    assert "/v1/tags/5/topic%3Akg/accept" in response.text
    assert "topic%3Aold" not in response.text
    assert "project%3Aalfa/accept" not in response.text


def test_document_page_highlights_chunk_paragraph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_document(
        monkeypatch,
        _view(markdown="## Metas\n\nPrimeiro parágrafo do chunk.\n\nOutro parágrafo.\n"),
    )
    monkeypatch.setattr(
        service,
        "document_chunk_text",
        lambda *args, **kwargs: "## Metas\n\nPrimeiro parágrafo do chunk.",
    )
    response = _client().get("/ui/documents/5?chunk=3")
    assert response.status_code == 200, response.text
    assert "<mark>Primeiro parágrafo do chunk.</mark></p>" in response.text
    assert "<p>Outro parágrafo.</p>" in response.text


def test_document_page_without_chunk_highlight_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_document(monkeypatch, _view(markdown="## Metas\n\nPrimeiro parágrafo.\n"))
    called: list[object] = []

    def fail_chunk(*args: object, **kwargs: object) -> str | None:
        called.append(args)
        return None

    monkeypatch.setattr(service, "document_chunk_text", fail_chunk)
    response = _client().get("/ui/documents/5")
    assert response.status_code == 200
    assert "<mark>" not in response.text
    assert called == []


def test_document_page_links_version_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    newer = service.RelatedDocument(
        document_id=9,
        title="Plano Alfa v2",
        path="work/plano-alfa-v2.md",
        doc_date=datetime(2026, 9, 1, tzinfo=UTC),
        status="current",
    )
    older = service.RelatedDocument(
        document_id=3,
        title="Plano Alfa 2025",
        path="archive/plano-alfa-2025.md",
        doc_date=None,
        status="superseded",
    )
    _patch_document(monkeypatch, _view())
    monkeypatch.setattr(service, "version_chain", lambda *args, **kwargs: ([newer], [older]))
    response = _client().get("/ui/documents/5")
    assert response.status_code == 200, response.text
    assert 'href="/ui/documents/9"' in response.text
    assert 'href="/ui/documents/3"' in response.text
    assert "Newer:" in response.text and "Older:" in response.text


def test_document_page_shows_history_and_office_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = service.VersionInfo(
        version_no=2,
        created_at=datetime(2026, 9, 10, tzinfo=UTC),
        doc_date=datetime(2026, 9, 1, tzinfo=UTC),
        rel_path="work/plano-alfa.md",
        status="embedded",
        chunk_count=7,
        is_latest=True,
        changed_sections=["Metas", "Prazos"],
    )
    _patch_document(monkeypatch, _view(format="docx"))
    monkeypatch.setattr(service, "document_history", lambda *args, **kwargs: [history])
    response = _client().get("/ui/documents/5")
    assert response.status_code == 200, response.text
    assert "v2" in response.text
    assert "changed: Metas, Prazos" in response.text
    assert 'hx-post="/v1/files/5/open"' in response.text
    assert "Open in app" in response.text
    assert 'href="/v1/files/5/raw"' in response.text


def test_unknown_document_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_not_found(*args: object, **kwargs: object) -> service.DocumentView:
        raise NotFound("no document with id 999")

    monkeypatch.setattr(service, "get_document", raise_not_found)
    response = _client().get("/ui/documents/999")
    assert response.status_code == 404


def test_reviews_page_lists_proposed_relation_with_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relation = RelationView(
        id=12,
        status="proposed",
        confidence=0.92,
        method="name_content",
        newer=DocRef(
            document_id=9,
            title="Plano v2",
            path="work/plano-v2.md",
            doc_date=datetime(2026, 9, 1, tzinfo=UTC),
            status="current",
        ),
        older=DocRef(
            document_id=5,
            title="Plano",
            path="work/plano.md",
            doc_date=datetime(2026, 3, 1, tzinfo=UTC),
            status="current",
        ),
        evidence={"jaccard": 0.82, "cosine": 0.91, "order_reason": "date and version marker"},
    )
    monkeypatch.setattr(service, "list_version_relations", lambda *args, **kwargs: [relation])
    monkeypatch.setattr(
        service,
        "document_previews",
        lambda ids, **kwargs: {9: "Novo plano" * 40, 5: "Plano antigo" * 40},
    )
    response = _client().get("/ui/reviews")
    assert response.status_code == 200, response.text
    assert 'href="/ui/documents/9"' in response.text
    assert 'href="/ui/documents/5"' in response.text
    assert "Plano v2" in response.text and "Plano antigo" in response.text
    assert "jaccard: 0.82" in response.text
    assert "cosine: 0.91" in response.text
    assert "order_reason: date and version marker" in response.text
    assert 'hx-post="/v1/versions/12/confirm"' in response.text
    assert 'hx-post="/v1/versions/12/reject"' in response.text
    assert 'hx-post="/v1/versions/12/swap"' in response.text


def test_reviews_tags_tab_lists_proposals_with_merge_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposals = [
        DocumentProposals(
            document_id=5,
            title="Plano",
            rel_path="work/plano.md",
            proposals=[
                ProposalView(
                    document_id=5,
                    tag_id=3,
                    name="topic:grafos",
                    confidence=0.7,
                    reason="mentions graphs",
                ),
            ],
        )
    ]
    monkeypatch.setattr(service, "tag_proposals", lambda *args, **kwargs: proposals)
    response = _client().get("/ui/partials/reviews/tags")
    assert response.status_code == 200, response.text
    assert "/v1/tags/5/topic%3Agrafos/accept" in response.text
    assert "/v1/tags/5/topic%3Agrafos/reject" in response.text
    assert 'hx-post="/v1/tags/merge"' in response.text
    assert 'hx-ext="json-enc"' in response.text
    assert 'name="from" value="topic:grafos"' in response.text


def test_reviews_entities_tab_lists_candidates_with_keep_older(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = service.EntityPair(
        a_id=8,
        b_id=3,
        a_name="Projeto Alfa",
        b_name="Proj Alfa",
        a_type="Project",
        b_type="Project",
        a_created_at=datetime(2026, 9, 24, tzinfo=UTC),
        b_created_at=datetime(2026, 9, 20, tzinfo=UTC),
        similarity=0.72,
    )
    monkeypatch.setattr(service, "entity_merge_candidates", lambda *args, **kwargs: [pair])
    response = _client().get("/ui/partials/reviews/entities")
    assert response.status_code == 200, response.text
    assert "Projeto Alfa" in response.text and "Proj Alfa" in response.text
    assert "name similarity 0.72" in response.text
    assert 'hx-post="/ui/reviews/entities/3/8/merge"' in response.text
    assert 'hx-post="/ui/reviews/entities/8/3/distinct"' in response.text


def test_entity_merge_action_calls_facade_and_rerenders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_merge(keep_id: int, drop_id: int, *, principal: object = None) -> None:
        seen["keep"] = keep_id
        seen["drop"] = drop_id

    monkeypatch.setattr(service, "merge_entities", fake_merge)
    monkeypatch.setattr(service, "entity_merge_candidates", lambda *args, **kwargs: [])
    response = _client().post("/ui/reviews/entities/3/8/merge")
    assert response.status_code == 200, response.text
    assert seen == {"keep": 3, "drop": 8}
    assert 'id="review-entities"' in response.text
    assert "No entity merge candidates." in response.text


def test_entity_distinct_action_calls_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[int, int]] = []

    def fake_distinct(a_id: int, b_id: int, *, principal: object = None) -> None:
        seen.append((a_id, b_id))

    monkeypatch.setattr(service, "mark_entities_distinct", fake_distinct)
    monkeypatch.setattr(service, "entity_merge_candidates", lambda *args, **kwargs: [])
    response = _client().post("/ui/reviews/entities/8/3/distinct")
    assert response.status_code == 200, response.text
    assert seen == [(8, 3)]


def test_graph_page_includes_cytoscape_and_graph_js() -> None:
    response = _client().get("/ui/graph")
    assert response.status_code == 200, response.text
    assert '<script src="/static/graph.js?v=' in response.text
    assert '<script src="/static/cytoscape-' in response.text
    cytoscape_files = sorted(STATIC_DIR.glob("cytoscape-*.min.js"))
    assert len(cytoscape_files) == 1
    assert cytoscape_files[0].read_bytes().startswith(b"/*")


def test_graph_page_with_entity_shows_card_attributes_timeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = {
        "id": 4,
        "name": "Projeto Alfa",
        "type": "Project",
        "status": "active",
        "description": "Projeto piloto de busca.",
        "aliases": ["Proj Alfa"],
        "top_documents": [
            {"document_id": 5, "title": "Plano", "rel_path": "work/plano.md", "mentions": 3}
        ],
    }
    attributes = [
        service.AttributeValue(
            attribute="status",
            value="on track",
            doc_date=datetime(2026, 9, 1, tzinfo=UTC),
            relation_id=11,
        )
    ]
    timeline = [
        service.TimelineItem(
            date=datetime(2026, 9, 1, tzinfo=UTC),
            kind="relation",
            text="Projeto Alfa –uses→ OpenSearch",  # noqa: RUF001
            status="active",
            document_path="work/plano.md",
        )
    ]
    monkeypatch.setattr(service, "entity_card", lambda *args, **kwargs: card)
    monkeypatch.setattr(service, "entity_attributes", lambda *args, **kwargs: attributes)
    monkeypatch.setattr(service, "timeline", lambda *args, **kwargs: timeline)
    response = _client().get("/ui/graph?entity=Projeto%20Alfa")
    assert response.status_code == 200, response.text
    assert "Projeto Alfa" in response.text
    assert "Proj Alfa" in response.text
    assert "status" in response.text and "on track" in response.text
    assert "2026-09-01" in response.text
    assert "Projeto Alfa –uses→ OpenSearch" in response.text  # noqa: RUF001
    assert 'data-entity-id="4"' in response.text
    assert 'id="graph-hops"' in response.text
    assert 'id="graph-expand"' in response.text


def test_graph_entity_partial_renders_candidates_when_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_ambiguous(*args: object, **kwargs: object) -> dict[str, object]:
        raise Ambiguous(
            "2 entities match 'alfa'",
            [
                {"id": 4, "name": "Projeto Alfa", "type": "Project"},
                {"id": 9, "name": "Projeto Alfa II", "type": "Project"},
            ],
        )

    monkeypatch.setattr(service, "entity_card", raise_ambiguous)
    response = _client().get("/ui/partials/entity-panel", params={"q": "alfa"})
    assert response.status_code == 200, response.text
    assert "2 entities match" in response.text
    # candidates link by id: two candidates can share a name (a Project and a Concept)
    assert 'href="/ui/graph?entity=%234">Projeto Alfa</a>' in response.text
    assert 'href="/ui/graph?entity=%239">Projeto Alfa II</a>' in response.text
    assert_panel_fragment_only(response.text)


def assert_panel_fragment_only(html: str) -> None:
    """The panel partial must not repeat the page shell it is swapped into."""
    assert "<html" not in html
    assert 'class="topbar"' not in html
    assert 'id="entity-panel"' not in html
    assert 'id="entity-q"' not in html


def test_graph_entity_partial_renders_card_without_page_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = {
        "id": 4,
        "name": "Projeto Alfa",
        "type": "Project",
        "status": "active",
        "description": None,
        "aliases": [],
        "top_documents": [],
    }
    monkeypatch.setattr(service, "entity_card", lambda *args, **kwargs: card)
    monkeypatch.setattr(service, "entity_attributes", lambda *args, **kwargs: [])
    monkeypatch.setattr(service, "timeline", lambda *args, **kwargs: [])
    response = _client().get("/ui/partials/entity-panel", params={"q": "Projeto Alfa"})
    assert response.status_code == 200, response.text
    assert "Projeto Alfa" in response.text
    assert 'data-entity-id="4"' in response.text
    assert 'id="graph-hops"' in response.text
    assert_panel_fragment_only(response.text)


def test_graph_entity_partial_empty_query_shows_hint() -> None:
    response = _client().get("/ui/partials/entity-panel", params={"q": "  "})
    assert response.status_code == 200, response.text
    assert "Search for an entity" in response.text
    assert_panel_fragment_only(response.text)


def test_status_page_shows_inspect_live_problem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = {
        "opensearch": {
            "aliases": {"mynotes_chunks": {"physical": "mynotes_chunks_v2", "docs": 2873}},
            "problems": ["mapping drift on chunk.text"],
        }
    }
    monkeypatch.setattr(service, "status", lambda *args, **kwargs: status)
    response = _client().get("/ui/status")
    assert response.status_code == 200, response.text
    assert "mapping drift on chunk.text" in response.text
    assert 'class="problem"' in response.text
    assert "mynotes_chunks_v2 (2873 docs)" in response.text


def test_status_page_renders_roots_queue_backlog_and_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = {
        "roots": [{"name": "ideas", "path": "/home/me/ideas", "source_root_id": 1}],
        "watchers": [
            {
                "root": "ideas",
                "state": "watching",
                "backend": "native",
                "since": "2026-09-25T08:00:00+00:00",
                "stale": False,
            }
        ],
        "last_scans": [
            {
                "root": "ideas",
                "kind": "incremental",
                "status": "ok",
                "started_at": datetime(2026, 9, 25, 8, 0, tzinfo=UTC),
                "finished_at": datetime(2026, 9, 25, 8, 1, tzinfo=UTC),
                "counts": {"new": 2, "unchanged": 566},
                "error": None,
            }
        ],
        "postgres": {
            "jobs": {"queued": 3, "running": 1, "failed": 0},
            "outbox_unprocessed": 5,
            "files_by_status": {"active": 568, "placeholder": 2},
        },
        "queue_by_type": {"chunk": 2, "kg_extract": 1},
        "kg_backlog": {
            "queued_kg_extract_jobs": 1,
            "groups_total": 100,
            "groups_cached": 40,
            "groups_pending": 60,
            "eta_seconds": 900,
        },
        "drift": {
            "saved_at": "2026-09-25T06:00:00+00:00",
            "expected": 2873,
            "indexed": 2870,
            "missing": 3,
            "extra": 0,
            "stale": 0,
        },
        "endpoints": [
            {
                "name": "llm",
                "state": "closed",
                "calls": 12,
                "p50_ms": 900.0,
                "p95_ms": 1500.0,
                "error_rate": 0.0,
                "last_error": None,
            }
        ],
        "unsupported": {"count": 1, "examples": ["ideas/notas.zip"]},
    }
    monkeypatch.setattr(service, "status", lambda *args, **kwargs: status)
    response = _client().get("/ui/status")
    assert response.status_code == 200, response.text
    assert "ideas" in response.text and "/home/me/ideas" in response.text
    assert "watching (native)" in response.text
    assert "incremental/ok" in response.text
    assert "new=2 unchanged=566" in response.text
    assert "queued 3 · running 1 · failed 0" in response.text
    assert "chunk=2, kg_extract=1" in response.text
    assert "60 pending group(s)" in response.text
    assert "ETA 15 min" in response.text
    assert "expected/indexed 2873/2870" in response.text
    assert "missing 3 · extra 0 · stale 0" in response.text
    assert "llm" in response.text and "900" in response.text and "0.0%" in response.text
    assert "ideas/notas.zip" in response.text
    assert "2 cloud placeholder file(s)" in response.text
    assert 'hx-post="/v1/scan"' in response.text


def test_status_panel_partial_auto_refreshes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "status", lambda *args, **kwargs: {})
    response = _client().get("/ui/partials/status")
    assert response.status_code == 200, response.text
    assert 'hx-trigger="every 30s, mynotes:status-changed from:body"' in response.text
    assert response.headers["content-security-policy"].startswith("default-src 'self'")


def test_nav_links_added_to_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "list_tags", lambda *args, **kwargs: [])
    response = _client().get("/ui/search")
    for href in ("/ui/reviews", "/ui/graph", "/ui/status"):
        assert f'href="{href}"' in response.text


def test_static_assets_served() -> None:
    client = _client()
    for name in ("graph.js", "json-enc-2.0.3.js", "cytoscape-3.34.3.min.js"):
        response = client.get(f"/static/{name}")
        assert response.status_code == 200, name
        assert "javascript" in response.headers["content-type"]
    assert (STATIC_DIR / "cytoscape-3.34.3.min.js").stat().st_size > 100_000


def test_favicon_is_served_and_linked() -> None:
    client = _client()
    for url in ("/favicon.ico", "/static/favicon.ico"):
        response = client.get(url)
        assert response.status_code == 200, url
        assert response.content[:4] == b"\x00\x00\x01\x00", url  # ICO header
    assert response.headers["content-type"] in ("image/x-icon", "image/vnd.microsoft.icon")
    assert '<link rel="icon" href="/static/favicon.ico"' in client.get("/ui/status").text


def test_help_page_defaults_to_setup_and_command_line() -> None:
    response = _client().get("/ui/help")
    assert response.status_code == 200, response.text
    assert "Help · Setup and command line · mynotes" in response.text
    tabs = response.text.split('class="tabs help-tabs"', 1)[1].split("</nav>", 1)[0]
    assert (
        tabs.index("Setup and command line")
        < tabs.index("Web interface")
        < tabs.index("Claude and MCP")
    )
    assert 'href="/ui/help/cli" aria-current="page"' in tabs


def test_help_web_manual_renders_with_working_links() -> None:
    client = _client()
    response = client.get("/ui/help/web")
    assert response.status_code == 200, response.text
    html = response.text
    assert "Content-Security-Policy" in response.headers
    assert 'href="/ui/help" aria-current="page"' in html  # nav entry
    assert 'href="/ui/help/cli"' in html and 'href="/ui/help/mcp"' in html  # tabs
    # Headings get the document-view anchors and the manual's TOC points at them.
    assert 'id="sec-4-chat-asking-questions"' in html
    assert 'href="#sec-4-chat-asking-questions"' in html
    assert 'href="#4-chat-asking-questions"' not in html
    # The link to the other manual becomes its Help tab.
    assert 'href="/ui/help/cli"' in html and "user-manual.md" not in html
    assert "<script" not in html.split("<main", 1)[1]  # no inline markup from the Markdown


def test_help_tabs_and_unknown_topic() -> None:
    client = _client()
    cli = client.get("/ui/help/cli")
    assert cli.status_code == 200 and "uv run mynotes ask" in cli.text
    assert 'href="/ui/help/web' in cli.text  # link to the web manual
    mcp = client.get("/ui/help/mcp")
    assert mcp.status_code == 200 and "claude mcp add" in mcp.text
    assert client.get("/ui/help/nope").status_code == 404


def test_help_link_rewriting_rules() -> None:
    from mynotes.web.render import render_help_markdown

    links = {"user-manual.md": "/ui/help/cli"}
    md = (
        "[a](#2-opening-the-web-interface) [b](user-manual.md#7-filters) "
        "[c](../deployment/README.md) [d](https://example.org/x) [e](mcp.md)"
    )
    html = render_help_markdown(md, doc_links=links, source_base_url="https://repo/blob/main")
    assert 'href="#sec-2-opening-the-web-interface"' in html
    assert 'href="/ui/help/cli#sec-7-filters"' in html
    assert 'href="https://repo/blob/main/deployment/README.md"' in html
    assert 'href="https://example.org/x"' in html
    assert 'href="https://repo/blob/main/docs/mcp.md"' in html


def test_theme_script_runs_before_the_stylesheet() -> None:
    """theme.js sets data-theme before the first paint: in <head>, before app.css, not deferred."""
    client = _client()
    served = client.get("/static/theme.js")
    assert served.status_code == 200
    assert "javascript" in served.headers["content-type"]
    assert "mynotes.pref.theme" in served.text

    html = client.get("/ui/chat").text
    head = html.split("</head>", 1)[0]
    script_at = head.index('<script src="/static/theme.js?v=')
    assert script_at < head.index('href="/static/app.css?v=')
    tag = head[script_at : head.index(">", script_at)]
    assert "defer" not in tag and "async" not in tag


def test_dark_theme_blocks_declare_the_same_tokens() -> None:
    """The OS-preference block and the explicit-choice block of app.css must not drift apart."""
    import re

    css = (STATIC_DIR / "app.css").read_text(encoding="utf-8")

    def tokens(block_start: str) -> dict[str, str]:
        start = css.index(block_start)
        body = css[start : css.index("}", start)]
        return dict(re.findall(r"(--[\w-]+):\s*([^;]+);", body))

    system_dark = tokens(':root:not([data-theme="light"]) {')
    chosen_dark = tokens(':root[data-theme="dark"] {\n  --bg')
    assert system_dark, "no dark tokens found"
    assert system_dark == chosen_dark
    light = tokens(":root {\n  color-scheme: light dark;")
    assert set(system_dark) <= set(light), "every dark token needs a light default"


def test_theme_switch_markup_on_every_page() -> None:
    """A three-way System/Light/Dark control in the top bar, wired by ui.js."""
    client = _client()
    served = client.get("/static/ui.js")
    assert served.status_code == 200 and "javascript" in served.headers["content-type"]
    for path in ("/ui/chat", "/ui/help"):
        html = client.get(path).text
        assert 'src="/static/ui.js?v=' in html.split("</head>", 1)[0]
        assert 'role="group" aria-label="Colour theme" data-theme-switch' in html
        for choice in ("system", "light", "dark"):
            assert f'<button type="button" data-theme-choice="{choice}" aria-pressed=' in html
        assert html.count("data-theme-choice=") == 3
        assert "data-persist" not in html.split("data-theme-switch", 1)[1].split("</nav>", 1)[0]


def test_shell_markup_skip_link_menu_and_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mynotes.core.chat_history.list_conversations", lambda principal: ([], 0))
    response = _client().get("/ui/history")
    if response.status_code != 200:
        pytest.skip("history page needs a live backend here")
    html = response.text
    assert '<a class="skip-link" href="#main">' in html
    assert '<nav class="topbar" aria-label="Main">' in html
    assert '<a class="brand" href="/ui/chat">mynotes</a>' in html
    assert (
        '<button type="button" class="nav-toggle" aria-expanded="false" aria-controls="nav-links">'
        in html
    )
    assert '<ul id="nav-links" class="nav-links">' in html
    assert '<main id="main" class="content" tabindex="-1">' in html


@pytest.mark.parametrize("path", ["/ui/chat", "/ui/search", "/ui/status", "/ui/help/web"])
def test_pages_have_no_inline_style_or_script(path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CSP (style-src/script-src 'self') would block them, so none may exist."""
    import re

    monkeypatch.setattr(service, "search", lambda *args, **kwargs: None)
    monkeypatch.setattr(service, "list_tags", lambda *args, **kwargs: [])
    monkeypatch.setattr(service, "status", lambda *args, **kwargs: {"roots": [], "queue": {}})
    response = _client().get(path)
    if response.status_code != 200:
        pytest.skip(f"{path} needs a live backend here ({response.status_code})")
    html = response.text
    assert "<style" not in html
    assert ' style="' not in html
    for tag in re.findall(r"<script[^>]*>", html):
        assert 'src="/static/' in tag, tag


def test_chat_and_search_have_labels_and_collapsible_panels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()
    chat = client.get("/ui/chat").text
    assert '<label for="question" class="visually-hidden">' in chat
    assert '<details class="panel options" open data-collapse-narrow>' in chat
    assert (
        chat.index('class="chat-options"')
        < chat.index('id="conversation"')
        < chat.index('id="question"')
    )
    assert "aria-live" not in chat  # the streaming answer is not a live region (chat.js announces)
    monkeypatch.setattr(service, "list_tags", lambda *args, **kwargs: [])
    search = client.get("/ui/search").text
    assert '<main id="main" class="content content--wide" tabindex="-1">' in search
    assert '<label for="search-query" class="visually-hidden">' in search
    assert '<details class="panel filters" open data-collapse-narrow>' in search
    assert (
        search.index('class="search-bar"')
        < search.index('class="search-filters"')
        < search.index('id="results"')
    )


def test_status_panel_id_appears_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """status.html used to wrap the partial in a second #status-panel (the panel polled twice)."""
    monkeypatch.setattr(service, "status", lambda *args, **kwargs: {"roots": [], "queue": {}})
    html = _client().get("/ui/status").text
    assert html.count('id="status-panel"') == 1
    assert html.count('class="table-scroll"') >= 1


def test_reviews_tabs_have_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "list_version_relations", lambda *args, **kwargs: [])
    monkeypatch.setattr(service, "document_previews", lambda ids, **kwargs: {})
    response = _client().get("/ui/reviews")
    assert response.status_code == 200, response.text
    html = response.text
    assert html.count('role="tab" aria-selected=') == 3
    assert html.count('role="tab" aria-selected="true"') == 1
    assert '<div id="review-panel" role="tabpanel">' in html


def test_css_uses_tokens_outside_the_token_section() -> None:
    """Every colour lives in the token blocks: no hex or rgb() in the other rules."""
    import re

    css = (STATIC_DIR / "app.css").read_text(encoding="utf-8")
    rules = css[css.index("/* 2. Reset and base */") :]
    colours = re.findall(r":\s*[^;{}]*?(#[0-9a-fA-F]{3,8}\b|rgba?\()", rules)
    assert colours == []
