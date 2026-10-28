# MCP server

mynotes ships an MCP (Model Context Protocol) server so Claude Code, Claude Desktop and other MCP clients can search and read your notes as tools. It is a thin layer over the same core facade as the CLI and the HTTP API: retrieval (hybrid search + rerank), cited answers, documents, version history, tags, the knowledge graph and pipeline status.

## Run it

```bash
uv run mynotes mcp
```

The transport is **stdio**: the MCP client launches `mynotes mcp` itself and talks over stdin/stdout. Logging goes to stderr, so you can watch it in the client's MCP log panel. The client needs access to the same `.env`/`mynotes.yaml` as the CLI (the server resolves the config the same way), and PostgreSQL/OpenSearch must be reachable. The **worker** (`uv run mynotes worker`) must run separately for the index to stay fresh; the server only reads.

## Tools

| Tool | What it does |
|---|---|
| `search_notes(query, tags?, folder?, history?, limit?)` | Hybrid keyword + semantic search, reranked. Returns ranked snippets with `rank`, `title`, `path`, `page`, `slide`, `section`, `snippet` (≤ 500 chars), `doc_date`, `is_current`, `document_id`. |
| `ask_notes(question, tags?, folder?, history?)` | Grounded answer with `[n]` citations and a `relevant` verdict. |
| `get_document(ref)` | Full extracted Markdown (truncated to 20,000 chars with a note when cut) plus metadata. `ref` is a document id, a path relative to a source root, or a title. Ambiguous references return `{"error": "ambiguous", "candidates": [...]}`. |
| `document_history(ref)` | Retained version snapshots, newest first, with changed sections. |
| `list_tags(prefix?)` | Tag vocabulary with accepted-usage counts. |
| `entity_neighbors(name, hops?)` | Knowledge-graph neighbourhood of an entity, scored by evidence. |
| `project_timeline(name)` | Chronological timeline of one `project:NAME` tag's documents. |
| `scan_status()` | Last scan per root, job-queue depth, KG extraction backlog, model-endpoint states. |

All tools run locally (`principal=LOCAL`, no path restrictions) and never raise: failures come back as `{"error": "<kind>", "message": "..."}` so the model can react.

In server mode this means the MCP server, which runs on the server, sees **every person's notes**: only the server's administrator should connect it.

## Resource

`note://{path}` returns the extracted Markdown of one document. The `path` is the document path relative to its source root, **percent-encoded** (resource URI parameters cannot contain slashes): `note://work%2Freports%2Fplan.md` for `work/reports/plan.md`.

## Claude Code

```bash
claude mcp add mynotes -- uv run --directory /path/to/mynotes mynotes mcp
```

Replace `/path/to/mynotes` with the absolute path of this repository. Verify with `claude mcp list`.

With everything in Docker ([deployment/full-stack](../deployment/full-stack/README.md)), run the server in a worker container instead (one per session), from the `deployment/full-stack` folder:

```bash
claude mcp add mynotes -- docker compose -f "$PWD/docker-compose.yml" run --rm -T --no-deps worker mynotes mcp
```

## Claude Desktop

Edit `claude_desktop_config.json` (**Settings → Developer → Edit Config**; on macOS `~/Library/Application Support/Claude/claude_desktop_config.json`, on Windows `%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "mynotes": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mynotes", "mynotes", "mcp"]
    }
  }
}
```

Restart Claude Desktop afterwards; the mynotes tools appear under the tools menu.

## Keeping the index fresh

The MCP server is read-only. Run the scheduler/worker alongside it so scans, embeddings, index sync and KG extraction keep up with file changes:

```bash
uv run mynotes worker          # scheduler + job consumer (live file watching: scan.watch.enabled in mynotes.yaml)
```

`scan_status` tells the model when the queue or the KG backlog is large, i.e. when answers may be stale.
