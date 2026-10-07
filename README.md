# mynotes

mynotes answers questions about a folder of notes. Point it at your Markdown notes (and PDF, Word, PowerPoint, Excel, HTML and EPUB files) and ask in plain language: it finds the relevant passages with hybrid (vector + keyword) search and writes an answer that cites them. It keeps working as your notes change: it re-reads only what changed, tracks versions so answers use the current one while history stays searchable, and builds a knowledge graph of the people, projects and tools your notes mention. A web interface (chat with history, search, document view, reviews, graph explorer), a command line and an MCP server for Claude are included.

Everything runs on your machine: PostgreSQL is the source of truth and OpenSearch the search index, both in Docker. The AI models are yours to choose: any OpenAI-compatible server for embeddings and chat (OpenAI, Ollama on your own computer, vLLM, …), and optionally a reranker.

## Quick start (Docker)

You need Docker with Docker Compose, about 6 GB of free memory, and access to an embedding model and a chat model.

1. Get the deployment files (this repository's `deployment/full-stack` folder):
   ```bash
   git clone https://github.com/alvesman/znotes.git
   cd znotes/deployment/full-stack
   cp .env.example .env && chmod 600 .env
   cp mynotes.example.yaml mynotes.yaml
   ```
2. Edit `.env`: your notes folder, two passwords you invent, and the model endpoints. There are no defaults; the file lists ready-made choices for OpenAI and Ollama.
3. Check the models, then start:
   ```bash
   docker compose run --rm init mynotes check-endpoints
   docker compose up -d --wait
   ```
4. Open http://127.0.0.1:8765/ui/chat. The worker indexes your notes in the background; **Status** shows its progress.

To update later: `docker compose pull && docker compose up -d`. Details, backups and the trade-offs against a native install are in [deployment/](deployment/README.md).

## Documentation

- **[User manual](docs/user-manual.md)**: installation (native or Docker), choosing models, asking and searching, filters, the web interface, Claude (MCP), keeping the index up to date, tags, versions, the knowledge graph, configuration, backups, troubleshooting and the full command reference.
- **[Web interface manual](docs/web-interface-manual.md)**: each page of the browser interface.
- **[MCP guide](docs/mcp.md)**: using mynotes from Claude Code and Claude Desktop.

## Development

Requires Python 3.12 with [uv](https://docs.astral.sh/uv/) and Docker for PostgreSQL and OpenSearch.

```bash
make sync      # uv sync (create/refresh the virtualenv)
make fmt       # uv run ruff format .
make lint      # uv run ruff check .
make test      # uv run pytest (unit tests)
make test-int  # uv run pytest -m integration (needs Docker)
make up        # docker compose up -d (the two databases)
make down      # docker compose down
```

The test groups:

```bash
uv run ruff format --check . && uv run ruff check .   # lint
uv run pytest -q                                      # unit tests
uv run pytest -q -m watcher                           # watcher tests (real file system, no Docker)
uv run pytest -q -m "integration and not watcher"     # PostgreSQL/OpenSearch tests (needs Docker)
```

## Licence

mynotes is free software: you can redistribute it and/or modify it under the terms of the [GNU Affero General Public License](LICENSE), version 3 or (at your option) any later version. If you run a modified version for others over a network, you must offer them its source code; the web interface's footer links to the source for that reason. Third-party components and their licences are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
