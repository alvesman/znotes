# Deploying mynotes

There are two ways to run mynotes. Both use the same PostgreSQL 18.6 and OpenSearch 3.8.0 in Docker; they differ in where mynotes itself (the worker, the web UI and API, the CLI and the MCP server) runs.

| Folder | What runs in Docker | mynotes itself |
|---|---|---|
| [`infrastructure/`](infrastructure/README.md) | PostgreSQL and OpenSearch | Natively on your machine, with Python 3.12 and uv |
| [`full-stack/`](full-stack/README.md) | PostgreSQL, OpenSearch, the worker and the web UI/API | In a container: the published image `ghcr.io/alvesman/znotes`, or one built from this repository |

## Trade-offs

| | Infrastructure only | Full stack |
|---|---|---|
| **Needs on the host** | Docker, Python 3.12, uv, this repository | Docker and the `full-stack` folder (the image is downloaded) |
| **Picking up edits instantly** | Native file events (inotify, FSEvents, Windows): cheap even on big folders | Polling every 2 s: more CPU and disk reads on large folders |
| **Open in app** (open a cited file in Word, a PDF viewer…) | Works | Not available (a container can't launch your desktop apps) |
| **Access to your notes** | Any folders, anywhere on the machine | One folder, mounted read-only (several roots must share a parent folder, or you add mounts to the compose file) |
| **Start at boot** | `mynotes service install` (systemd, launchd, Task Scheduler) | Docker restarts the containers (`restart: unless-stopped`) whenever Docker starts |
| **Commands** | `uv run mynotes …` | `docker compose exec worker mynotes …` |
| **Claude Code (MCP)** | `uv run --directory …` | `docker compose run …` |
| **Backups** | `mynotes backup` / `mynotes restore` | `pg_dump` / `pg_restore` in the postgres container |
| **Updating** | `git pull`, `uv sync`, `mynotes init`, restart the worker | `docker compose pull`, `docker compose up -d` |
| **Exposure** | OpenSearch and PostgreSQL on localhost ports | Only the web UI/API port on localhost; the databases stay inside Docker |

Both keep everything on your machine and have no login in local mode, so the web UI/API is published on `127.0.0.1` only.

**Which one to choose:**
- **Infrastructure only** for your own laptop or desktop. You get every feature: native file watching, open-in-app, and the OS autostart service.
- **Full stack** for an always-on machine or a server, or if you don't want Python on the host. Everything starts and updates with one command, at the cost of open-in-app and native file watching.

## Things to know for both

- **Separate data.** Each option has its own database and search index. Switching means indexing your notes again (the folder paths differ: host paths vs `/notes/...` in the container). A few thousand files take minutes to index; the knowledge graph takes much longer, because it calls the LLM for each document.
- **Memory.** OpenSearch's heap is 2 GB by default (`MYNOTES_COMPOSE_OS_HEAP`). Plan about 4 GB for the databases, plus 1–4 GB for the worker while it indexes. With Docker Desktop (macOS, Windows), give its VM at least 6 GB for the infrastructure option and 8 GB for the full stack.
- **Secrets.** `.env` holds passwords and API keys: keep it private (`chmod 600 .env`) and never commit it. Settings without secrets go in `mynotes.yaml`.
- **Model endpoints.** Both need an embedding model and a chat model, set in `.env`; a reranker is optional. There are no defaults: the `.env.example` files list ready-made choices for OpenAI and Ollama, and the user manual's [Choosing models](../docs/user-manual.md#choosing-models) section compares them. Check them with `mynotes check-endpoints` before the first start.
