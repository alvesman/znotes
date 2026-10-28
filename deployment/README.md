# Deploying mynotes

There are three ways to run mynotes. All use the same PostgreSQL 18.6 and OpenSearch 3.8.0 in Docker; they differ in where mynotes itself (the worker, the web UI and API, the CLI and the MCP server) runs and where the notes are.

| Folder | What runs in Docker | mynotes itself | Notes |
|---|---|---|---|
| [`infrastructure/`](infrastructure/README.md) | PostgreSQL and OpenSearch | Natively on your machine, with Python 3.12 and uv | Folders on this machine |
| [`full-stack/`](full-stack/README.md) | PostgreSQL, OpenSearch, the worker and the web UI/API | In a container: the published image `ghcr.io/alvesman/znotes`, or one built from this repository | One folder mounted into the containers |
| [`server/`](server/README.md) | Everything, plus a TLS proxy, for several people | In containers on a server; a small agent on each person's machine | Uploaded by the agents; each person sees their own |

## Trade-offs

| | Infrastructure only | Full stack | Server |
|---|---|---|---|
| **Who uses it** | You, on this machine; no sign-in | You, on this machine; no sign-in | Several people (or you on several machines); each signs in with a personal token and sees only their own notes |
| **Needs on the host** | Docker, Python 3.12, uv, this repository | Docker and the `full-stack` folder (the image is downloaded) | Server: Docker, the `server` folder, and HTTPS (Caddy or your own proxy). Each person: the agent, with uv or Docker |
| **Picking up edits instantly** | Native file events (inotify, FSEvents, Windows): cheap even on big folders | Polling every 2 s: more CPU and disk reads on large folders | The agent reacts within seconds and re-lists the folder every 15 minutes |
| **Open in app** (open a cited file in Word, a PDF viewer…) | Works | Not available (a container can't launch your desktop apps) | Not available (the originals are on the owners' machines); Open and Download serve the server's copy |
| **Access to your notes** | Any folders, anywhere on the machine | One folder, mounted read-only (several roots must share a parent folder, or you add mounts to the compose file) | The folder each agent watches; the server keeps an uploaded copy and never sees the folders |
| **Start at boot** | `mynotes service install` (systemd, launchd, Task Scheduler) | Docker restarts the containers (`restart: unless-stopped`) whenever Docker starts | Server: like the full stack. Agent: `--restart unless-stopped` in a container; a native agent is started by hand |
| **Commands** | `uv run mynotes …` | `docker compose exec worker mynotes …` | `docker compose exec worker mynotes …` on the server; `mynotes agent …` on each machine |
| **Claude Code (MCP)** | `uv run --directory …` | `docker compose run …` | On the server only (it needs direct database access) |
| **Backups** | `mynotes backup` / `mynotes restore` | `pg_dump` / `pg_restore` in the postgres container | `pg_dump` plus the `apphome` volume (the uploaded files) |
| **Updating** | `git pull`, `uv sync`, `mynotes init`, restart the worker | `docker compose pull`, `docker compose up -d` | Bump `MYNOTES_VERSION`, `docker compose pull`, `docker compose up -d`; update the agents to the same version |
| **Exposure** | OpenSearch and PostgreSQL on localhost ports | Only the web UI/API port on localhost; the databases stay inside Docker | The web UI/API over HTTPS; the databases stay inside Docker |

The first two keep everything on your machine and have no login, so the web UI/API is published on `127.0.0.1` only.

**Which one to choose:**
- **Infrastructure only** for your own laptop or desktop. You get every feature: native file watching, open-in-app, and the OS autostart service.
- **Full stack** for an always-on machine or a server, or if you don't want Python on the host. Everything starts and updates with one command, at the cost of open-in-app and native file watching.
- **Server** for a team, or for your own notes on several machines. One server indexes everything and is reachable over HTTPS; each machine only runs the lightweight agent.

## Things to know for all three

- **Separate data.** Each option has its own database and search index. Switching means indexing your notes again (the folder paths differ: host paths vs `/notes/...` in the container, and server mode keeps uploaded copies). A few thousand files take minutes to index; the knowledge graph takes much longer, because it calls the LLM for each document.
- **Memory.** OpenSearch's heap is 2 GB by default (`MYNOTES_COMPOSE_OS_HEAP`). Plan about 4 GB for the databases, plus 1–4 GB for the worker while it indexes. With Docker Desktop (macOS, Windows), give its VM at least 6 GB for the infrastructure option and 8 GB for the full stack.
- **Secrets.** `.env` holds passwords and API keys: keep it private (`chmod 600 .env`) and never commit it. Settings without secrets go in `mynotes.yaml`.
- **Model endpoints.** All three need an embedding model and a chat model, set in `.env` (in server mode, on the server only; the agents need none); a reranker is optional. There are no defaults: the `.env.example` files list ready-made choices for OpenAI and Ollama, and the user manual's [Choosing models](../docs/user-manual.md#choosing-models) section compares them. Check them with `mynotes check-endpoints` before the first start.
