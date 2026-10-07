# Infrastructure only: databases in Docker, mynotes natively

PostgreSQL and OpenSearch run in Docker; mynotes runs on your machine with uv. See [../README.md](../README.md) for how this compares with the full-stack option.

This compose file is the same stack as the repository's own `docker-compose.yml` (same project name, containers and volumes), so both manage the same databases.

| File | Purpose |
|---|---|
| `docker-compose.yml` | PostgreSQL and OpenSearch, published on localhost only |
| `.env.example` | Template for `.env`: passwords, ports, model endpoints and API keys |
| `mynotes.example.yaml` | Template for the settings: notes folders, scan schedule, tagging, retrieval |

## Setup

Run every command from the **repository root**: mynotes reads `.env` from the directory it runs in, and the worker service runs there too.

1. **Install** Docker, Python 3.12 and [uv](https://docs.astral.sh/uv/), then the dependencies:
   ```bash
   uv sync
   ```
2. **Secrets and endpoints.** Copy the template to the repository root and fill in the empty values:
   ```bash
   cp deployment/infrastructure/.env.example .env
   chmod 600 .env
   ```
   Put the PostgreSQL password in both `MYNOTES_COMPOSE_PG_PASSWORD` and `MYNOTES_PG_DSN`. The OpenSearch password must have at least 8 characters, with upper and lower case, a digit and a special character.
3. **Settings.** Copy the settings template to your user config folder and list your notes folders under `roots:`:
   ```bash
   mkdir -p ~/.config/mynotes
   cp deployment/infrastructure/mynotes.example.yaml ~/.config/mynotes/mynotes.yaml
   ```
   macOS: `~/Library/Application Support/mynotes/mynotes.yaml`. Windows: `%LOCALAPPDATA%\mynotes\mynotes\mynotes.yaml`.
4. **Start the databases** and wait until both are healthy:
   ```bash
   docker compose --env-file .env -f deployment/infrastructure/docker-compose.yml up -d --wait
   ```
5. **Prepare and check:**
   ```bash
   uv run mynotes init              # schema, source roots, search indices
   uv run mynotes check-endpoints   # embeddings, reranker and LLM reachable?
   ```
6. **Start the worker.** It scans on start, then on schedule, and processes everything in the background:
   ```bash
   uv run mynotes service install   # start at login (systemd, launchd or Task Scheduler)
   # or, in a terminal you keep open:
   uv run mynotes worker
   ```

## Use

```bash
uv run mynotes ask "What is the deadline for project Alfa?"
uv run mynotes serve               # web UI at http://127.0.0.1:8765/
claude mcp add mynotes -- uv run --directory "$PWD" mynotes mcp
```

The main [README](../../README.md#using-mynotes) covers searching, tags, versions and the web UI.

## Operate

| Task | Command (from the repository root) |
|---|---|
| Status | `uv run mynotes status` |
| Database logs | `docker compose --env-file .env -f deployment/infrastructure/docker-compose.yml logs -f` |
| Stop the databases | `docker compose --env-file .env -f deployment/infrastructure/docker-compose.yml down` |
| Backup | `uv run mynotes backup` (database and settings; never `.env`) |
| Restore | `uv run mynotes restore FILE.dump` (then rebuilds the search indices) |
| Update mynotes | `git pull`, `uv sync`, `uv run mynotes init`, then restart the worker (Linux service: `systemctl --user restart mynotes-worker`; or stop and start `mynotes worker`) |

`down` keeps the data. `down -v` deletes the database and index volumes.
