# Full stack: everything in Docker

PostgreSQL, OpenSearch, the mynotes worker and the web UI/API all run in Docker. See [../README.md](../README.md) for how this compares with the other two options.

| File | Purpose |
|---|---|
| `docker-compose.yml` | The services: `postgres`, `opensearch`, `init` (one-shot setup), `worker` and `web`, using the published image `ghcr.io/alvesman/znotes` |
| `docker-compose.build.yml` | Optional override that builds the image from this repository instead |
| `Dockerfile` | The mynotes image (Python 3.12, runs as an unprivileged user) |
| `Dockerfile.dockerignore` | Sends only the source to the image build, never `.env` |
| `.env.example` | Template for `.env`: notes folder, passwords, web port, model endpoints and API keys |
| `mynotes.example.yaml` | Template for `mynotes.yaml`: notes folders inside the container, scan schedule, tagging, retrieval |

## Setup

Run every command from **this folder** (`deployment/full-stack`).

1. **Secrets, endpoints and your notes folder:**
   ```bash
   cp .env.example .env
   chmod 600 .env
   ```
   Fill in the empty values:
   - `MYNOTES_NOTES_DIR`: absolute path of your notes folder on the host. It is mounted read-only at `/notes`.
   - `MYNOTES_COMPOSE_PG_PASSWORD`: letters and digits only.
   - `MYNOTES_OPENSEARCH_PASSWORD`: at least 8 characters, with upper and lower case, a digit and a special character.
   - The model endpoints: `MYNOTES_EMBEDDING_*` and `MYNOTES_LLM_*` (required), `MYNOTES_RERANKER_*` (optional). There are no defaults; the file lists choices for OpenAI and Ollama. A model server running on this machine (such as Ollama) is reached from the containers at `http://host.docker.internal:<port>/v1`.
2. **Settings:**
   ```bash
   cp mynotes.example.yaml mynotes.yaml
   ```
   The default root is the whole notes folder (`/notes`). To index only some subfolders, list them as separate roots (`/notes/work`, `/notes/ideas`, …).
3. **Check the model endpoints** (downloads the image the first time):
   ```bash
   docker compose run --rm init mynotes check-endpoints
   ```
   Every line should say `ok`; the reranker says `warn` when you left it out on purpose.
4. **Start:**
   ```bash
   docker compose up -d --wait
   ```
   `init` prepares the database and search indices and exits (it stops with a clear message if a model setting is missing); then `worker` and `web` start. The worker scans your notes right away. To use a specific release, set `MYNOTES_VERSION` in `.env` (for example `0.2.0`); the default is the latest. To build the image from this repository instead, add `-f docker-compose.yml -f docker-compose.build.yml` to the compose commands and `--build` to `up`.
5. **Open http://127.0.0.1:8765/** (or the port set in `MYNOTES_WEB_PORT`).

## Use

The CLI runs inside the worker container:

```bash
docker compose exec worker mynotes ask "What is the deadline for project Alfa?"
docker compose exec worker mynotes search "kubernetes ingress" -n 5
docker compose exec worker mynotes tags review      # interactive
```

Claude Code (MCP server, one container per session):

```bash
claude mcp add mynotes -- docker compose -f "$PWD/docker-compose.yml" run --rm -T --no-deps worker mynotes mcp
```

The [user manual](../../docs/user-manual.md) covers asking, searching, tags, versions and the knowledge graph; the [web interface manual](../../docs/web-interface-manual.md) covers the web UI.

## Operate

| Task | Command (from this folder) |
|---|---|
| Status | `docker compose ps` and `docker compose exec worker mynotes status` |
| Logs | `docker compose logs -f worker web` |
| Scan now | `docker compose exec worker mynotes scan` |
| Stop / start | `docker compose stop` / `docker compose start` |
| Update mynotes | `docker compose pull`, then `docker compose up -d` (`init` applies schema upgrades). From source: `git pull`, then `docker compose -f docker-compose.yml -f docker-compose.build.yml up -d --build` |
| psql | `docker compose exec postgres psql -U mynotes` |

`docker compose down` keeps the data; `down -v` deletes it (database, search index and state).

### Backup and restore

```bash
# Backup: the database, plus your settings
docker compose exec -T postgres pg_dump -U mynotes -Fc mynotes > mynotes-$(date +%F).dump
cp mynotes.yaml mynotes-$(date +%F).yaml

# Restore: load the dump, then rebuild the search indices from the database
docker compose stop worker web
docker compose exec -T postgres pg_restore -U mynotes -d mynotes --clean --if-exists --no-owner < mynotes-YYYY-MM-DD.dump
docker compose run --rm worker mynotes init
docker compose run --rm worker mynotes index rebuild --yes
docker compose run --rm worker mynotes index rebuild --entities --yes
docker compose start worker web
```

`.env` is not part of the backup: keep a copy somewhere safe.

## Limits of running in Docker

- **No open-in-app.** The "Open in app" button and `mynotes open` can't launch desktop apps from a container. The web UI can still show and download the original files.
- **Polling watcher.** Inside a container the watcher polls every 2 s instead of using native file events. On very large folders, set `watch: false` on the root and rely on the scheduled scans (every 15 minutes by default).
- **Read-only notes.** mynotes never writes to your notes; the mount enforces it.
- **Host-only commands don't apply.** `mynotes service`, `mynotes backup` and `mynotes restore` are for the native install; use Docker's restart policy and the backup commands above.
- **Paths.** Documents show paths relative to their root, so they look the same as on the host, but the root itself is `/notes/...`.
