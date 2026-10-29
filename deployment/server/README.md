# Server: one mynotes for several people, with agents on their machines

The server runs PostgreSQL, OpenSearch, the mynotes worker and the web UI/API in Docker. It
never sees a notes folder: on each person's machine a small **agent** watches their folder and
uploads the files through the API; the server keeps a copy (the blob store), extracts, embeds and
indexes them. Each person sees their own notes only and signs in to the web UI with a personal
token. The pieces:

| Where                 | What                                                                                        |
| --------------------- | ------------------------------------------------------------------------------------------- |
| Server (this folder)  | `postgres`, `opensearch`, `init`, `worker`, `web`, and optionally `caddy` (TLS) |
| Each person's machine | `mynotes agent run`, natively with uv or in the agent container (`<version>-agent`)       |

## Server setup

1. Copy the files and fill them in:

   ```bash
   cp .env.example .env && chmod 600 .env        # passwords, web binding, model endpoints
   cp mynotes.example.yaml mynotes.yaml
   ```
2. Decide how HTTPS is terminated. Tokens travel in requests and in a cookie, so the web UI/API
   must only be reachable over TLS:

   - **You already have a TLS-terminating proxy** (pfSense/HAProxy, nginx, Traefik, a cloud load
     balancer): set `MYNOTES_WEB_BIND` in `.env` to the address the proxy reaches (for example
     `0.0.0.0:8765`, or the LAN interface's address), point the proxy at that port, and have it
     send `X-Forwarded-Proto: https` so the sign-in cookie is marked `Secure`. Nothing else.
   - **You have none:** `cp Caddyfile.example Caddyfile`, set `MYNOTES_SERVER_HOST` to the
     server's DNS name, and start with `--profile tls` below. Caddy gets a certificate when ports
     80 and 443 are reachable from the internet (see `Caddyfile` for a private network). Leave
     `MYNOTES_WEB_BIND` at its localhost default: Caddy reaches `web` inside Docker.
3. Check the models, then start:

   ```bash
   docker compose run --rm init mynotes check-endpoints
   docker compose up -d --wait                   # with Caddy: docker compose --profile tls up -d --wait
   ```
4. Create tokens for each person (an admin token for yourself first):

   ```bash
   docker compose exec worker mynotes token create --user admin --name me --admin
   docker compose exec worker mynotes token create --user alice --name laptop --root alice-notes
   docker compose exec worker mynotes token create --user alice --name web --web
   docker compose exec worker mynotes acl grant admin ""   # let the admin read every root
   ```

   What a person can read comes from ACL grants, not from the token: `--root` grants its user
   that root, but `--admin` only allows the admin actions. Without the `acl grant` above, the admin
   sees no notes and empty review queues. The grant names the token's `--user` (`acl grant manuel ""`
   for a token created with `--user manuel`), and a running server picks up a new grant within a
   minute (it caches each user's grants for 60 s).
   `--root alice-notes` creates that remote root for alice if it does not exist yet (with her
   read access under it). Each `token create` prints its secret once: pass the agent token to
   alice for her agent and the web token for signing in at `https://<server>/ui/login`. One
   token can carry several scopes (`--root ... --web --admin`). `mynotes token list` and
   `mynotes token revoke <id>` manage them later; `mynotes root list` shows the roots and when
   each agent was last seen.

## Which token to create

All commands run on the server, from the compose directory; each prints its secret once.

| Who                                     | Command                                                                                                     | What it allows                                                                                                                               |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| The agent on a person's machine         | `docker compose exec worker mynotes token create --user alice --name laptop --root alice-notes`           | Uploads to that root only (created for alice if missing); it cannot sign in or read                                                          |
| A person, in the browser                | `docker compose exec worker mynotes token create --user alice --name web --web`                           | Signs in at `/ui/login`; reads the roots alice owns and whatever `acl grant` adds                                                        |
| An administrator                        | `docker compose exec worker mynotes token create --user admin --name web --admin`                         | Admin actions and sign-in; reading still needs `mynotes acl grant admin ""` (once)                                                         |
| One person who is agent owner and admin | `docker compose exec worker mynotes token create --user manuel --name laptop --root manuel-notes --admin` | Everything above with a single secret, used by the agent and the browser; reading other people's roots needs `mynotes acl grant manuel ""` |
| An agent feeding several roots          | `... token create --user alice --name laptop --root alice-notes --root alice-work`                        | Repeat `--root` for each                                                                                                                   |

Later: `docker compose exec worker mynotes token list` shows them (id, user, scopes, last use) and
`docker compose exec worker mynotes token revoke <id>` disables one; revoke and create a new one
when a secret may have leaked.

## The agent, on a person's machine

The agent must run the **same version as the server** (`MYNOTES_VERSION` in the server's
`.env`): the ingest API is not promised to stay the same between versions.

Natively (Python 3.12 and uv, any OS):

```bash
git clone https://github.com/alvesman/znotes.git mynotes && cd mynotes
git checkout v0.4.0rc14                       # the server's version
uv sync --no-dev                             # the agent's dependencies only
export MYNOTES_AGENT_SERVER_URL=https://notes.example.org
export MYNOTES_AGENT_TOKEN=mn_...            # the agent token
export MYNOTES_AGENT_ROOT=alice-notes
export MYNOTES_AGENT_PATH=~/Documents/notes
uv run mynotes agent sync                    # one full sync, then exit
uv run mynotes agent run                     # sync, then watch the folder until stopped
```

Or in a container, with the folder mounted read-only. The `-agent` image carries only what the
agent needs (about a third of the server image):

```bash
docker run -d --name mynotes-agent --restart unless-stopped \
  -v "$HOME/Documents/notes:/notes:ro" \
  -e MYNOTES_AGENT_SERVER_URL=https://notes.example.org -e MYNOTES_AGENT_TOKEN=mn_... \
  -e MYNOTES_AGENT_ROOT=alice-notes -e MYNOTES_AGENT_PATH=/notes \
  ghcr.io/alvesman/znotes:0.4.0rc14-agent              # the server's version, not "latest"
```

The same settings fit in a `mynotes.yaml` next to the agent (`agent:` section, see the
repository's `mynotes.example.yaml`); keep the token out of files you share. The agent needs no
database, search index or model endpoint. It hashes files once (cached), uploads only new or
changed ones, re-lists the folder every 15 minutes, reacts to changes within seconds, and
retries with backoff when the server is unreachable. `mynotes agent status` shows what the
server knows about the root.

## What people see

- **Web UI** at `https://<server>/`: sign in with the web token; Chat, Search, History, Graph
  and the document pages show only the notes of that person's roots. **Open** and **Download**
  serve the server's copy of the file. **Open in app** is not offered for uploaded notes (the
  original is on the owner's machine).
- **Reviews**: everyone can accept or reject the suggested tags of the documents they can read.
  Confirming version links, merging entities and merging tags need an admin token; other people
  see those proposals without the buttons.
- **Admin** (a token with `--admin`): the Status page (each agent-fed root appears with the watcher
  state *agent*, `agent on <host>` and its last heartbeat, or *down* after 2 minutes without one) and the admin review actions above. The blob
  store's size is in `mynotes status` and `/v1/status`. **Scan now** scans local roots only, so
  it does nothing for agent-fed roots: their agents send the changes.
- **Not found means not yours.** An entity of the knowledge graph exists for a person only if
  some of its evidence is in a document they may read; otherwise its card, neighbours and
  timeline answer "not found".

## Operate

- `docker compose exec worker mynotes status` and `root list` for an overview.
- Deleting a person's notes: `mynotes root remove <name>` removes them from search (rows and
  files are kept; `retention.deleted_days` later purges and frees them). Revoke their tokens
  with `mynotes token revoke`.
- Backups: `pg_dump` of the database plus the `apphome` volume (the uploaded files).
- Updating: bump `MYNOTES_VERSION` in `.env`, then `docker compose pull && docker compose up -d`
  (init applies migrations). Update the agents to the same version.
- Limits: the files are uploaded whole (`storage.max_upload_mb`, default 200 MB); the MCP server
  and the `ask`/`search` commands still need direct database access, so they run on the server.
  They run without access control: they see **every person's notes**, so only the server's
  administrator should use them.
