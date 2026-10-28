# The HTTP API

mynotes's web UI uses the same FastAPI app as this API; the interactive description is at
`/docs` on a running server. This page covers what a program needs: how to authenticate, the
query endpoints in brief, and the **ingest API** the agent of server mode uses.

## Authentication

| `auth.mode` | Who the request is | How |
|---|---|---|
| `none` (default) | The local user (admin, no restrictions) | Nothing; bind the server to localhost |
| `token` (server mode) | The owner of a personal token | `Authorization: Bearer mn_...`, or the `mynotes_token` cookie set by `/ui/login` |
| `oidc` | The user in a JWT from your identity provider | `Authorization: Bearer <jwt>` |

Personal tokens (`mynotes token create`) are honoured in every mode. Their scopes:
`ingest:<root>` (an agent feeding one remote root), `web` (a person's read and review access,
limited by their ACL grants) and `admin` (the admin actions; reading still follows the ACL
grants). Unknown or revoked tokens get **401**. A token without `web` or `admin` reads nothing
(empty results, "not found") and cannot sign in to the web UI; an ingest route for a root the
token does not feed, or an admin action without `admin`, gets **403**.

## Query and review endpoints

Every endpoint below needs a token with the `web` or `admin` scope in `token` mode, and answers
only with what the caller's ACL grants let them read.

| Endpoint | What it does |
|---|---|
| `POST /v1/ask` | A cited answer. Body: `{"question", "filters"?}` |
| `POST /v1/ask/stream` | The same as server-sent events: `rewritten` (a follow-up, rewritten into a standalone question), `token` (answer text), `sources` (where a note uploaded by an agent can't be opened in an app, its citation carries `openable: false`), `done`, or `error`. Extra body fields: `conversation_id` (groups the turns of one chat; earlier turns make follow-ups work) and `save_history` (default `true`: keep the answer in History) |
| `POST /v1/search` | Ranked passages. Body: `{"query", "filters"?, "rerank"? (default true), "limit"? (default 10)}` |
| `GET /v1/documents/{id}`, `GET /v1/documents/lookup?ref=` | A document's text and metadata, by id or by path, title or id |
| `GET /v1/documents/{id}/history` | The document's snapshots |
| `GET /v1/files/{id}/raw` | The original file |
| `POST /v1/files/{id}/open` | Open the original in its desktop application (local mode only; 204) |
| `GET /v1/tags?prefix=` | Tags with their usage counts |
| `POST /v1/tags/{doc}/{tag}/accept`, `.../reject` | Decide on a suggested tag |
| `POST /v1/tags/merge` | Merge tag `from` into `into` (admin) |
| `GET /v1/versions?status=` | Version links (`proposed`, `auto`, `confirmed`, `rejected`) |
| `POST /v1/versions/{id}/confirm`, `.../reject`, `.../swap` | Decide on a version link (admin) |
| `GET /v1/graph/entities?q=` | An entity's card |
| `GET /v1/graph/entities/{ref}/neighbors?hops=` | Its neighbourhood (`hops` default 1) |
| `GET /v1/graph/timeline?entity=` | The timeline of an entity or of `project:NAME` |
| `GET /v1/status` | The status report (admin) |
| `POST /v1/scan` | Start a scan of the local roots (admin; 202) |
| `GET /healthz` | Liveness (no auth) |

`filters` is `{"tags": [...], "folders": [...], "formats": [...], "since": "...", "until": "...", "history": false}`;
inline filters in the question or query (`#tag`, `in:folder`, …) work too.

Errors from the core come as JSON `{"error": "...", "detail": "..."}` with 404 (not found), 409
(ambiguous, with `candidates`), 403 (forbidden) or 503 (a model endpoint is down). Authentication
failures (401) and the ingest API's errors use FastAPI's `{"detail": "..."}`.

Tokens, ACL grants and roots have no HTTP endpoints: manage them with the `mynotes token`,
`mynotes acl` and `mynotes root` commands on the server.

## Ingest API (server mode)

All paths are relative to the agent's folder (POSIX separators, NFC). The server stores them
prefixed with the root's name, which is what the ACL and the web UI show. Every route needs a
token with `ingest:<root>`.

| Endpoint | Body / headers | Answer |
|---|---|---|
| `POST /v1/roots/{root}/manifest` | `{"scope": "full" \| "paths", "paths": [...], "entries": [{"rel_path", "size", "mtime_ns", "blake3", "placeholder"}]}`; `Content-Encoding: gzip` accepted | `{"scan_run_id", "counts", "upload": [paths whose bytes the server lacks], "actions": {path: action}, "skipped": [ignored extensions]}` |
| `PUT /v1/roots/{root}/files/{rel_path}` | raw bytes; headers `X-Content-Blake3`, `X-Mtime-Ns`, `X-Size` | `{"action", "content_hash", "size"}`; 422 on a hash mismatch, 413 over `storage.max_upload_mb`, 415 for a format mynotes does not extract |
| `DELETE /v1/roots/{root}/files/{rel_path}[?force=true]` | | `{"action": "missing" \| "deleted"}` (missing first, deleted after `scan.delete_grace_seconds`; `force` deletes at once) |
| `POST /v1/roots/{root}/files/{rel_path}/move` | `{"to", "size", "mtime_ns", "blake3"}` | `{"action": "moved" \| "new", "upload": [...]}` |
| `POST /v1/roots/{root}/heartbeat` | `{"agent_version", "hostname", "path", "watch_state"}` | 204 |
| `GET /v1/roots/{root}/status` | | files by status, extraction state of the latest versions, the last manifest, the agent's last heartbeat, `max_upload_mb` |

Other answers: 400 for a body that is not valid gzip, 413 for a manifest with more than
`storage.max_manifest_entries` entries, 404 for an unknown root, 409 for a root that is local or
disabled, 503 when the server has no blob store (`storage.dir`).

How a sync works: the agent sends a manifest (full, or the changed paths with `scope: paths`),
uploads the files listed in `upload`, and the server records each upload with the same rules as
a local scan (new, modified, moved, duplicate by hash, missing, deleted). Uploads are
idempotent: the same bytes again change nothing. A rename of an already uploaded file needs no
new bytes.
