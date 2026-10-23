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
limited by their ACL grants) and `admin`. Unknown or revoked tokens get **401**; a token used
outside its scopes gets **403**.

## Query endpoints

`POST /v1/ask`, `POST /v1/ask/stream` (server-sent events), `POST /v1/search`,
`GET /v1/documents/{id}`, `GET /v1/documents/lookup?ref=`, `GET /v1/files/{id}/raw`,
`GET /v1/tags`, `GET /v1/versions`, `GET /v1/graph/entities?q=`,
`GET /v1/graph/entities/{ref}/neighbors`, `GET /v1/graph/timeline?entity=`, `GET /v1/status`
(admin), `POST /v1/scan` (admin, local roots only), `GET /healthz` (no auth). Errors are JSON:
`{"error": "...", "detail": "..."}` with 404 (not found), 409 (ambiguous, with `candidates`),
403 (forbidden) or 503 (a model endpoint is down).

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
| `GET /v1/roots/{root}/status` | | files by status, extraction state of the latest versions, the last manifest, the agent's last heartbeat |

How a sync works: the agent sends a manifest (full, or the changed paths with `scope: paths`),
uploads the files listed in `upload`, and the server records each upload with the same rules as
a local scan (new, modified, moved, duplicate by hash, missing, deleted). Uploads are
idempotent: the same bytes again change nothing. A rename of an already uploaded file needs no
new bytes.
