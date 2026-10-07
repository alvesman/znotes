# Backup and restore

mynotes keeps everything it needs to reconstruct its search and knowledge graph
in **PostgreSQL**. The backup procedure therefore dumps Postgres and nothing
else; both OpenSearch indices are rebuilt from Postgres after a restore.

## What `mynotes backup` produces

`mynotes backup [--out DIR]` writes into `DIR` (default:
`<user data dir>/mynotes/backups`):

| File | Contents |
|---|---|
| `mynotes_<UTC timestamp>.dump` | `pg_dump -Fc` (custom format) of the configured `mynotes` database: documents, versions, chunks, tags, relations, KG, caches, jobs, outbox and the `alembic_version` marker |
| `mynotes.yaml` | Copy of the configuration file *if one is in use* (repo root, user config dir or `MYNOTES_CONFIG`) |

**Never** `.env`: it holds secrets (Postgres/OpenSearch passwords, API keys)
and is intentionally excluded. Keep it safe yourself, e.g. in a password
manager.

The dump uses the Docker Compose service (`docker compose exec -T postgres
pg_dump …`) when the configured DSN points at it, and falls back to a local
`pg_dump` from `PATH` otherwise. You can copy the `.dump` file anywhere; it is
a single self-contained archive.

Scheduling is up to you (the worker does not run backups on its own), for
example nightly in cron:

```bash
0 5 * * * cd /path/to/mynotes && uv run mynotes backup
```

## Why OpenSearch is not backed up

OpenSearch is a derived store: every chunks/entities document is re-rendered
from Postgres rows plus the `embedding_cache`, and
`mynotes index rebuild --yes` builds a fresh physical index from those rows and
swaps the alias atomically. Restoring a stale index dump could even reintroduce
drift, so backups only cover the source of truth.

## What the backup does not cover: your notes folder

The notes folder itself is **your responsibility**. mynotes never modifies
files under a source root; it only reads them (and caches derived data). A
database backup without the notes folder is of limited use on a new machine;
include the folder in whatever file-level backup you already use.

## Restoring

`mynotes restore FILE [--yes]` runs, in order:

1. `pg_restore --clean --if-exists` of the dump into the configured database
   (same Compose-or-local detection as the backup). **This overwrites all
   current mynotes data.**
2. `alembic upgrade head`, so a dump from an older schema is brought up to date.
3. `mynotes index rebuild --yes` for the chunks index and the entities index,
   recreating OpenSearch from the restored Postgres.

Without `--yes` the command asks for confirmation first.

## Restoring on a new machine

1. Install `uv` and get the repository (same release).
2. Recreate `.env` from `.env.example` with the **same** secrets as before
   (`MYNOTES_PG_DSN`, `MYNOTES_OPENSEARCH_PASSWORD`, …); copy `mynotes.yaml`
   to the repo root (or the user config dir) if you use one.
3. `docker compose up -d` and wait until both services are `healthy`.
4. `uv run mynotes restore mynotes_YYYYMMDDTHHMMSSZ.dump`
5. Point `roots` in `mynotes.yaml` at your notes folder (a path from the old
   machine can be kept via a symlink) and run `uv run mynotes scan --full` if
   the folder contents or paths changed; the reconciler reports any drift.

The restored database already contains the workspace, source roots, documents
and the KG; the rebuild only recreates the search indices from it.
