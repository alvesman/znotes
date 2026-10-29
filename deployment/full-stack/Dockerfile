# mynotes application image: the CLI, worker, web/API server and MCP server.
# Build context is the repository root (see docker-compose.yml); only the files
# listed in Dockerfile.dockerignore are sent to the builder, never .env.
#
# Two stages: the builder has uv and resolves the virtualenv; the runtime image copies
# /app (the virtualenv and the sources it points at) and never carries uv (57 MB).

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# The project itself; alembic/ must sit next to src/ (mynotes init runs the migrations from it).
COPY README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY alembic ./alembic
COPY docs ./docs
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev


FROM python:3.12-slim

# Set by scripts/release/publish_image.sh; harmless defaults for local builds.
ARG VERSION=dev
ARG SOURCE_URL=https://github.com/alvesman/znotes
LABEL org.opencontainers.image.title="mynotes" \
      org.opencontainers.image.description="Ask questions about a folder of notes: local RAG with citations, versions and a knowledge graph" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.version="${VERSION}"

ENV PYTHONUNBUFFERED=1

# Same interpreter path as the builder, so the virtualenv's symlinks keep working.
COPY --from=builder /app /app

# Unprivileged user. Its home holds the state files (reconcile report, watcher heartbeats)
# and the tokenizer caches; docker-compose.yml mounts a volume there.
RUN useradd --create-home --uid 1000 mynotes \
    && mkdir -p /home/mynotes/.local/state /home/mynotes/.cache \
    && chown -R mynotes:mynotes /home/mynotes
ENV PATH=/app/.venv/bin:$PATH \
    HOME=/home/mynotes \
    TIKTOKEN_CACHE_DIR=/home/mynotes/.cache/tiktoken
WORKDIR /app
USER mynotes

CMD ["mynotes", "--help"]
