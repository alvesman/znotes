# mynotes images, from one Dockerfile with two final stages:
#   server (the default, last stage): the CLI, worker, web/API server and MCP server;
#   agent  (`--target agent`): only `mynotes agent`, for each person's machine, about a third
#          of the size, since it installs the core dependencies without the `server` extra.
# Build context is the repository root (see docker-compose.yml); only the files
# listed in Dockerfile.dockerignore are sent to the builder, never .env.

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE THIRD_PARTY_NOTICES.md ./


# --- agent: core dependencies, the sources, nothing else
FROM builder AS build-agent

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev


# --- server: the `server` extra; alembic/ must sit next to src/ (mynotes init runs the migrations)
FROM builder AS build-server

# Dependencies first, so code changes don't reinstall them.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra server --no-install-project
COPY alembic ./alembic
COPY docs ./docs
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra server


# --- what both final images share: labels, the unprivileged user, the environment
FROM python:3.12-slim AS runtime

# Set by scripts/release/publish_image.sh; harmless defaults for local builds.
ARG VERSION=dev
ARG SOURCE_URL=https://github.com/alvesman/znotes
LABEL org.opencontainers.image.title="mynotes" \
      org.opencontainers.image.description="Ask questions about a folder of notes: local RAG with citations, versions and a knowledge graph" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.version="${VERSION}"

ENV PYTHONUNBUFFERED=1

# Unprivileged user. Its home holds the state files (reconcile report, watcher heartbeats,
# the agent's hash cache) and the tokenizer caches; the compose files mount a volume there.
RUN useradd --create-home --uid 1000 mynotes \
    && mkdir -p /home/mynotes/.local/state /home/mynotes/.cache \
    && chown -R mynotes:mynotes /home/mynotes
ENV PATH=/app/.venv/bin:$PATH \
    HOME=/home/mynotes \
    TIKTOKEN_CACHE_DIR=/home/mynotes/.cache/tiktoken
WORKDIR /app
USER mynotes


FROM runtime AS agent
# Same interpreter path as the builder, so the virtualenv's symlinks keep working.
COPY --from=build-agent /app /app
CMD ["mynotes", "agent", "run"]


FROM runtime AS server
COPY --from=build-server /app /app
CMD ["mynotes", "--help"]
