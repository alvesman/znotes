sync:
	uv sync

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

test:
	uv run pytest

test-int:
	uv run pytest -m integration

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

psql:
	docker compose exec postgres psql -U mynotes -d mynotes

TAG ?= run

eval-retrieval:
	uv run python -m eval.run_eval --mode retrieval --tag $(TAG)

eval-full:
	uv run python -m eval.run_eval --mode both --tag $(TAG)

eval-compare:
	uv run python -m eval.compare $(A) $(B)

# Synthetic smoke tests (docs/eval/smoke-tests.md): isolated database and indices, no judge.
# Run after changes to jobs, the graph or retrieval. ~1.5 + ~1 + ~5 min (LLM extraction).
SMOKE_FLAGS = --reset --smoke --mode retrieval --no-judge --runs auto

smoke-version:
	uv run python scripts/eval/run_version_eval.py --kind version $(SMOKE_FLAGS)

smoke-multihop:
	uv run python scripts/eval/run_version_eval.py --kind multihop --with-graph $(SMOKE_FLAGS)

smoke-series:
	uv run python scripts/eval/run_version_eval.py --kind series $(SMOKE_FLAGS)

smoke: smoke-version smoke-series smoke-multihop

.PHONY: sync fmt lint test test-int up down logs psql eval-retrieval eval-full eval-compare smoke smoke-version smoke-series smoke-multihop
