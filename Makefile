# Local gate — run before every commit. CI (host TBD) will invoke `make check`.

.PHONY: check sync lint format types contracts test test-all image frontend-check frontend

check: sync lint format types contracts test frontend-check

sync:
	uv sync --frozen

lint:
	uv run ruff check .

format:
	uv run ruff format --check .

types:
	uv run pyright

contracts:
	uv run lint-imports

test:
	uv run pytest -m "not integration"

test-all:
	uv run pytest

# Image tag: `make image` -> turnstile:latest, `make image TAG=testing` ->
# turnstile:testing. (Make has no positional args — TAG=x is the idiom.)
TAG ?= latest

image:
	docker build -t turnstile:$(TAG) \
		--build-arg GIT_COMMIT=$$(git rev-parse HEAD) \
		--build-arg DOCKER_TAG=$(TAG) .

# The frontend half of the gate (typecheck, lint, vitest, build). Part of
# `make check` so one command guards every commit; installs deps on first run.
frontend-check:
	test -d frontend/node_modules || (cd frontend && npm install)
	cd frontend && npm run check

# The frontend image (nginx + built SPA), from its own build context.
frontend:
	docker build -t turnstile-frontend:$(TAG) \
		--build-arg GIT_COMMIT=$$(git rev-parse HEAD) \
		--build-arg DOCKER_TAG=$(TAG) frontend
