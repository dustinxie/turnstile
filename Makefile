# Local gate — run before every commit. CI (host TBD) will invoke `make check`.

.PHONY: check sync lint format types contracts test test-all image

check: sync lint format types contracts test

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
