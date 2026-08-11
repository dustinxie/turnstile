# Local gate — run before every commit. CI (host TBD) will invoke `make check`.

.PHONY: check sync lint format types contracts test test-all

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
