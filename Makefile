.PHONY: fmt lint types check test install-dev

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .

types:
	uv run mypy tmrl

check: lint types

test:
	uv run pytest

install-dev:
	uv sync --group dev
