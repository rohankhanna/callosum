.PHONY: verify lint format format-check type test serve sync

sync:
	uv sync

verify: lint format-check type test

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

type:
	uv run mypy src/callosum

test:
	uv run pytest

serve:
	uv run callosum serve
