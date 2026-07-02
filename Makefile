.PHONY: verify lint format format-check type test serve sync

UV_CACHE_DIR ?= /tmp/uv-cache
UV_RUN = UV_CACHE_DIR=$(UV_CACHE_DIR) uv run
UV_RUN_NOSYNC = UV_CACHE_DIR=$(UV_CACHE_DIR) UV_NO_SYNC=1 uv run

sync:
	uv sync

verify: lint format-check type test

lint:
	$(UV_RUN_NOSYNC) ruff check .

format:
	$(UV_RUN_NOSYNC) ruff format .

format-check:
	$(UV_RUN_NOSYNC) ruff format --check .

type:
	$(UV_RUN_NOSYNC) python -m mypy src/callosum

test:
	PYTHONPATH=src $(UV_RUN_NOSYNC) python -m pytest

serve:
	$(UV_RUN) callosum serve
