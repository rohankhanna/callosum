.PHONY: verify lint format format-check type test arch-check serve sync

UV_CACHE_DIR ?= /tmp/uv-cache
UV_RUN = UV_CACHE_DIR=$(UV_CACHE_DIR) uv run
UV_RUN_NOSYNC = UV_CACHE_DIR=$(UV_CACHE_DIR) UV_NO_SYNC=1 uv run

sync:
	uv sync

verify: lint format-check type arch-check test

lint:
	$(UV_RUN_NOSYNC) ruff check .

format:
	$(UV_RUN_NOSYNC) ruff format .

format-check:
	$(UV_RUN_NOSYNC) ruff format --check .

type:
	$(UV_RUN_NOSYNC) python -m mypy src/callosum

arch-check:
	@tmpdir=$$(mktemp -d); \
	trap 'rm -rf $$tmpdir' EXIT; \
	$(UV_RUN_NOSYNC) python scripts/generate_static_graphs.py --out-dir $$tmpdir && \
	diff -r --exclude=review_thread_prompts.md $$tmpdir docs/architecture/generated/

test:
	PYTHONPATH=src $(UV_RUN_NOSYNC) python -m pytest

serve:
	$(UV_RUN) callosum serve
