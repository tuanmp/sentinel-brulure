.PHONY: lint test sync dev format help

help:
	@echo "Available targets:"
	@echo "  sync    - Install dependencies (uv sync)"
	@echo "  dev     - Install dev dependencies (includes ruff)"
	@echo "  lint    - Run ruff linting"
	@echo "  format  - Auto-format code with ruff"
	@echo "  test    - Run tests with pytest"

sync:
	uv sync

dev:
	uv sync --group dev

lint:
	uv run ruff check . --exclude notebooks

format:
	uv run ruff format . --exclude notebooks

test:
	PYTHONPATH=. uv run pytest tests/ -v