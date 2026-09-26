.PHONY: install run visualize debug clean lint

install:
	uv sync

run:
	uv run python -m src

visualize:
	uv run python -m src --visualize

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +

lint:
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs
