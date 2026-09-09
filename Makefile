install:
	uv sync

run:
uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	
lint:

lint-strict:
