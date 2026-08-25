.PHONY: install test lint demo

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

demo:
	python scripts/run_demo.py --config configs/default.yaml