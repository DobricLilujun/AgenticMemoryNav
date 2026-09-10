.PHONY: install test lint demo

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests main.py
	ruff format --check src tests main.py

demo:
	python main.py --config configs/default.yaml