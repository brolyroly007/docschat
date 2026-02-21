.PHONY: install dev lint format test run clean

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt
	pre-commit install

lint:
	ruff check .

format:
	ruff format .

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=. --cov-report=term-missing

run:
	python app.py

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache .ruff_cache .mypy_cache
	rm -rf test_data/ data/ htmlcov/ .coverage
	rm -rf *.egg-info dist/ build/
