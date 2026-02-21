# Contributing

## Development Setup

```bash
git clone https://github.com/brolyroly007/docschat.git
cd docschat
pip install -r requirements-dev.txt
pre-commit install
cp .env.example .env
```

## Workflow

1. Create a feature branch from `main`
2. Make your changes
3. Run `ruff check .` and `ruff format .`
4. Run `pytest tests/ -v`
5. Commit and push
6. Open a pull request

## Code Style

- Line length: 100 characters
- Formatter: Ruff
- Linter: Ruff (E, F, W, I, N, UP, B, SIM rules)

## Testing

```bash
pytest tests/ -v --tb=short
```

## Project Structure

- `api/` — FastAPI endpoints
- `cli/` — Typer CLI commands
- `core/` — RAG pipeline (chunker, embeddings, ingestion, retriever, rag)
- `database/` — SQLite connection and repositories
- `providers/` — LLM provider implementations
- `tests/` — Test suite
