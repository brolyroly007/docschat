# DocsChat

[![CI](https://github.com/brolyroly007/docschat/actions/workflows/ci.yml/badge.svg)](https://github.com/brolyroly007/docschat/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F00?logo=databricks&logoColor=white)](https://www.trychroma.com)

RAG chat system: ingest documents (PDF, DOCX, TXT, MD), embed them in ChromaDB, and chat with them using any LLM.

Exposes both a **FastAPI REST API** and a **Typer CLI**.

## Features

- **Multi-format ingestion** — PDF, DOCX, TXT, Markdown
- **Pluggable embeddings** — OpenAI or local sentence-transformers
- **Multi-LLM support** — OpenAI, Gemini, Ollama
- **Vector search** — ChromaDB with persistent storage
- **Multi-turn chat** — Conversation history with query rephrasing
- **Streaming** — SSE (API) and Rich Live (CLI)
- **REST API** — Full CRUD for collections and conversations

## Quick Start

```bash
# Clone and install
git clone https://github.com/brolyroly007/docschat.git
cd docschat
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys

# Ingest documents
python -m cli.app ingest ./docs/
python -m cli.app ingest report.pdf -c myproject

# Chat
python -m cli.app chat                         # Interactive REPL
python -m cli.app chat -q "What is X?"         # Single question
python -m cli.app chat -c myproject -p ollama  # Specific collection + provider

# Manage collections
python -m cli.app collections
python -m cli.app collections delete myproject

# System status
python -m cli.app status

# Start API server
python app.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Send message (streaming SSE) |
| `POST` | `/api/ingest` | Upload + ingest files |
| `GET` | `/api/collections` | List collections |
| `DELETE` | `/api/collections/{name}` | Delete collection |
| `GET` | `/api/collections/{name}/documents` | List documents |
| `GET` | `/api/conversations` | List conversations |
| `GET` | `/api/conversations/{id}` | Get conversation history |
| `DELETE` | `/api/conversations/{id}` | Delete conversation |
| `GET` | `/api/health` | Health check |

## Configuration

All settings are configured via `.env` file or environment variables. See `.env.example` for all options.

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `DEFAULT_PROVIDER` | `openai` | Default LLM provider |
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `local` |
| `CHUNK_SIZE` | `1000` | Text chunk size |
| `TOP_K` | `5` | Number of chunks to retrieve |

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Lint
ruff check .

# Format
ruff format .
```

## Tech Stack

- **FastAPI** — REST API
- **Typer + Rich** — CLI
- **ChromaDB** — Vector store
- **aiosqlite** — Async SQLite
- **Pydantic Settings** — Configuration
- **Loguru** — Logging

## License

MIT
