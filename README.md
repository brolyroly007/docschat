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

- **Multi-format ingestion** — PDF, DOCX, TXT, Markdown (max 50MB per file)
- **Pluggable embeddings** — OpenAI or local sentence-transformers
- **Multi-LLM support** — OpenAI, Gemini, Ollama
- **Vector search** — ChromaDB with persistent storage + standalone search endpoint
- **Multi-turn chat** — Conversation history with query rephrasing
- **Streaming** — SSE (API) and Rich Live (CLI) with configurable timeout
- **REST API** — Full CRUD for collections, documents, and conversations
- **Input validation** — Question length, collection names, file types, provider names
- **Rate limiting** — Per-IP sliding window (default 30 req/min)
- **Connection pooling** — Singleton SQLite connection with WAL mode

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
| `POST` | `/api/search` | Vector search without LLM (returns raw results) |
| `POST` | `/api/ingest` | Upload + ingest files |
| `GET` | `/api/collections` | List collections |
| `DELETE` | `/api/collections/{name}` | Delete collection |
| `GET` | `/api/collections/{name}/documents` | List documents |
| `DELETE` | `/api/collections/{name}/documents/{id}` | Delete individual document |
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
| `STREAM_TIMEOUT` | `120` | SSE stream timeout in seconds |
| `RATE_LIMIT_RPM` | `30` | Max requests per minute per IP |

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

## Input Validation

The API validates all inputs:

- **Question**: 1-5000 characters
- **Collection names**: alphanumeric, hyphens, underscores only (`^[a-zA-Z0-9_-]+$`)
- **Provider**: must be `openai`, `gemini`, or `ollama`
- **top_k**: 1-20
- **File uploads**: max 50MB, only `.pdf`, `.docx`, `.txt`, `.md`

Invalid requests return HTTP 422 with details.

## License

MIT
