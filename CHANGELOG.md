# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-02-21

### Added
- Document ingestion pipeline (PDF, DOCX, TXT, MD)
- ChromaDB vector store with persistent storage
- Pluggable embeddings (OpenAI and local sentence-transformers)
- Multi-LLM support (OpenAI, Gemini, Ollama)
- RAG orchestrator with multi-turn chat and query rephrasing
- FastAPI REST API with SSE streaming
- Typer CLI with interactive REPL
- SQLite database for conversations and document metadata
- Optional API key authentication
- Collection management (list, delete)
- Conversation history (list, view, delete)
- CI/CD with GitHub Actions
