"""Shared test fixtures."""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

# Set test environment before importing app modules
os.environ["DATABASE_URL"] = "sqlite:///test_data/test.db"
os.environ["CHROMA_PERSIST_DIR"] = "test_data/chromadb"
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["EMBEDDING_PROVIDER"] = "local"


@pytest.fixture
def sample_text():
    """Sample text for chunking tests."""
    return (
        "Python is a high-level programming language. "
        "It was created by Guido van Rossum and first released in 1991. "
        "Python's design philosophy emphasizes code readability. "
        "It supports multiple programming paradigms including structured, "
        "object-oriented, and functional programming.\n\n"
        "Python is dynamically typed and garbage-collected. "
        "It has a large standard library and an active community. "
        "Python is widely used in web development, data science, "
        "artificial intelligence, and scientific computing.\n\n"
        "The language has simple and consistent syntax that makes it "
        "easy to learn for beginners while remaining powerful enough "
        "for experienced developers."
    )


@pytest.fixture
def sample_txt_file(tmp_path):
    """Create a temporary text file for ingestion tests."""
    file = tmp_path / "test_doc.txt"
    file.write_text(
        "This is a test document for DocsChat.\n\n"
        "It contains multiple paragraphs to test chunking.\n\n"
        "The document discusses testing strategies and approaches.\n\n"
        "Each paragraph provides some content for the system to process."
    )
    return file


@pytest.fixture
def mock_provider():
    """Mock LLM provider for testing."""
    from providers.base import GenerationResult

    provider = MagicMock()
    provider.name = "mock"
    provider.model = "mock-model"
    provider.is_available.return_value = True

    result = GenerationResult(
        content="This is a test response.",
        provider="mock",
        model="mock-model",
        tokens_used=42,
    )
    provider.generate = AsyncMock(return_value=result)

    async def mock_stream(*args, **kwargs):
        for word in ["This ", "is ", "a ", "test ", "response."]:
            yield word

    provider.stream = mock_stream
    return provider


@pytest.fixture
def mock_embedding():
    """Mock embedding provider."""
    embedding = MagicMock()

    async def mock_embed(texts):
        return [[0.1] * 384 for _ in texts]

    async def mock_embed_query(text):
        return [0.1] * 384

    embedding.embed = mock_embed
    embedding.embed_query = mock_embed_query
    return embedding
