"""Tests for the RAG orchestrator."""

from unittest.mock import AsyncMock, patch

import pytest

from core.rag import RAGOrchestrator
from core.retriever import RetrievedChunk


class TestRAGOrchestrator:
    @pytest.mark.asyncio
    async def test_query_returns_answer(self, mock_provider):
        """Should return an answer from the RAG pipeline."""
        chunks = [
            RetrievedChunk(
                text="Python was created by Guido van Rossum.",
                source="test.txt",
                chunk_index=0,
                score=0.9,
            )
        ]

        with (
            patch("core.rag.search", new_callable=AsyncMock, return_value=chunks),
            patch("core.rag.get_provider", return_value=mock_provider),
            patch("core.rag.get_db") as mock_get_db,
            patch("core.rag.create_conversation", new_callable=AsyncMock, return_value="conv-123"),
            patch("core.rag.get_messages", new_callable=AsyncMock, return_value=[]),
            patch("core.rag.add_message", new_callable=AsyncMock),
            patch("core.rag.update_conversation", new_callable=AsyncMock),
        ):
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db

            rag = RAGOrchestrator(collection="test")
            result = await rag.query("Who created Python?")

            assert "answer" in result
            assert result["conversation_id"] == "conv-123"
            assert result["provider"] == "mock"
            assert len(result["sources"]) > 0

    @pytest.mark.asyncio
    async def test_query_without_context(self, mock_provider):
        """Should handle case when no documents are found."""
        with (
            patch("core.rag.search", new_callable=AsyncMock, return_value=[]),
            patch("core.rag.get_provider", return_value=mock_provider),
            patch("core.rag.get_db") as mock_get_db,
            patch("core.rag.create_conversation", new_callable=AsyncMock, return_value="conv-456"),
            patch("core.rag.get_messages", new_callable=AsyncMock, return_value=[]),
            patch("core.rag.add_message", new_callable=AsyncMock),
            patch("core.rag.update_conversation", new_callable=AsyncMock),
        ):
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db

            rag = RAGOrchestrator(collection="empty")
            result = await rag.query("What is this about?")

            assert "answer" in result
            assert result["sources"] == []

    @pytest.mark.asyncio
    async def test_stream_yields_events(self, mock_provider):
        """Should yield source, token, and done events."""
        chunks = [
            RetrievedChunk(
                text="Test context.",
                source="doc.txt",
                chunk_index=0,
                score=0.85,
            )
        ]

        with (
            patch("core.rag.search", new_callable=AsyncMock, return_value=chunks),
            patch("core.rag.get_provider", return_value=mock_provider),
            patch("core.rag.get_db") as mock_get_db,
            patch("core.rag.create_conversation", new_callable=AsyncMock, return_value="conv-789"),
            patch("core.rag.get_messages", new_callable=AsyncMock, return_value=[]),
            patch("core.rag.add_message", new_callable=AsyncMock),
            patch("core.rag.update_conversation", new_callable=AsyncMock),
        ):
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db

            rag = RAGOrchestrator(collection="test")
            events = []
            async for event in rag.stream("test question"):
                events.append(event)

            event_types = [e["type"] for e in events]
            assert "sources" in event_types
            assert "token" in event_types
            assert "done" in event_types
