"""Tests for API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client."""
    with patch("database.connection.init_db", new_callable=AsyncMock):
        from app import app

        return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        """Health endpoint should return 200 with status ok."""
        with patch("api.health.list_providers", return_value=[]):
            response = client.get("/api/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["version"] == "1.0.0"


class TestCollectionsEndpoint:
    def test_list_collections(self, client):
        """Should return list of collections."""
        mock_col = MagicMock()
        mock_col.name = "test"
        mock_col.count.return_value = 10

        with patch("api.collections.get_chroma_client") as mock_client:
            mock_client.return_value.list_collections.return_value = [mock_col]
            response = client.get("/api/collections")
            assert response.status_code == 200
            data = response.json()
            assert len(data["collections"]) == 1
            assert data["collections"][0]["name"] == "test"


class TestConversationsEndpoint:
    def test_list_conversations(self, client):
        """Should return list of conversations."""
        with (
            patch("api.conversations.get_db", new_callable=AsyncMock) as mock_get_db,
            patch(
                "api.conversations.list_conversations",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db
            response = client.get("/api/conversations")
            assert response.status_code == 200
            assert response.json()["conversations"] == []

    def test_get_conversation_not_found(self, client):
        """Should return 404 for unknown conversation."""
        with (
            patch("api.conversations.get_db", new_callable=AsyncMock) as mock_get_db,
            patch(
                "api.conversations.get_conversation",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db
            response = client.get("/api/conversations/unknown-id")
            assert response.status_code == 404
