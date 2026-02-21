"""Tests for document ingestion pipeline."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.ingestion import SUPPORTED_EXTENSIONS, parse_file


class TestParseFile:
    def test_parse_txt(self, sample_txt_file):
        """Should parse .txt files."""
        text = parse_file(sample_txt_file)
        assert "test document" in text
        assert len(text) > 0

    def test_parse_md(self, tmp_path):
        """Should parse .md files."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Title\n\nSome markdown content.")
        text = parse_file(md_file)
        assert "Title" in text
        assert "markdown content" in text

    def test_unsupported_extension(self, tmp_path):
        """Should raise for unsupported file types."""
        file = tmp_path / "test.xyz"
        file.write_text("content")
        with pytest.raises(ValueError, match="Unsupported"):
            parse_file(file)


class TestSupportedExtensions:
    def test_pdf_supported(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS

    def test_docx_supported(self):
        assert ".docx" in SUPPORTED_EXTENSIONS

    def test_txt_supported(self):
        assert ".txt" in SUPPORTED_EXTENSIONS

    def test_md_supported(self):
        assert ".md" in SUPPORTED_EXTENSIONS


class TestIngestFile:
    @pytest.mark.asyncio
    async def test_ingest_txt_file(self, sample_txt_file, mock_embedding):
        """Should ingest a text file end-to-end."""
        with (
            patch("core.ingestion.get_embedding_provider", return_value=mock_embedding),
            patch("core.ingestion.get_chroma_client") as mock_chroma,
            patch("core.ingestion.get_db") as mock_get_db,
            patch("core.ingestion.create_document", new_callable=AsyncMock, return_value=1),
        ):
            # Mock ChromaDB
            mock_collection = MagicMock()
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

            # Mock DB
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db

            from core.ingestion import ingest_file

            result = await ingest_file(sample_txt_file, collection_name="test")

            assert result["filename"] == "test_doc.txt"
            assert result["collection"] == "test"
            assert result["chunks"] > 0
            mock_collection.upsert.assert_called_once()
