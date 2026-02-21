"""Tests for CLI commands."""

from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from cli.app import app

runner = CliRunner()


class TestCLIApp:
    def test_version(self):
        """--version should print version and exit."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output

    def test_no_args_shows_help(self):
        """No arguments should show help text (exit code 0 or 2)."""
        result = runner.invoke(app, [])
        assert result.exit_code in (0, 2)  # Typer no_args_is_help returns 2
        assert "DocsChat" in result.output


class TestIngestCommand:
    def test_ingest_missing_path(self):
        """Should error when path doesn't exist."""
        result = runner.invoke(app, ["ingest", "/nonexistent/path"])
        assert result.exit_code != 0

    def test_ingest_txt_file(self, sample_txt_file):
        """Should ingest a text file successfully."""
        mock_result = {
            "document_id": 1,
            "filename": "test_doc.txt",
            "collection": "default",
            "chunks": 3,
            "file_size": 200,
        }

        with patch(
            "cli.ingest_cmd.ingest_file",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = runner.invoke(app, ["ingest", str(sample_txt_file)])
            assert result.exit_code == 0
            assert "test_doc.txt" in result.output


class TestCollectionsCommand:
    def test_list_empty(self):
        """Should handle empty collections."""
        with patch("cli.collections_cmd.get_chroma_client") as mock_client:
            mock_client.return_value.list_collections.return_value = []
            result = runner.invoke(app, ["collections"])
            assert result.exit_code == 0
            assert "No collections" in result.output

    def test_list_with_collections(self):
        """Should display collections table."""
        mock_col = MagicMock()
        mock_col.name = "docs"
        mock_col.count.return_value = 42

        with patch("cli.collections_cmd.get_chroma_client") as mock_client:
            mock_client.return_value.list_collections.return_value = [mock_col]
            result = runner.invoke(app, ["collections"])
            assert result.exit_code == 0
            assert "docs" in result.output


class TestStatusCommand:
    def test_status_display(self):
        """Should display status information."""
        with (
            patch("cli.status_cmd.DB_PATH") as mock_path,
            patch("cli.status_cmd.get_db", new_callable=AsyncMock),
            patch("cli.status_cmd.get_chroma_client") as mock_chroma,
            patch("cli.status_cmd.list_providers", return_value=[]),
        ):
            mock_path.exists.return_value = False
            mock_chroma.return_value.list_collections.return_value = []

            result = runner.invoke(app, ["status"])
            assert result.exit_code == 0
            assert "DocsChat" in result.output
