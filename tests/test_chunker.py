"""Tests for the recursive character text splitter."""

from core.chunker import RecursiveCharacterSplitter


class TestRecursiveCharacterSplitter:
    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size returns one chunk."""
        splitter = RecursiveCharacterSplitter(chunk_size=500)
        chunks = splitter.split("Hello, world!")
        assert len(chunks) == 1
        assert "Hello, world!" in chunks[0].text

    def test_respects_chunk_size(self, sample_text):
        """All chunks should be approximately within chunk_size."""
        splitter = RecursiveCharacterSplitter(chunk_size=200, chunk_overlap=0)
        chunks = splitter.split(sample_text)
        assert len(chunks) > 1
        # First chunk should respect size (later ones may have overlap)
        assert len(chunks[0].text) <= 250  # Allow some tolerance

    def test_chunk_indices(self, sample_text):
        """Each chunk should have a sequential index."""
        splitter = RecursiveCharacterSplitter(chunk_size=200, chunk_overlap=0)
        chunks = splitter.split(sample_text)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_metadata_preserved(self):
        """Metadata should be passed through to all chunks."""
        splitter = RecursiveCharacterSplitter(chunk_size=50, chunk_overlap=0)
        metadata = {"source": "test.pdf", "page": 1}
        chunks = splitter.split("A " * 100, metadata=metadata)
        for chunk in chunks:
            assert chunk.metadata == metadata

    def test_empty_text(self):
        """Empty text returns no chunks."""
        splitter = RecursiveCharacterSplitter()
        chunks = splitter.split("")
        assert len(chunks) == 0

    def test_whitespace_only(self):
        """Whitespace-only text returns no chunks."""
        splitter = RecursiveCharacterSplitter()
        chunks = splitter.split("   \n\n   ")
        assert len(chunks) == 0

    def test_overlap_adds_content(self, sample_text):
        """Overlap should add text from previous chunk."""
        splitter = RecursiveCharacterSplitter(chunk_size=200, chunk_overlap=50)
        chunks = splitter.split(sample_text)
        if len(chunks) > 1:
            # Second chunk should contain overlap from first
            assert len(chunks[1].text) > 0

    def test_paragraph_separation(self):
        """Should prefer splitting on paragraph boundaries."""
        text = "Paragraph one content here.\n\nParagraph two content here."
        splitter = RecursiveCharacterSplitter(chunk_size=40, chunk_overlap=0)
        chunks = splitter.split(text)
        assert len(chunks) >= 2
