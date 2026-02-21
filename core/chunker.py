"""Recursive character text splitter (no langchain dependency)."""

from dataclasses import dataclass


@dataclass
class Chunk:
    """A text chunk with metadata."""

    text: str
    index: int
    metadata: dict


class RecursiveCharacterSplitter:
    """Split text recursively by separators, respecting chunk size and overlap."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """Split text into chunks with overlap."""
        metadata = metadata or {}
        pieces = self._recursive_split(text, self.separators)
        chunks = self._merge_pieces(pieces)
        return [
            Chunk(text=chunk, index=i, metadata=metadata)
            for i, chunk in enumerate(chunks)
            if chunk.strip()
        ]

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text by trying separators in order."""
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # No separators left — hard split
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts = text.split(sep)
        result = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                # If part itself is too large, split it recursively
                if len(part) > self.chunk_size:
                    result.extend(self._recursive_split(part, remaining_seps))
                    current = ""
                else:
                    current = part

        if current:
            result.append(current)

        return result

    def _merge_pieces(self, pieces: list[str]) -> list[str]:
        """Merge small pieces and add overlap between chunks."""
        if not pieces:
            return []

        chunks = []
        for piece in pieces:
            if chunks and len(chunks[-1]) + len(piece) + 1 <= self.chunk_size:
                chunks[-1] += " " + piece
            else:
                chunks.append(piece)

        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        # Add overlap from previous chunk
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-self.chunk_overlap :] if len(prev) > self.chunk_overlap else prev
            overlapped.append(overlap_text + " " + chunks[i])

        return overlapped
