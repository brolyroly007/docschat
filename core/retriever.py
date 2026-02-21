"""ChromaDB vector search and context building."""

from dataclasses import dataclass

from loguru import logger

from config import settings
from core.embeddings import get_embedding_provider
from core.ingestion import get_chroma_client


@dataclass
class RetrievedChunk:
    """A chunk retrieved from vector search."""

    text: str
    source: str
    chunk_index: int
    score: float


async def search(
    query: str,
    collection_name: str | None = None,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Search ChromaDB for chunks relevant to the query."""
    collection_name = collection_name or settings.default_collection
    top_k = top_k or settings.top_k

    client = get_chroma_client()

    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        logger.warning(f"Collection '{collection_name}' not found")
        return []

    # Embed the query
    embedding_provider = get_embedding_provider()
    query_embedding = await embedding_provider.embed_query(query)

    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()) if collection.count() > 0 else top_k,
    )

    if not results["documents"] or not results["documents"][0]:
        return []

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i] if results["metadatas"] else {}
        distance = results["distances"][0][i] if results["distances"] else 0.0
        chunks.append(
            RetrievedChunk(
                text=doc,
                source=metadata.get("source", "unknown"),
                chunk_index=metadata.get("chunk_index", 0),
                score=1.0 - distance,  # Convert distance to similarity
            )
        )

    return chunks


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Build context string from retrieved chunks."""
    if not chunks:
        return ""

    context_parts = []
    for chunk in chunks:
        context_parts.append(f"[Source: {chunk.source}, Chunk {chunk.chunk_index}]\n{chunk.text}")

    return "\n\n---\n\n".join(context_parts)


def build_sources(chunks: list[RetrievedChunk]) -> list[dict]:
    """Build source references for storage."""
    return [
        {
            "source": chunk.source,
            "chunk_index": chunk.chunk_index,
            "score": round(chunk.score, 4),
            "preview": chunk.text[:200],
        }
        for chunk in chunks
    ]
