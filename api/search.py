"""Vector search endpoint — raw similarity search, no LLM."""

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from core.retriever import search

router = APIRouter()


class SearchRequest(BaseModel):
    """Search request body."""

    query: str = Field(..., min_length=1, max_length=5000)
    collection: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]+$")
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    """A single search result."""

    content: str
    metadata: dict
    score: float


@router.post("/search", response_model=list[SearchResult])
async def vector_search(request: SearchRequest):
    """Run a pure vector similarity search against a collection.

    Returns matching chunks ranked by similarity score.
    No LLM call, no conversation tracking.
    """
    try:
        chunks = await search(
            query=request.query,
            collection_name=request.collection,
            top_k=request.top_k,
        )
    except Exception as exc:
        logger.error(f"Vector search failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    return [
        SearchResult(
            content=chunk.text,
            metadata={"source": chunk.source, "chunk_index": chunk.chunk_index},
            score=round(chunk.score, 4),
        )
        for chunk in chunks
    ]
