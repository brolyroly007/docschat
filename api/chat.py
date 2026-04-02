"""Chat endpoint with SSE streaming."""

import json
from typing import Literal

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from core.rag import RAGOrchestrator

router = APIRouter()

VALID_PROVIDERS = ("openai", "gemini", "ollama")


class ChatRequest(BaseModel):
    """Chat request body."""

    question: str = Field(..., min_length=1, max_length=5000)
    collection: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]+$")
    provider: Literal["openai", "gemini", "ollama"] | None = None
    conversation_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    stream: bool = True


@router.post("/chat")
async def chat(request: ChatRequest):
    """Send a message and get a RAG-powered response.

    When stream=True (default), returns SSE events.
    When stream=False, returns JSON.
    """
    try:
        rag = RAGOrchestrator(
            collection=request.collection,
            provider_name=request.provider,
            conversation_id=request.conversation_id,
        )
    except Exception as exc:
        logger.error(f"Failed to initialize RAG orchestrator: {exc}")
        raise HTTPException(status_code=500, detail="Failed to initialize chat") from exc

    try:
        if not request.stream:
            result = await rag.query(request.question)
            return result

        async def event_generator():
            async for event in rag.stream(request.question):
                yield {
                    "event": event["type"],
                    "data": json.dumps(event["data"])
                    if isinstance(event["data"], (dict, list))
                    else event["data"],
                }

        return EventSourceResponse(event_generator())
    except Exception as exc:
        logger.error(f"Chat query failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Chat query failed: {exc}") from exc
