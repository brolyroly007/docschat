"""Chat endpoint with SSE streaming."""

import json

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from core.rag import RAGOrchestrator

router = APIRouter()


class ChatRequest(BaseModel):
    """Chat request body."""

    question: str
    collection: str | None = None
    provider: str | None = None
    conversation_id: str | None = None
    stream: bool = True


@router.post("/chat")
async def chat(request: ChatRequest):
    """Send a message and get a RAG-powered response.

    When stream=True (default), returns SSE events.
    When stream=False, returns JSON.
    """
    rag = RAGOrchestrator(
        collection=request.collection,
        provider_name=request.provider,
        conversation_id=request.conversation_id,
    )

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
