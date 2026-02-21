"""RAG orchestrator: retrieve context → build prompt → generate response."""

from collections.abc import AsyncIterator

from loguru import logger

from config import settings
from core.retriever import RetrievedChunk, build_context, build_sources, search
from database import get_db
from database.repositories import (
    add_message,
    create_conversation,
    get_messages,
    update_conversation,
)
from providers import get_provider

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided document context.

Use the following context to answer the user's question. If the context doesn't contain enough information to answer, say so clearly.

Context:
{context}

Instructions:
- Answer based on the context above
- Cite sources when possible (e.g., "According to [source]...")
- If the context doesn't cover the question, say you don't have enough information
- Be concise and accurate"""

REPHRASE_PROMPT = """Given the conversation history and the latest question, rephrase the question to be a standalone search query.

Conversation:
{history}

Latest question: {question}

Standalone search query:"""


class RAGOrchestrator:
    """Orchestrates the full RAG pipeline."""

    def __init__(
        self,
        collection: str | None = None,
        provider_name: str | None = None,
        conversation_id: str | None = None,
    ):
        self.collection = collection or settings.default_collection
        self.provider_name = provider_name
        self.conversation_id = conversation_id

    async def _ensure_conversation(self) -> str:
        """Get or create a conversation."""
        if self.conversation_id:
            return self.conversation_id

        db = await get_db()
        try:
            self.conversation_id = await create_conversation(db, collection=self.collection)
        finally:
            await db.close()
        return self.conversation_id

    async def _get_chat_history(self) -> list[dict]:
        """Get chat history as messages list."""
        if not self.conversation_id:
            return []

        db = await get_db()
        try:
            msgs = await get_messages(db, self.conversation_id)
        finally:
            await db.close()

        return [{"role": m["role"], "content": m["content"]} for m in msgs]

    async def _rephrase_query(self, question: str, history: list[dict]) -> str:
        """Rephrase question using chat history for better retrieval."""
        if not history:
            return question

        history_text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:])

        provider = get_provider(self.provider_name)
        messages = [
            {
                "role": "user",
                "content": REPHRASE_PROMPT.format(history=history_text, question=question),
            }
        ]

        result = await provider.generate(messages, temperature=0.0, max_tokens=200)
        rephrased = result.content.strip()
        logger.debug(f"Rephrased query: {rephrased}")
        return rephrased

    async def query(self, question: str) -> dict:
        """Run the full RAG pipeline (non-streaming)."""
        conv_id = await self._ensure_conversation()
        history = await self._get_chat_history()

        # Rephrase for multi-turn
        search_query = await self._rephrase_query(question, history) if history else question

        # Retrieve
        chunks: list[RetrievedChunk] = await search(search_query, collection_name=self.collection)
        context = build_context(chunks)
        sources = build_sources(chunks)

        # Build messages
        system_prompt = (
            SYSTEM_PROMPT_TEMPLATE.format(context=context)
            if context
            else (
                "You are a helpful assistant. No document context is available — "
                "answer based on your general knowledge and let the user know."
            )
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history[-10:])  # Last 10 messages for context
        messages.append({"role": "user", "content": question})

        # Generate
        provider = get_provider(self.provider_name)
        result = await provider.generate(messages)

        # Store messages
        db = await get_db()
        try:
            await add_message(db, conv_id, "user", question)
            await add_message(
                db,
                conv_id,
                "assistant",
                result.content,
                sources=sources,
                tokens_used=result.tokens_used,
            )
            # Set conversation title from first question
            if not history:
                title = question[:100] + ("..." if len(question) > 100 else "")
                await update_conversation(db, conv_id, title=title)
        finally:
            await db.close()

        return {
            "conversation_id": conv_id,
            "answer": result.content,
            "sources": sources,
            "provider": result.provider,
            "model": result.model,
            "tokens_used": result.tokens_used,
        }

    async def stream(self, question: str) -> AsyncIterator[dict]:
        """Run RAG pipeline with streaming response."""
        conv_id = await self._ensure_conversation()
        history = await self._get_chat_history()

        # Rephrase for multi-turn
        search_query = await self._rephrase_query(question, history) if history else question

        # Retrieve
        chunks: list[RetrievedChunk] = await search(search_query, collection_name=self.collection)
        context = build_context(chunks)
        sources = build_sources(chunks)

        # Build messages
        system_prompt = (
            SYSTEM_PROMPT_TEMPLATE.format(context=context)
            if context
            else (
                "You are a helpful assistant. No document context is available — "
                "answer based on your general knowledge and let the user know."
            )
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history[-10:])
        messages.append({"role": "user", "content": question})

        # Yield sources first
        yield {"type": "sources", "data": sources}

        # Stream response
        provider = get_provider(self.provider_name)
        full_response = ""
        async for chunk in provider.stream(messages):
            full_response += chunk
            yield {"type": "token", "data": chunk}

        # Store messages
        db = await get_db()
        try:
            await add_message(db, conv_id, "user", question)
            await add_message(db, conv_id, "assistant", full_response, sources=sources)
            if not history:
                title = question[:100] + ("..." if len(question) > 100 else "")
                await update_conversation(db, conv_id, title=title)
        finally:
            await db.close()

        yield {
            "type": "done",
            "data": {"conversation_id": conv_id, "provider": provider.name},
        }
