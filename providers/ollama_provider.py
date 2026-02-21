"""Ollama local LLM provider implementation (messages-based)."""

import json
from collections.abc import AsyncIterator

import httpx
from loguru import logger

from providers.base import BaseProvider, GenerationResult


class OllamaProvider(BaseProvider):
    """Ollama local LLM provider.

    Uses /api/chat (messages-based) instead of /api/generate.
    """

    name = "ollama"
    models = ["llama3.2", "llama3.1", "mistral", "codellama", "phi3"]

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> GenerationResult:
        """Generate using Ollama /api/chat endpoint."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                tokens_used = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
                content = data.get("message", {}).get("content", "")

                return GenerationResult(
                    content=content,
                    provider=self.name,
                    model=self.model,
                    tokens_used=tokens_used,
                    finish_reason="stop" if data.get("done") else "length",
                )
        except httpx.HTTPError as e:
            logger.error(f"Ollama API error: {e}")
            raise

    async def stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> AsyncIterator[str]:
        """Stream using Ollama /api/chat endpoint."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            async with (
                httpx.AsyncClient(timeout=120.0) as client,
                client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json=payload,
                ) as response,
            ):
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if data.get("done"):
                            break
        except httpx.HTTPError as e:
            logger.error(f"Ollama streaming error: {e}")
            raise

    def is_available(self) -> bool:
        """Check if Ollama is reachable (best-effort sync check)."""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False
