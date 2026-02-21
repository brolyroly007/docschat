"""OpenAI GPT provider implementation (messages-based)."""

from collections.abc import AsyncIterator

import openai
from loguru import logger

from providers.base import BaseProvider, GenerationResult


class OpenAIProvider(BaseProvider):
    """OpenAI GPT provider — messages are passed natively."""

    name = "openai"
    models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> GenerationResult:
        """Generate using OpenAI chat completions."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return GenerationResult(
                content=response.choices[0].message.content or "",
                provider=self.name,
                model=self.model,
                tokens_used=response.usage.total_tokens if response.usage else 0,
                finish_reason=response.choices[0].finish_reason or "stop",
            )
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> AsyncIterator[str]:
        """Stream using OpenAI chat completions."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except openai.APIError as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise

    def is_available(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.client.api_key)
