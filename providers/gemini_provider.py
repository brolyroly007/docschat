"""Google Gemini provider implementation (messages-based)."""

from collections.abc import AsyncIterator

import google.generativeai as genai
from loguru import logger

from providers.base import BaseProvider, GenerationResult


class GeminiProvider(BaseProvider):
    """Google Gemini provider.

    Translates messages list:
    - system message → system_instruction
    - assistant → model role
    - user stays user
    """

    name = "gemini"
    models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model_name = model
        genai.configure(api_key=api_key)

    def _prepare(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """Extract system prompt and convert messages to Gemini format."""
        system_prompt = ""
        gemini_history = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "assistant":
                gemini_history.append({"role": "model", "parts": [msg["content"]]})
            else:
                gemini_history.append({"role": "user", "parts": [msg["content"]]})

        return system_prompt, gemini_history

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> GenerationResult:
        """Generate using Gemini API."""
        system_prompt, history = self._prepare(messages)

        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt if system_prompt else None,
            )

            # Use last user message as the prompt, rest as history
            if len(history) > 1:
                chat = model.start_chat(history=history[:-1])
                response = await chat.send_message_async(
                    history[-1]["parts"][0],
                    generation_config=genai.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
            else:
                prompt = history[-1]["parts"][0] if history else ""
                response = await model.generate_content_async(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )

            content = response.text or ""
            tokens_used = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                tokens_used = getattr(response.usage_metadata, "total_token_count", 0)

            return GenerationResult(
                content=content,
                provider=self.name,
                model=self.model_name,
                tokens_used=tokens_used,
            )
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    async def stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> AsyncIterator[str]:
        """Stream using Gemini API."""
        system_prompt, history = self._prepare(messages)

        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt if system_prompt else None,
            )

            if len(history) > 1:
                chat = model.start_chat(history=history[:-1])
                response = await chat.send_message_async(
                    history[-1]["parts"][0],
                    generation_config=genai.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                    stream=True,
                )
            else:
                prompt = history[-1]["parts"][0] if history else ""
                response = await model.generate_content_async(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                    stream=True,
                )

            async for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise

    def is_available(self) -> bool:
        """Check if Gemini API key is configured."""
        return bool(self.api_key)
