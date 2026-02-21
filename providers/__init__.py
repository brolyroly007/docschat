"""Provider factory and registry."""

from config import settings
from providers.base import BaseProvider, GenerationResult
from providers.gemini_provider import GeminiProvider
from providers.ollama_provider import OllamaProvider
from providers.openai_provider import OpenAIProvider

_PROVIDERS: dict[str, BaseProvider] = {}
_initialized = False


def _init_providers():
    global _initialized
    if _initialized:
        return
    if settings.openai_api_key:
        _PROVIDERS["openai"] = OpenAIProvider(settings.openai_api_key, settings.openai_model)
    if settings.gemini_api_key:
        _PROVIDERS["gemini"] = GeminiProvider(settings.gemini_api_key, settings.gemini_model)
    _PROVIDERS["ollama"] = OllamaProvider(settings.ollama_base_url, settings.ollama_model)
    _initialized = True


def get_provider(name: str | None = None) -> BaseProvider:
    """Get a provider by name. Defaults to the configured default."""
    _init_providers()
    name = name or settings.default_provider
    if name not in _PROVIDERS:
        available = list(_PROVIDERS.keys())
        raise ValueError(f"Provider '{name}' not available. Available: {available}")
    return _PROVIDERS[name]


def list_providers() -> list[dict]:
    """List all registered providers with their metadata."""
    _init_providers()
    return [p.info() for p in _PROVIDERS.values()]


__all__ = ["get_provider", "list_providers", "BaseProvider", "GenerationResult"]
