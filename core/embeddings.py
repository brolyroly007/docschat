"""Embedding providers: OpenAI and local sentence-transformers."""

from abc import ABC, abstractmethod

from loguru import logger

from config import settings


class BaseEmbedding(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query."""
        ...


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embeddings (text-embedding-3-small, etc.)."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        import openai

        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch embed texts using OpenAI API."""
        response = await self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        result = await self.embed([text])
        return result[0]


class LocalEmbedding(BaseEmbedding):
    """Local sentence-transformers embeddings (no API key needed)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading local embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using local model."""
        model = self._get_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        result = await self.embed([text])
        return result[0]


def get_embedding_provider() -> BaseEmbedding:
    """Factory: return configured embedding provider."""
    if settings.embedding_provider == "local":
        return LocalEmbedding(model_name=settings.local_embedding_model)
    else:
        if not settings.openai_api_key:
            logger.warning("No OpenAI API key set, falling back to local embeddings")
            return LocalEmbedding(model_name=settings.local_embedding_model)
        return OpenAIEmbedding(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )
