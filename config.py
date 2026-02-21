"""Central configuration using Pydantic Settings."""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # General
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # Database
    database_url: str = Field(default="sqlite:///data/docschat.db")

    # ChromaDB
    chroma_persist_dir: str = Field(default="data/chromadb")
    default_collection: str = Field(default="default")

    # Embeddings
    embedding_provider: str = Field(default="openai")
    embedding_model: str = Field(default="text-embedding-3-small")
    local_embedding_model: str = Field(default="all-MiniLM-L6-v2")

    # LLM Providers
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4o-mini")
    gemini_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-2.0-flash")
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2")
    default_provider: str = Field(default="openai")

    # RAG Settings
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    top_k: int = Field(default=5)

    # API Security
    api_key: str = Field(default="")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid:
            raise ValueError(f"log_level must be one of: {valid}")
        return v.upper()

    @field_validator("default_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        valid = ["openai", "gemini", "ollama"]
        if v.lower() not in valid:
            raise ValueError(f"default_provider must be one of: {valid}")
        return v.lower()

    @field_validator("embedding_provider")
    @classmethod
    def validate_embedding_provider(cls, v: str) -> str:
        valid = ["openai", "local"]
        if v.lower() not in valid:
            raise ValueError(f"embedding_provider must be one of: {valid}")
        return v.lower()


settings = Settings()
