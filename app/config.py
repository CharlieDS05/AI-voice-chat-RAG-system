"""Central, validated configuration for the RAG system.

All settings are read from environment variables / the .env file.
This is the single source of truth for the local-vs-hosted LLM switch.
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # reads from .env in development, but in production env vars can be set directly
        # and it will ignore the missing .env file
        env_file=".env",
        # explicitly specify encoding to avoid issues on different platforms
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated env variables instead of crashing
    )

    # The switch. Pydantic enforces it's one of exactly these two values.
    # "="local"" assigns local as the default option
    llm_provider: Literal["local", "hosted"] = "local"

    # Local provider (Ollama)
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.2:3b"

    # Hosted provider (OpenAI-compatible)
    hosted_base_url: str = "https://api.openai.com/v1"
    hosted_model: str = "gpt-4o-mini"
    # repr=False: never print the secret when the object is printed
    hosted_api_key: str = Field(default="", repr=False)

    # Ingestion & chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150
    min_chunk_chars: int = 50


# A single shared instance the rest of the app imports.
# Here the instructions for how to configure the app are centralized, and Pydantic
# will validate the settings on startup, erroring if something is misconfigured.
settings = Settings()
