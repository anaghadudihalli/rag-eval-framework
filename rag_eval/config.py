"""Application configuration via pydantic-settings.

All settings are read from environment variables (with .env file support).
Use `get_settings()` to access the singleton instance throughout the app.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration for the RAG eval pipeline.

    Values are read from environment variables or a .env file.
    All fields have sensible defaults so the system works out-of-the-box
    against local Docker OpenSearch with minimal configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ─────────────────────────────────────────────────────────────
    openai_api_key: SecretStr = SecretStr("sk-placeholder")
    judge_model: str = "gpt-3.5-turbo"

    # ── Vector store ────────────────────────────────────────────────────────
    vector_store_backend: Literal["opensearch", "chroma"] = "opensearch"

    # ── OpenSearch ──────────────────────────────────────────────────────────
    opensearch_url: str = "http://localhost:9200"
    opensearch_index: str = "rag-eval-docs"

    # ── Embeddings ──────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Retrieval ───────────────────────────────────────────────────────────
    top_k: int = 5

    # ── Metric thresholds ───────────────────────────────────────────────────
    confidence_threshold: float = 0.7
    similarity_threshold: float = 0.75
    hallucination_threshold: float = 0.5  # judge confidence below this → uncertain

    # ── Data paths ──────────────────────────────────────────────────────────
    golden_dataset_path: str = "data/golden_dataset.json"
    documents_dir: str = "data/documents"
    reports_dir: str = "reports"

    @field_validator("top_k")
    @classmethod
    def top_k_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("top_k must be >= 1")
        return v

    @field_validator("confidence_threshold", "similarity_threshold", "hallucination_threshold")
    @classmethod
    def threshold_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Thresholds must be between 0.0 and 1.0")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton.

    Using lru_cache ensures the .env file is read once and the same
    instance is shared across the entire application.
    """
    return Settings()
