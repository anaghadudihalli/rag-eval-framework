"""LLM factory — returns the configured chat model.

Supports two providers:
  - openai (default): GPT-3.5-turbo. Requires OPENAI_API_KEY with billing.
  - groq (free tier): llama-3.1-8b-instant via Groq. Requires only GROQ_API_KEY.
    Sign up free (no credit card) at https://console.groq.com.
    Free tier: 14,400 requests/day, 6,000 requests/minute.

Switch providers with a single env var:
    LLM_PROVIDER=groq   GROQ_API_KEY=gsk_...   make eval

Note: Groq does not support logprobs. Confidence routing will automatically
fall back to the cosine-similarity proxy — this is handled transparently.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from rag_eval.config import Settings, get_settings

logger = logging.getLogger(__name__)


def get_chat_model(
    settings: Settings | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    # OpenAI-specific: logprobs for confidence routing
    with_logprobs: bool = False,
) -> BaseChatModel:
    """Return a configured chat model for the active LLM provider.

    Args:
        settings: Settings override (defaults to global singleton).
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens: Maximum tokens to generate.
        with_logprobs: If True and provider is OpenAI, enables per-token
            log-probabilities for confidence routing. Ignored for Groq
            (Groq does not expose logprobs).

    Returns:
        A LangChain BaseChatModel ready for .invoke().
    """
    cfg = settings or get_settings()

    if cfg.llm_provider == "groq":
        return _make_groq_model(cfg, temperature=temperature, max_tokens=max_tokens)

    return _make_openai_model(
        cfg,
        temperature=temperature,
        max_tokens=max_tokens,
        with_logprobs=with_logprobs,
    )


def _make_openai_model(
    cfg: Settings,
    temperature: float,
    max_tokens: int,
    with_logprobs: bool,
) -> BaseChatModel:
    """Build a ChatOpenAI instance."""
    from langchain_openai import ChatOpenAI

    kwargs: dict = dict(
        model=cfg.judge_model,
        api_key=cfg.openai_api_key.get_secret_value(),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if with_logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 1

    logger.debug("LLM: OpenAI / %s", cfg.judge_model)
    return ChatOpenAI(**kwargs)


def _make_groq_model(
    cfg: Settings,
    temperature: float,
    max_tokens: int,
) -> BaseChatModel:
    """Build a ChatGroq instance (free tier, no logprobs support)."""
    from langchain_groq import ChatGroq

    if cfg.groq_api_key is None:
        raise ValueError(
            "LLM_PROVIDER=groq requires GROQ_API_KEY to be set. "
            "Get a free key at https://console.groq.com"
        )

    logger.debug("LLM: Groq / %s (logprobs unavailable — similarity fallback active)", cfg.groq_model)
    return ChatGroq(
        model=cfg.groq_model,
        api_key=cfg.groq_api_key.get_secret_value(),
        temperature=temperature,
        max_tokens=max_tokens,
    )
