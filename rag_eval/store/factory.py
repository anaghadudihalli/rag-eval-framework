"""Vector store factory.

Reads VECTOR_STORE_BACKEND from settings and returns the appropriate
VectorStoreBackend. If OpenSearch is configured but unreachable, it
automatically falls back to ChromaDB and logs a warning.

Usage:
    from rag_eval.store.factory import get_vector_store
    store = get_vector_store()  # uses Settings singleton
"""

from __future__ import annotations

import logging

try:
    # langchain-huggingface is the modern package (langchain_community.embeddings.HuggingFaceEmbeddings
    # was deprecated in LangChain 0.2.2 and will be removed in 1.0).
    # Install with: pip install langchain-huggingface
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        # Fallback: suppress the deprecation warning from the community package
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[no-redef]
    except ImportError:
        from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore[no-redef]

from rag_eval.config import Settings, get_settings
from rag_eval.store.base import VectorStoreBackend

logger = logging.getLogger(__name__)


def _build_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    """Construct the sentence-transformers embedding model.

    Uses all-MiniLM-L6-v2 by default — the same model as EA's production
    system. Loaded once and shared across the store and similarity evaluator.
    """
    logger.info("Loading embedding model: %s", settings.embedding_model)
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def get_vector_store(settings: Settings | None = None) -> VectorStoreBackend:
    """Instantiate and return the configured vector store backend.

    Decision logic:
    1. If VECTOR_STORE_BACKEND=chroma → use ChromaDB directly.
    2. If VECTOR_STORE_BACKEND=opensearch → try OpenSearch health check.
       - If healthy → use OpenSearch.
       - If unhealthy → fall back to ChromaDB and log a warning.

    Args:
        settings: Optional Settings override (useful in tests). Defaults
                  to the global Settings singleton.

    Returns:
        A VectorStoreBackend ready for ingest and search.
    """
    cfg = settings or get_settings()
    embeddings = _build_embeddings(cfg)

    if cfg.vector_store_backend == "chroma":
        logger.info("Using ChromaDB backend (explicitly configured).")
        return _make_chroma(cfg, embeddings)

    # Attempt OpenSearch
    try:
        from rag_eval.store.opensearch_store import OpenSearchBackend

        backend = OpenSearchBackend(
            opensearch_url=cfg.opensearch_url,
            index_name=cfg.opensearch_index,
            embeddings=embeddings,
        )
        if backend.is_healthy():
            logger.info("Using OpenSearch backend at %s.", cfg.opensearch_url)
            return backend
        else:
            logger.warning(
                "OpenSearch at %s is not reachable. Falling back to ChromaDB.",
                cfg.opensearch_url,
            )
    except ImportError:
        logger.warning(
            "langchain-opensearch not installed. Falling back to ChromaDB."
        )

    return _make_chroma(cfg, embeddings)


def _make_chroma(cfg: Settings, embeddings: object) -> VectorStoreBackend:
    """Construct a ChromaDB backend with current settings."""
    from rag_eval.store.chroma_store import ChromaBackend

    return ChromaBackend(
        collection_name=cfg.opensearch_index,  # reuse index name as collection name
        embeddings=embeddings,
    )
