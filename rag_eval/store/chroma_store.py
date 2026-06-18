"""ChromaDB vector store backend.

This is the FALLBACK backend used when OpenSearch is unavailable
(e.g., no Docker, CI without services, or VECTOR_STORE_BACKEND=chroma).

Persists data to ./chroma_db/ by default. The retriever interface is
identical to OpenSearchBackend — all downstream metric code is unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_eval.store.base import VectorStoreBackend

logger = logging.getLogger(__name__)

_PERSIST_DIR = "chroma_db"


class ChromaBackend(VectorStoreBackend):
    """Vector store backend backed by ChromaDB (local persistence).

    Used as the fallback when OpenSearch is unreachable. Preserves all
    the same abstractions so no metric or pipeline code needs to change.

    Args:
        collection_name: Name of the ChromaDB collection.
        embeddings: LangChain Embeddings instance.
        persist_dir: Directory for ChromaDB on-disk persistence.
    """

    def __init__(
        self,
        collection_name: str,
        embeddings: Any,
        persist_dir: str = _PERSIST_DIR,
    ) -> None:
        self._collection = collection_name
        self._embeddings = embeddings
        self._persist_dir = persist_dir
        self._store: Any = None

    def _get_store(self) -> Any:
        """Lazily initialize the Chroma instance."""
        if self._store is None:
            from langchain_chroma import Chroma

            self._store = Chroma(
                collection_name=self._collection,
                embedding_function=self._embeddings,
                persist_directory=self._persist_dir,
            )
        return self._store

    def ingest(self, docs: list[Document]) -> None:
        """Ingest documents into ChromaDB, replacing any existing collection."""
        from langchain_chroma import Chroma

        logger.info(
            "Ingesting %d documents into ChromaDB collection '%s'",
            len(docs),
            self._collection,
        )

        # Reset and recreate so repeated ingest runs don't duplicate docs
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
        self._store = Chroma.from_documents(
            documents=docs,
            embedding=self._embeddings,
            collection_name=self._collection,
            persist_directory=self._persist_dir,
        )
        logger.info("ChromaDB ingestion complete. Persisted to '%s'.", self._persist_dir)

    def as_retriever(self, k: int) -> BaseRetriever:
        """Return a LangChain retriever for the top-k results."""
        return self._get_store().as_retriever(search_kwargs={"k": k})

    def similarity_search_with_score(
        self,
        query: str,
        k: int,
    ) -> list[tuple[Document, float]]:
        """L2-distance search — Chroma returns distance, not cosine similarity.

        Note: Chroma's default metric is L2. We convert to a [0, 1] similarity
        approximation via: sim = 1 / (1 + distance). This is a proxy, not true
        cosine similarity. For production, configure Chroma with cosine distance.
        """
        # similarity_search_with_score returns (doc, distance) for Chroma
        results = self._get_store().similarity_search_with_score(query, k=k)
        # Convert L2 distance to similarity-like score for uniform downstream use
        return [(doc, 1.0 / (1.0 + dist)) for doc, dist in results]

    def is_healthy(self) -> bool:
        """ChromaDB is always healthy if the package is installed."""
        try:
            import chromadb  # noqa: F401

            return True
        except ImportError:
            return False
