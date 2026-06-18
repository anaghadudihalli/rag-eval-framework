"""OpenSearch vector store backend.

Wraps langchain_opensearch.OpenSearchVectorSearch, which itself wraps
the opensearch-py client. The index is created with knn=True on first
ingest so a fresh Docker environment requires no manual index setup.

This backend mirrors the pattern used in EA's production RAG systems
(OpenSearch on AWS with the k-NN plugin).
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from opensearchpy import OpenSearch

from rag_eval.store.base import VectorStoreBackend

logger = logging.getLogger(__name__)


class OpenSearchBackend(VectorStoreBackend):
    """Vector store backend backed by OpenSearch with the k-NN plugin.

    The index uses HNSW (Hierarchical Navigable Small World) for approximate
    nearest-neighbor search — the same algorithm used in the EA OpenSearch setup.

    Args:
        opensearch_url: HTTP URL of the OpenSearch node.
        index_name: Name of the k-NN index to use (created if not exists).
        embeddings: LangChain Embeddings instance for vectorizing text.
    """

    def __init__(
        self,
        opensearch_url: str,
        index_name: str,
        embeddings: Any,
    ) -> None:
        self._url = opensearch_url
        self._index = index_name
        self._embeddings = embeddings
        self._store: Any = None  # lazily initialized on first ingest/search

    def _get_store(self) -> Any:
        """Lazily initialize the OpenSearchVectorSearch instance."""
        if self._store is None:
            # Import here to allow the module to load even if langchain-opensearch
            # is not installed (factory handles the ImportError gracefully)
            from langchain_opensearch import OpenSearchVectorSearch

            self._store = OpenSearchVectorSearch(
                opensearch_url=self._url,
                index_name=self._index,
                embedding_function=self._embeddings,
                # engine="faiss" is available but nmslib is the OS default
                engine="nmslib",
                space_type="cosinesimil",
            )
        return self._store

    def ingest(self, docs: list[Document]) -> None:
        """Ingest documents into OpenSearch.

        Uses add_documents on an existing store, or from_documents to
        create the index if it doesn't exist yet.
        """
        from langchain_opensearch import OpenSearchVectorSearch

        logger.info("Ingesting %d documents into OpenSearch index '%s'", len(docs), self._index)

        # from_documents creates the index with correct knn settings if needed
        self._store = OpenSearchVectorSearch.from_documents(
            documents=docs,
            embedding=self._embeddings,
            opensearch_url=self._url,
            index_name=self._index,
            engine="nmslib",
            space_type="cosinesimil",
        )
        logger.info("Ingestion complete.")

    def as_retriever(self, k: int) -> BaseRetriever:
        """Return a LangChain retriever for the top-k results."""
        return self._get_store().as_retriever(search_kwargs={"k": k})

    def similarity_search_with_score(
        self,
        query: str,
        k: int,
    ) -> list[tuple[Document, float]]:
        """Cosine similarity search returning (doc, score) pairs."""
        return self._get_store().similarity_search_with_score(query, k=k)

    def is_healthy(self) -> bool:
        """Ping the OpenSearch cluster health endpoint."""
        try:
            # Use a lightweight raw client ping — avoids embedding model overhead
            host = self._url.replace("http://", "").replace("https://", "")
            host_part, _, port_part = host.partition(":")
            port = int(port_part) if port_part else 9200
            client = OpenSearch(
                hosts=[{"host": host_part, "port": port}],
                http_compress=True,
                use_ssl=False,
                verify_certs=False,
            )
            return client.ping()
        except Exception as exc:
            logger.warning("OpenSearch health check failed: %s", exc)
            return False
