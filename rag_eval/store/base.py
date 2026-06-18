"""Abstract base class for vector store backends.

Both OpenSearch and ChromaDB backends implement this interface.
All metric and pipeline code depends only on this ABC — never on a
concrete backend — keeping the retriever interface backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class VectorStoreBackend(ABC):
    """Protocol for pluggable vector store backends.

    Implementing classes: OpenSearchBackend, ChromaBackend.
    Use store.factory.get_vector_store() to instantiate the correct backend.
    """

    @abstractmethod
    def ingest(self, docs: list[Document]) -> None:
        """Add documents to the vector store.

        Args:
            docs: LangChain Document objects with page_content and metadata.
                  metadata MUST include a "doc_id" field for retrieval scoring.
        """
        ...

    @abstractmethod
    def as_retriever(self, k: int) -> BaseRetriever:
        """Return a LangChain retriever configured to fetch the top-k docs.

        Args:
            k: Number of documents to retrieve per query.

        Returns:
            A BaseRetriever that can be plugged into any LangChain chain.
        """
        ...

    @abstractmethod
    def similarity_search_with_score(
        self,
        query: str,
        k: int,
    ) -> list[tuple[Document, float]]:
        """Return documents with their similarity scores.

        Used by the confidence router to obtain raw similarity values.

        Args:
            query: The natural-language query.
            k: Number of results to return.

        Returns:
            List of (Document, score) tuples sorted by descending relevance.
            Score semantics vary by backend (cosine for both, but range may differ).
        """
        ...

    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if the backend is reachable and ready.

        Used by the factory to decide whether to fall back to ChromaDB.
        """
        ...
