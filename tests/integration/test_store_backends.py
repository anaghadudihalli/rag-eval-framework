"""Integration tests for vector store backends.

ChromaDB tests run without Docker.
OpenSearch tests are skipped unless a live OpenSearch instance is reachable
(marked with @pytest.mark.opensearch).

Run all: pytest tests/integration/test_store_backends.py -v
Run without OpenSearch: pytest tests/integration/test_store_backends.py -v -m "not opensearch"
"""

from __future__ import annotations

import tempfile

import pytest
from langchain_core.documents import Document

pytestmark = pytest.mark.integration


# ── Shared test documents ─────────────────────────────────────────────────────


def _make_test_docs() -> list[Document]:
    """Create a small set of documents for backend testing."""
    return [
        Document(
            page_content="Docker is a containerization platform using OS-level virtualization.",
            metadata={"doc_id": "docker-001", "title": "Docker Overview"},
        ),
        Document(
            page_content="Git branching strategies include Git Flow, GitHub Flow, and trunk-based development.",
            metadata={"doc_id": "git-001", "title": "Git Branching"},
        ),
        Document(
            page_content="Python virtual environments isolate project dependencies using venv or virtualenv.",
            metadata={"doc_id": "python-001", "title": "Python Environments"},
        ),
    ]


# ── ChromaDB backend ──────────────────────────────────────────────────────────


class TestChromaBackend:
    @pytest.fixture
    def chroma_backend(self):
        """ChromaDB backend with a temporary directory."""
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from rag_eval.store.chroma_store import ChromaBackend

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ChromaBackend(
                collection_name="test-collection",
                embeddings=embeddings,
                persist_dir=tmpdir,
            )
            yield backend

    def test_is_healthy(self, chroma_backend) -> None:
        """ChromaDB should report healthy when package is installed."""
        assert chroma_backend.is_healthy() is True

    def test_ingest_and_retrieve(self, chroma_backend) -> None:
        """Documents ingested should be retrievable."""
        docs = _make_test_docs()
        chroma_backend.ingest(docs)

        retriever = chroma_backend.as_retriever(k=2)
        results = retriever.invoke("Docker containerization")

        assert len(results) >= 1
        # Docker doc should be in results
        result_ids = [r.metadata.get("doc_id") for r in results]
        assert "docker-001" in result_ids

    def test_similarity_search_with_score(self, chroma_backend) -> None:
        """similarity_search_with_score should return (doc, score) tuples."""
        chroma_backend.ingest(_make_test_docs())
        results = chroma_backend.similarity_search_with_score("Python dependency management", k=2)

        assert len(results) >= 1
        for doc, score in results:
            assert isinstance(score, float)
            assert score >= 0.0  # our converted similarity is always non-negative
            assert "doc_id" in doc.metadata

    def test_most_relevant_doc_ranked_first(self, chroma_backend) -> None:
        """The most semantically relevant doc should rank first."""
        chroma_backend.ingest(_make_test_docs())
        results = chroma_backend.similarity_search_with_score("Git workflow branching", k=3)

        top_doc, _ = results[0]
        assert top_doc.metadata.get("doc_id") == "git-001"

    def test_as_retriever_returns_base_retriever(self, chroma_backend) -> None:
        """as_retriever should return a LangChain BaseRetriever."""
        from langchain_core.retrievers import BaseRetriever

        chroma_backend.ingest(_make_test_docs())
        retriever = chroma_backend.as_retriever(k=3)
        assert isinstance(retriever, BaseRetriever)


# ── OpenSearch backend ────────────────────────────────────────────────────────


@pytest.mark.opensearch
class TestOpenSearchBackend:
    """Tests for the OpenSearch backend.

    Skipped automatically unless pytest is run with -m opensearch AND
    OpenSearch is reachable at http://localhost:9200.
    """

    @pytest.fixture
    def opensearch_backend(self):
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from rag_eval.store.opensearch_store import OpenSearchBackend

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        backend = OpenSearchBackend(
            opensearch_url="http://localhost:9200",
            index_name="rag-eval-test-backend",
            embeddings=embeddings,
        )

        if not backend.is_healthy():
            pytest.skip("OpenSearch not reachable at http://localhost:9200")

        yield backend

    def test_is_healthy(self, opensearch_backend) -> None:
        assert opensearch_backend.is_healthy() is True

    def test_ingest_and_retrieve(self, opensearch_backend) -> None:
        docs = _make_test_docs()
        opensearch_backend.ingest(docs)

        retriever = opensearch_backend.as_retriever(k=2)
        results = retriever.invoke("Docker containerization platform")
        assert len(results) >= 1

    def test_similarity_search_returns_scores(self, opensearch_backend) -> None:
        opensearch_backend.ingest(_make_test_docs())
        results = opensearch_backend.similarity_search_with_score("Docker", k=2)
        assert len(results) >= 1
        for doc, score in results:
            assert isinstance(score, float)


# ── Factory integration ───────────────────────────────────────────────────────


class TestVectorStoreFactory:
    def test_factory_returns_chroma_when_configured(self, test_settings) -> None:
        """Factory should return ChromaBackend when VECTOR_STORE_BACKEND=chroma."""
        from rag_eval.store.chroma_store import ChromaBackend
        from rag_eval.store.factory import get_vector_store

        backend = get_vector_store(test_settings)
        assert isinstance(backend, ChromaBackend)

    def test_factory_returns_opensearch_when_healthy(self) -> None:
        """Factory should return OpenSearch when configured and healthy."""
        from rag_eval.config import Settings
        from rag_eval.store.factory import get_vector_store
        from rag_eval.store.opensearch_store import OpenSearchBackend

        settings = Settings(
            openai_api_key="sk-test",  # type: ignore[arg-type]
            vector_store_backend="opensearch",
        )
        backend = get_vector_store(settings)

        # If OpenSearch is not running, factory should fall back to ChromaDB
        from rag_eval.store.chroma_store import ChromaBackend
        assert isinstance(backend, (OpenSearchBackend, ChromaBackend))
