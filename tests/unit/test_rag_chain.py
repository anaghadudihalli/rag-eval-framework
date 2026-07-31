"""Unit tests for the refactored RAGChain.

Verifies that logprobs are extracted directly from AIMessage.response_metadata
(single-pass) and that no second "repeat verbatim" API call is made.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from rag_eval.pipeline.rag_chain import RAGChain, _build_messages, _format_context


# ── Helper builders ──────────────────────────────────────────────────────────


def _make_ai_message(content: str, logprobs: list[dict] | None = None) -> AIMessage:
    """Build a mock AIMessage with optional logprob metadata."""
    metadata: dict[str, Any] = {}
    if logprobs is not None:
        metadata["logprobs"] = {"content": logprobs}
    return AIMessage(content=content, response_metadata=metadata)


def _make_rag_chain(llm_response: AIMessage, docs: list[Document]) -> RAGChain:
    """Build a RAGChain with a mocked retriever and mocked LLM."""
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = docs
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = llm_response

    with patch("rag_eval.pipeline.rag_chain.get_settings") as ms, \
         patch("rag_eval.llm_factory.get_chat_model", return_value=mock_llm):
        ms.return_value = MagicMock(
            openai_api_key=MagicMock(get_secret_value=lambda: "sk-test"),
            llm_provider="openai",
        )
        chain = RAGChain(retriever=mock_retriever, api_key="sk-test")
        chain._llm = mock_llm
        chain._retriever = mock_retriever

    return chain


# ── Module-level utility tests ───────────────────────────────────────────────


class TestFormatContext:
    def test_single_doc(self) -> None:
        docs = [Document(page_content="Hello world.")]
        assert _format_context(docs) == "Hello world."

    def test_multiple_docs_joined_with_double_newline(self) -> None:
        docs = [
            Document(page_content="First chunk."),
            Document(page_content="Second chunk."),
        ]
        assert _format_context(docs) == "First chunk.\n\nSecond chunk."

    def test_empty_docs_returns_empty_string(self) -> None:
        assert _format_context([]) == ""


class TestBuildMessages:
    def test_returns_system_and_human_message(self) -> None:
        from langchain_core.messages import HumanMessage, SystemMessage

        msgs = _build_messages("What is Docker?", "Docker is a container platform.")
        assert len(msgs) == 2
        assert isinstance(msgs[0], SystemMessage)
        assert isinstance(msgs[1], HumanMessage)

    def test_context_embedded_in_system_message(self) -> None:
        context = "Docker is a container platform."
        msgs = _build_messages("What is Docker?", context)
        assert context in msgs[0].content

    def test_query_in_human_message(self) -> None:
        query = "What is Docker?"
        msgs = _build_messages(query, "some context")
        assert msgs[1].content == query


# ── RAGChain._extract_logprobs tests ─────────────────────────────────────────


class TestExtractLogprobs:
    """Tests for the single-pass logprob extraction from AIMessage."""

    def _chain(self) -> RAGChain:
        """Minimal RAGChain instance with mocked internals."""
        with patch("rag_eval.pipeline.rag_chain.get_settings") as ms, \
             patch("rag_eval.llm_factory.get_chat_model"):
            ms.return_value = MagicMock(
                openai_api_key=MagicMock(get_secret_value=lambda: "sk-test"),
                llm_provider="openai",
            )
            return RAGChain(retriever=MagicMock(), api_key="sk-test")

    def test_extracts_logprobs_from_response_metadata(self) -> None:
        chain = self._chain()
        message = _make_ai_message(
            content="Docker uses containers.",
            logprobs=[
                {"token": "Docker", "logprob": -0.1},
                {"token": " uses", "logprob": -0.2},
                {"token": " containers", "logprob": -0.05},
                {"token": ".", "logprob": -0.01},
            ],
        )
        result = chain._extract_logprobs(message)
        assert result == pytest.approx([-0.1, -0.2, -0.05, -0.01])

    def test_returns_empty_list_when_no_response_metadata(self) -> None:
        chain = self._chain()
        message = AIMessage(content="No metadata here.")
        result = chain._extract_logprobs(message)
        assert result == []

    def test_returns_empty_list_when_logprobs_key_missing(self) -> None:
        chain = self._chain()
        message = AIMessage(content="Answer.", response_metadata={"model": "gpt-3.5-turbo"})
        result = chain._extract_logprobs(message)
        assert result == []

    def test_returns_empty_list_when_content_key_missing(self) -> None:
        chain = self._chain()
        message = AIMessage(
            content="Answer.", response_metadata={"logprobs": {"no_content_key": []}}
        )
        result = chain._extract_logprobs(message)
        assert result == []

    def test_filters_out_none_logprob_entries(self) -> None:
        chain = self._chain()
        message = _make_ai_message(
            content="Answer.",
            logprobs=[
                {"token": "Answer", "logprob": -0.1},
                {"token": ".", "logprob": None},  # should be skipped
                {"token": "!", "logprob": -0.05},
            ],
        )
        result = chain._extract_logprobs(message)
        assert result == pytest.approx([-0.1, -0.05])

    def test_handles_empty_content_list(self) -> None:
        chain = self._chain()
        message = _make_ai_message(content="Answer.", logprobs=[])
        result = chain._extract_logprobs(message)
        assert result == []


# ── RAGChain.run integration (mocked LLM + retriever) ────────────────────────


class TestRAGChainRun:
    """Tests for the full RAGChain.run() call with mocked LLM and retriever."""

    def _run_chain(
        self,
        answer: str = "Docker uses OS-level virtualization.",
        logprobs: list[dict] | None = None,
        docs: list[Document] | None = None,
    ) -> tuple:
        """Helper: builds chain, runs it, returns the 4-tuple."""
        if docs is None:
            docs = [
                Document(page_content="Docker is a platform.", metadata={"doc_id": "docker-001"}),
                Document(page_content="Containers share the OS.", metadata={"doc_id": "docker-002"}),
            ]
        ai_message = _make_ai_message(content=answer, logprobs=logprobs)
        chain = _make_rag_chain(ai_message, docs)
        return chain.run("What is Docker?")

    def test_run_returns_four_tuple(self) -> None:
        result = self._run_chain()
        assert len(result) == 4

    def test_answer_is_string(self) -> None:
        answer_text = "Docker uses containers."
        result_answer, _, _, _ = self._run_chain(answer=answer_text)
        assert result_answer == answer_text

    def test_docs_are_returned(self) -> None:
        docs = [Document(page_content="Doc content.", metadata={"doc_id": "d-001"})]
        _, result_docs, _, _ = self._run_chain(docs=docs)
        assert len(result_docs) == 1
        assert result_docs[0].page_content == "Doc content."

    def test_llm_called_exactly_once(self) -> None:
        """The LLM must be called exactly ONCE — no second call for logprobs."""
        ai_message = _make_ai_message(
            content="Answer.",
            logprobs=[{"token": "Answer", "logprob": -0.15}],
        )
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = ai_message

        with patch("rag_eval.pipeline.rag_chain.get_settings") as ms, \
             patch("rag_eval.llm_factory.get_chat_model", return_value=mock_llm):
            ms.return_value = MagicMock(
                openai_api_key=MagicMock(get_secret_value=lambda: "sk-test"),
                llm_provider="openai",
            )
            chain = RAGChain(retriever=mock_retriever, api_key="sk-test")
            chain._llm = mock_llm
            chain._retriever = mock_retriever
            _, _, logprobs, _ = chain.run("Test query")

        # Critical: exactly one LLM call — the original generation
        assert mock_llm.invoke.call_count == 1
        assert logprobs == pytest.approx([-0.15])

    def test_logprobs_empty_when_no_metadata(self) -> None:
        _, _, logprobs, _ = self._run_chain(logprobs=None)
        assert logprobs == []

    def test_latency_ms_is_non_negative_float(self) -> None:
        _, _, _, latency_ms = self._run_chain()
        assert isinstance(latency_ms, float)
        assert latency_ms >= 0.0

    def test_retriever_called_with_query(self) -> None:
        query = "What is Docker?"
        ai_message = _make_ai_message(content="Answer.")
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = ai_message

        with patch("rag_eval.pipeline.rag_chain.get_settings") as ms, \
             patch("rag_eval.llm_factory.get_chat_model", return_value=mock_llm):
            ms.return_value = MagicMock(
                openai_api_key=MagicMock(get_secret_value=lambda: "sk-test"),
                llm_provider="openai",
            )
            chain = RAGChain(retriever=mock_retriever, api_key="sk-test")
            chain._llm = mock_llm
            chain._retriever = mock_retriever
            chain.run(query)

        mock_retriever.invoke.assert_called_once_with(query)
