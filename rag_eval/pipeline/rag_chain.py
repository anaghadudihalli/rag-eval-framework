"""RAG chain — retrieval-augmented generation pipeline.

Retrieves relevant document chunks from the vector store, then generates
an answer using ChatOpenAI with logprobs enabled.

Design note: We use a manual LCEL-style pipeline (retrieve → format context
→ build prompt → llm.invoke) rather than the opaque create_retrieval_chain
wrapper. This gives us direct access to the AIMessage returned by the LLM,
whose response_metadata["logprobs"] carries per-token log-probabilities.
Extracting logprobs from the AIMessage avoids the previous approach of making
a second "repeat verbatim" API call, halving the token cost per sample.
"""

from __future__ import annotations

import logging
import time

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever

from rag_eval.config import get_settings
from rag_eval.llm_factory import get_chat_model

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the provided context.\n"
    "If the context does not contain enough information to answer the question, say so clearly.\n"
    "Be concise and factual. Do not introduce information not present in the context."
)


def _format_context(docs: list[Document]) -> str:
    """Concatenate retrieved document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def _build_messages(query: str, context: str) -> list[BaseMessage]:
    """Build the prompt message list for the LLM."""
    system_content = f"{_SYSTEM_PROMPT}\n\nContext:\n{context}"
    return [
        SystemMessage(content=system_content),
        HumanMessage(content=query),
    ]


class RAGChain:
    """Retrieval-Augmented Generation chain with single-pass logprob extraction.

    Uses a manual LCEL-style pipeline so the AIMessage from the LLM is directly
    accessible, allowing logprobs to be read from response_metadata without
    any additional API calls.

    Args:
        retriever: A LangChain BaseRetriever (from VectorStoreBackend.as_retriever).
        api_key: OpenAI API key.
        model_name: OpenAI model for generation.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        api_key: str | None = None,
        model_name: str = "gpt-3.5-turbo",
    ) -> None:
        settings = get_settings()
        self._retriever = retriever
        self._model_name = model_name
        self._api_key = api_key or settings.openai_api_key.get_secret_value()

        # logprobs=True enables per-token log-probabilities for confidence routing.
        # Only supported by OpenAI — Groq silently returns empty logprobs, which
        # the confidence router handles by falling back to cosine similarity.
        self._llm = get_chat_model(
            settings=get_settings(),
            temperature=0.0,
            max_tokens=512,
            with_logprobs=True,  # no-op for Groq
        )

    def run(self, query: str) -> tuple[str, list[Document], list[float], float]:
        """Execute the RAG pipeline for a single query.

        Pipeline steps (all in one pass):
            1. Retrieve relevant document chunks from the vector store.
            2. Format the chunks into a context string.
            3. Build a prompt (SystemMessage + HumanMessage).
            4. Call the LLM — returns an AIMessage whose response_metadata
               carries logprobs with zero additional API cost.
            5. Extract the answer string and logprobs from the AIMessage.

        Args:
            query: The user's natural-language question.

        Returns:
            Tuple of:
                - answer (str): The generated answer.
                - docs (list[Document]): Retrieved source documents.
                - logprobs (list[float]): Per-token log-probs (may be empty if
                  unavailable, e.g. when the model doesn't support logprobs).
                - latency_ms (float): End-to-end wall-clock time in milliseconds.
        """
        start = time.perf_counter()

        # Step 1: Retrieve
        docs: list[Document] = self._retriever.invoke(query)

        # Step 2 & 3: Format context and build prompt
        context = _format_context(docs)
        messages = _build_messages(query, context)

        # Step 4: Single LLM call — AIMessage includes logprobs in response_metadata
        ai_message: AIMessage = self._llm.invoke(messages)  # type: ignore[assignment]

        latency_ms = (time.perf_counter() - start) * 1000

        # Step 5: Extract answer and logprobs from the AIMessage directly
        answer: str = ai_message.content if isinstance(ai_message.content, str) else ""
        logprobs = self._extract_logprobs(ai_message)

        logger.debug(
            "RAG run — query='%s...' | answer='%s...' | docs=%d | latency=%.1fms",
            query[:60],
            answer[:60],
            len(docs),
            latency_ms,
        )

        return answer, docs, logprobs, latency_ms

    def _extract_logprobs(self, message: AIMessage) -> list[float]:
        """Extract per-token log-probabilities from the AIMessage.

        OpenAI returns logprobs in response_metadata["logprobs"]["content"],
        where each entry is a dict with at least a "logprob" float key.

        Returns an empty list if logprobs are absent or malformed — the
        confidence router will fall back to cosine similarity in that case.

        Args:
            message: The AIMessage returned by ChatOpenAI.invoke().

        Returns:
            List of per-token log-probability floats (negative values).
        """
        try:
            logprob_data = (message.response_metadata or {}).get("logprobs", {})
            if logprob_data and "content" in logprob_data:
                return [
                    token["logprob"]
                    for token in logprob_data["content"]
                    if token.get("logprob") is not None
                ]
        except Exception as exc:
            logger.debug("Could not extract logprobs: %s — will use similarity fallback.", exc)

        return []
