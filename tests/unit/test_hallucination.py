"""Unit tests for the hallucination evaluator (LLM-as-judge).

All OpenAI API calls are mocked — these tests verify:
1. The judge prompt is constructed correctly
2. JSON parsing works for valid and malformed responses
3. Error handling falls back gracefully
4. HallucinationResult fields are populated correctly

Note on mocking strategy: ChatOpenAI is a Pydantic model in langchain-openai,
so patch.object cannot set attributes on it directly. We mock at the class level
using patch() on the module path, or mock the invoke method via spec=ChatOpenAI.
The cleanest approach is to patch the HallucinationEvaluator's _llm attribute
by replacing it with a MagicMock after construction.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from rag_eval.metrics.hallucination import HallucinationEvaluator, _SYSTEM_PROMPT
from rag_eval.models.results import HallucinationResult


@pytest.fixture
def evaluator() -> HallucinationEvaluator:
    """HallucinationEvaluator with a test API key."""
    with patch("rag_eval.metrics.hallucination.get_chat_model"), \
         patch("rag_eval.metrics.hallucination.get_settings") as ms:
        ms.return_value = MagicMock(
            judge_model="gpt-3.5-turbo",
            openai_api_key=MagicMock(get_secret_value=lambda: "sk-test"),
            llm_provider="openai",
            groq_api_key=None,
        )
        ev = HallucinationEvaluator(model_name="gpt-3.5-turbo", api_key="sk-test-key")
    # Replace _llm with a MagicMock to avoid Pydantic model attribute restrictions.
    # patch.object doesn't work on Pydantic model instances; direct assignment via
    # object.__setattr__ bypasses Pydantic's field protection for testing.
    mock_llm = MagicMock()
    object.__setattr__(ev, "_llm", mock_llm)
    return ev


def _make_mock_response(content: str) -> MagicMock:
    """Helper: create a mock LLM response with the given content string."""
    mock_response = MagicMock()
    mock_response.content = content
    return mock_response


def _set_mock_response(evaluator: HallucinationEvaluator, content: str) -> None:
    """Configure the evaluator's mock LLM to return the given content."""
    evaluator._llm.invoke.return_value = _make_mock_response(content)


def _set_mock_error(evaluator: HallucinationEvaluator, exc: Exception) -> None:
    """Configure the evaluator's mock LLM to raise an exception."""
    evaluator._llm.invoke.side_effect = exc


class TestHallucinationEvaluator:
    def test_returns_no_hallucination_for_supported_answer(
        self, evaluator: HallucinationEvaluator
    ) -> None:
        """Clean JSON response → correct HallucinationResult."""
        _set_mock_response(evaluator, json.dumps({
            "is_hallucination": False,
            "reasoning": "All claims are supported by the provided context.",
            "confidence": 0.95,
        }))
        result = evaluator.evaluate(
            query="What is Docker?",
            context="Docker is a containerization platform.",
            generated_answer="Docker is a containerization platform.",
        )

        assert isinstance(result, HallucinationResult)
        assert result.is_hallucination is False
        assert result.judge_confidence == pytest.approx(0.95)
        assert "supported" in result.judge_reasoning.lower()

    def test_returns_hallucination_for_unsupported_answer(
        self, evaluator: HallucinationEvaluator
    ) -> None:
        """Hallucination verdict → is_hallucination=True."""
        _set_mock_response(evaluator, json.dumps({
            "is_hallucination": True,
            "reasoning": "The answer mentions Python 3.15 which is not in the context.",
            "confidence": 0.88,
        }))
        result = evaluator.evaluate(
            query="What Python version?",
            context="Python 3.11 is the current stable release.",
            generated_answer="Python 3.15 was released in 2025.",
        )

        assert result.is_hallucination is True
        assert result.judge_confidence == pytest.approx(0.88)

    def test_handles_malformed_json_gracefully(
        self, evaluator: HallucinationEvaluator
    ) -> None:
        """Malformed JSON → fail-safe default (no hallucination, confidence=0)."""
        _set_mock_response(evaluator, "This is not JSON!")
        result = evaluator.evaluate(
            query="test", context="test context", generated_answer="test answer"
        )

        # Fail-safe: don't block valid answers on parse error
        assert result.is_hallucination is False
        assert result.judge_confidence == pytest.approx(0.0)
        assert "JSON parse error" in result.judge_reasoning

    def test_handles_api_exception_gracefully(
        self, evaluator: HallucinationEvaluator
    ) -> None:
        """API exception → fail-safe default, no exception propagation."""
        _set_mock_error(evaluator, Exception("Connection timeout"))
        result = evaluator.evaluate(
            query="test", context="test context", generated_answer="test answer"
        )

        assert result.is_hallucination is False
        assert result.judge_confidence == pytest.approx(0.0)
        assert "Exception" in result.judge_reasoning

    def test_confidence_clamped_to_valid_range(
        self, evaluator: HallucinationEvaluator
    ) -> None:
        """Confidence values outside [0,1] are clamped."""
        _set_mock_response(evaluator, json.dumps({
            "is_hallucination": False,
            "reasoning": "Looks good.",
            "confidence": 1.5,  # out of range
        }))
        result = evaluator.evaluate(
            query="test", context="context", generated_answer="answer"
        )

        assert 0.0 <= result.judge_confidence <= 1.0

    def test_judge_model_recorded_in_result(
        self, evaluator: HallucinationEvaluator
    ) -> None:
        """The judge model name should be recorded in the result."""
        _set_mock_response(evaluator, json.dumps({
            "is_hallucination": False,
            "reasoning": "Fine.",
            "confidence": 0.8,
        }))
        result = evaluator.evaluate("q", "ctx", "answer")
        assert result.judge_model == "gpt-3.5-turbo"

    def test_system_prompt_mentions_hallucination(self) -> None:
        """The system prompt should clearly define the task."""
        assert "hallucination" in _SYSTEM_PROMPT.lower()
        assert "json" in _SYSTEM_PROMPT.lower()

    def test_context_truncated_for_long_input(
        self, evaluator: HallucinationEvaluator
    ) -> None:
        """Very long context should not cause an error (truncated to 4000 chars)."""
        long_context = "This is context. " * 500  # ~9000 chars
        _set_mock_response(evaluator, json.dumps({
            "is_hallucination": False,
            "reasoning": "OK",
            "confidence": 0.9,
        }))
        result = evaluator.evaluate("query", long_context, "answer")
        assert result is not None
        # Verify invoke was called (context was processed)
        evaluator._llm.invoke.assert_called_once()
