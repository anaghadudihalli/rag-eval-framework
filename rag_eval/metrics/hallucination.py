"""LLM-as-judge hallucination evaluator.

Uses GPT-3.5-turbo to evaluate whether a generated answer introduces
claims not supported by the retrieved context. This is the "faithfulness"
metric from RAGAS, implemented here from first principles.

Pattern:
    Input:  query + retrieved_context + generated_answer
    Output: {"is_hallucination": bool, "reasoning": str, "confidence": float}

The judge is instructed to output only valid JSON so we can parse
it deterministically. We use response_format=json_object to enforce this.

Design note: GPT-3.5-turbo is used (not GPT-4) for cost efficiency in
CI/CD pipelines. The judge prompt is designed to be conservative — when
in doubt, it returns is_hallucination=false. False negatives (missed
hallucinations) are less harmful than false positives (blocking valid answers).
"""

from __future__ import annotations

import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from rag_eval.config import get_settings
from rag_eval.llm_factory import get_chat_model
from rag_eval.models.results import HallucinationResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a factual grounding evaluator for a RAG (Retrieval-Augmented Generation) system.

Your task: Determine if a generated answer contains claims that are NOT supported by the provided context.

Rules:
1. ONLY flag as hallucination if the answer makes a specific factual claim that contradicts or is absent from the context.
2. Do NOT flag as hallucination if the answer simply rephrases or summarizes the context.
3. Do NOT flag as hallucination if the answer appropriately says it doesn't know or qualifies its uncertainty.
4. Minor stylistic differences or paraphrasing are acceptable.

Return ONLY a JSON object — no markdown, no explanation outside the JSON:
{"is_hallucination": true/false, "reasoning": "brief explanation", "confidence": 0.0-1.0}

Where confidence reflects how certain you are about your verdict (1.0 = completely certain)."""

_USER_PROMPT_TEMPLATE = """Query: {query}

Retrieved Context:
{context}

Generated Answer:
{answer}

Evaluate the answer and return your verdict as JSON."""


class HallucinationEvaluator:
    """Evaluates hallucination using GPT-3.5-turbo as a judge.

    Args:
        model_name: OpenAI model to use as judge (default: settings.judge_model).
        api_key: OpenAI API key (default: settings.openai_api_key).
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
    ) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.judge_model
        self._api_key = api_key or settings.openai_api_key.get_secret_value()

        # Use the LLM factory so this works with both OpenAI and Groq.
        # response_format=json_object enforces JSON on OpenAI; Groq relies on
        # the prompt's instruction ("Return ONLY a JSON object") instead.
        base_llm = get_chat_model(settings=settings, temperature=0.0, max_tokens=256)

        if settings.llm_provider == "openai":
            # Rebind with JSON mode for OpenAI (gpt-3.5-turbo-1106+)
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self._model_name,
                api_key=self._api_key,
                temperature=0.0,
                max_tokens=256,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
        else:
            # Groq: no response_format support; the system prompt's JSON
            # instruction is sufficient for llama-3.1 models.
            self._llm = base_llm

    def evaluate(
        self,
        query: str,
        context: str,
        generated_answer: str,
    ) -> HallucinationResult:
        """Ask the LLM judge to evaluate faithfulness.

        Args:
            query: The original user query.
            context: The concatenated retrieved documents (used as evidence).
            generated_answer: The RAG system's output to evaluate.

        Returns:
            HallucinationResult with verdict, reasoning, and judge confidence.
        """
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            query=query,
            context=context[:4000],  # guard against context window overflow
            answer=generated_answer,
        )

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self._llm.invoke(messages)
            raw_content = response.content

            verdict = json.loads(raw_content)
            is_hallucination = bool(verdict.get("is_hallucination", False))
            reasoning = str(verdict.get("reasoning", ""))
            confidence = float(verdict.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))  # clamp to [0, 1]

        except json.JSONDecodeError as exc:
            # If the model returns malformed JSON despite response_format,
            # fail safe: don't flag as hallucination, log for review.
            logger.warning(
                "Failed to parse hallucination judge response: %s | raw='%s'",
                exc,
                raw_content[:200] if "raw_content" in dir() else "N/A",
            )
            is_hallucination = False
            reasoning = "JSON parse error — verdict defaulted to no hallucination."
            confidence = 0.0

        except Exception as exc:
            logger.error("Hallucination evaluation failed: %s", exc)
            is_hallucination = False
            reasoning = f"Evaluation error: {type(exc).__name__}"
            confidence = 0.0

        logger.debug(
            "Hallucination — is_hallucination=%s, confidence=%.2f | query='%s...'",
            is_hallucination,
            confidence,
            query[:60],
        )

        return HallucinationResult(
            is_hallucination=is_hallucination,
            judge_reasoning=reasoning,
            judge_confidence=confidence,
            judge_model=self._model_name,
        )
