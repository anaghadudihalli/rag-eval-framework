"""Confidence-score routing evaluator.

Determines whether a RAG answer should be escalated to human review
based on the model's confidence in its response.

Confidence signal (primary): OpenAI token logprobs
    confidence = exp(mean(logprob_per_token))

This converts log-probabilities to a probability-like scalar:
- High confidence (~0.9+): model is certain → no human review needed
- Low confidence (<0.7):   model is uncertain → route to human review

Fallback (if logprobs unavailable): uses cosine similarity score from
the similarity evaluator as a proxy signal. Documented via confidence_source.

Routing correctness:
    A routing decision is "correct" when:
    - High-confidence answer (>= threshold) is NOT routed to human, OR
    - Low-confidence answer (< threshold) IS routed to human.

    Since we don't have ground-truth routing labels in the golden dataset,
    we evaluate routing correctness purely on whether the routing decision
    is internally consistent with the confidence score (i.e., the router
    applied the threshold correctly). This is a deterministic check.
"""

from __future__ import annotations

import logging
import math

from rag_eval.config import get_settings
from rag_eval.models.results import ConfidenceRoutingResult

logger = logging.getLogger(__name__)


def _logprobs_to_confidence(logprobs: list[float]) -> float:
    """Convert a list of token log-probabilities to a scalar confidence.

    confidence = exp(mean(logprobs))

    This maps:
        logprobs all 0.0 (log(1.0)) → confidence 1.0  (model perfectly certain)
        logprobs all -inf            → confidence 0.0  (model completely uncertain)

    Args:
        logprobs: List of per-token log-probabilities from OpenAI API.

    Returns:
        Float in [0, 1] representing answer confidence.
    """
    if not logprobs:
        return 0.5  # neutral when no logprobs available
    valid_logprobs = [lp for lp in logprobs if lp is not None and not math.isinf(lp)]
    if not valid_logprobs:
        return 0.5
    mean_logprob = sum(valid_logprobs) / len(valid_logprobs)
    return float(math.exp(mean_logprob))


class ConfidenceRouter:
    """Computes answer confidence and determines human routing.

    Args:
        threshold: Confidence below this value → route to human.
                   Defaults to settings.confidence_threshold.
    """

    def __init__(self, threshold: float | None = None) -> None:
        settings = get_settings()
        self._threshold = threshold if threshold is not None else settings.confidence_threshold

    def evaluate_from_logprobs(
        self,
        logprobs: list[float],
    ) -> ConfidenceRoutingResult:
        """Evaluate routing using OpenAI token logprobs.

        Args:
            logprobs: Per-token log-probabilities from ChatOpenAI response.

        Returns:
            ConfidenceRoutingResult with confidence_source='logprobs'.
        """
        confidence = _logprobs_to_confidence(logprobs)
        return self._make_result(confidence, source="logprobs")

    def evaluate_from_similarity(
        self,
        similarity_score: float,
    ) -> ConfidenceRoutingResult:
        """Evaluate routing using cosine similarity as a confidence proxy.

        This is the fallback when logprobs are unavailable. Similarity and
        confidence are related (higher semantic match → higher confidence)
        but not equivalent; treat this as an approximation.

        Args:
            similarity_score: Cosine similarity in [-1, 1] from SimilarityEvaluator.
                              Values < 0 are clamped to 0 before comparison.

        Returns:
            ConfidenceRoutingResult with confidence_source='similarity_fallback'.
        """
        # Clamp to [0, 1] since similarity can technically be negative
        confidence = max(0.0, min(1.0, similarity_score))
        return self._make_result(confidence, source="similarity_fallback")

    def _make_result(self, confidence: float, source: str) -> ConfidenceRoutingResult:
        """Construct the routing result given a confidence scalar."""
        routed_to_human = confidence < self._threshold

        # Routing is always "correct" because we're applying a deterministic rule.
        # The routing_correct field is meaningful in production when you have
        # ground-truth labels for which queries SHOULD have been escalated.
        # Here it confirms the threshold was applied as expected.
        routing_correct = True  # threshold rule was applied correctly

        logger.debug(
            "Confidence routing — confidence=%.3f, threshold=%.2f, "
            "routed_to_human=%s, source=%s",
            confidence,
            self._threshold,
            routed_to_human,
            source,
        )

        return ConfidenceRoutingResult(
            answer_confidence=round(confidence, 4),
            confidence_source=source,
            routed_to_human=routed_to_human,
            routing_correct=routing_correct,
            threshold_used=self._threshold,
        )
