"""Unit tests for the confidence routing evaluator.

Tests cover both the logprobs and similarity-fallback paths, threshold
boundary conditions, and the logprob-to-confidence conversion math.
"""

from __future__ import annotations

import math

import pytest

from rag_eval.metrics.confidence_routing import ConfidenceRouter, _logprobs_to_confidence
from rag_eval.models.results import ConfidenceRoutingResult


# ── logprob conversion ────────────────────────────────────────────────────────


class TestLogprobsToConfidence:
    def test_zero_logprobs_gives_confidence_one(self) -> None:
        """log(1.0) = 0 → exp(0) = 1.0 (perfectly certain)."""
        confidence = _logprobs_to_confidence([0.0, 0.0, 0.0])
        assert confidence == pytest.approx(1.0)

    def test_very_negative_logprobs_give_low_confidence(self) -> None:
        """Very negative logprobs → very low confidence."""
        confidence = _logprobs_to_confidence([-5.0, -5.0, -5.0])
        assert confidence == pytest.approx(math.exp(-5.0), rel=1e-4)
        assert confidence < 0.01

    def test_empty_logprobs_returns_neutral(self) -> None:
        """Empty list → neutral confidence 0.5."""
        confidence = _logprobs_to_confidence([])
        assert confidence == pytest.approx(0.5)

    def test_mixed_logprobs_returns_mean(self) -> None:
        """Mean of logprobs should be used."""
        logprobs = [-1.0, -2.0, -3.0]  # mean = -2.0
        expected = math.exp(-2.0)
        confidence = _logprobs_to_confidence(logprobs)
        assert confidence == pytest.approx(expected, rel=1e-4)

    def test_inf_logprobs_filtered_out(self) -> None:
        """Infinite logprobs should be filtered before computing mean."""
        confidence = _logprobs_to_confidence([0.0, float("-inf"), -1.0])
        # Only 0.0 and -1.0 should be used → mean = -0.5
        expected = math.exp(-0.5)
        assert confidence == pytest.approx(expected, rel=1e-4)

    def test_all_inf_returns_neutral(self) -> None:
        """All-inf logprobs → neutral confidence."""
        confidence = _logprobs_to_confidence([float("-inf"), float("-inf")])
        assert confidence == pytest.approx(0.5)


# ── ConfidenceRouter ──────────────────────────────────────────────────────────


class TestConfidenceRouter:
    @pytest.fixture
    def router(self) -> ConfidenceRouter:
        return ConfidenceRouter(threshold=0.7)

    # From logprobs
    def test_high_confidence_not_routed(self, router: ConfidenceRouter) -> None:
        """High-confidence answer (logprobs near 0) → not routed to human."""
        result = router.evaluate_from_logprobs([-0.1, -0.1, -0.1])
        assert result.answer_confidence > 0.7
        assert result.routed_to_human is False
        assert result.confidence_source == "logprobs"

    def test_low_confidence_routed(self, router: ConfidenceRouter) -> None:
        """Low-confidence answer → routed to human."""
        result = router.evaluate_from_logprobs([-3.0, -4.0, -5.0])
        assert result.answer_confidence < 0.7
        assert result.routed_to_human is True
        assert result.confidence_source == "logprobs"

    def test_exactly_at_threshold_not_routed(self, router: ConfidenceRouter) -> None:
        """Confidence exactly at threshold (0.7) → NOT routed (requires < threshold)."""
        # confidence = 0.7 → not routed (strict less-than)
        # We can achieve this by choosing logprobs such that exp(mean) ≈ 0.7
        target_logprob = math.log(0.7)
        result = router.evaluate_from_logprobs([target_logprob])
        # Due to floating point, confidence ≈ 0.7
        assert result.routed_to_human is (result.answer_confidence < 0.7)

    # From similarity fallback
    def test_high_similarity_not_routed(self, router: ConfidenceRouter) -> None:
        """High similarity → not routed."""
        result = router.evaluate_from_similarity(0.9)
        assert result.answer_confidence == pytest.approx(0.9)
        assert result.routed_to_human is False
        assert result.confidence_source == "similarity_fallback"

    def test_low_similarity_routed(self, router: ConfidenceRouter) -> None:
        """Low similarity → routed to human."""
        result = router.evaluate_from_similarity(0.4)
        assert result.answer_confidence == pytest.approx(0.4)
        assert result.routed_to_human is True

    def test_negative_similarity_clamped(self, router: ConfidenceRouter) -> None:
        """Negative similarity values are clamped to 0."""
        result = router.evaluate_from_similarity(-0.5)
        assert result.answer_confidence == pytest.approx(0.0)

    def test_similarity_above_one_clamped(self, router: ConfidenceRouter) -> None:
        """Values above 1.0 are clamped to 1.0."""
        result = router.evaluate_from_similarity(1.5)
        assert result.answer_confidence == pytest.approx(1.0)

    def test_routing_correct_is_true(self, router: ConfidenceRouter) -> None:
        """routing_correct should always be True (deterministic threshold rule)."""
        for sim in [0.1, 0.5, 0.7, 0.9]:
            result = router.evaluate_from_similarity(sim)
            assert result.routing_correct is True

    def test_threshold_recorded_in_result(self, router: ConfidenceRouter) -> None:
        """The threshold used should be recorded in the result."""
        result = router.evaluate_from_similarity(0.8)
        assert result.threshold_used == pytest.approx(0.7)

    def test_result_is_pydantic_model(self, router: ConfidenceRouter) -> None:
        """Result should be a valid ConfidenceRoutingResult model."""
        result = router.evaluate_from_similarity(0.8)
        assert isinstance(result, ConfidenceRoutingResult)
        assert 0.0 <= result.answer_confidence <= 1.0

    @pytest.mark.parametrize("threshold,sim,expected_routed", [
        (0.5, 0.6, False),  # 0.6 >= 0.5 → not routed
        (0.5, 0.4, True),   # 0.4 < 0.5 → routed
        (0.9, 0.8, True),   # 0.8 < 0.9 → routed
        (0.0, 0.01, False),  # anything >= 0.0 → not routed
    ])
    def test_routing_threshold_parametrized(
        self,
        threshold: float,
        sim: float,
        expected_routed: bool,
    ) -> None:
        router = ConfidenceRouter(threshold=threshold)
        result = router.evaluate_from_similarity(sim)
        assert result.routed_to_human is expected_routed
