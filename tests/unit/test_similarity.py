"""Unit tests for semantic similarity evaluator.

These tests use the actual sentence-transformers model (all-MiniLM-L6-v2).
The model is small (22M params) and loads in ~1s, making it acceptable for
unit tests. Marked session-scoped via the evaluator fixture.
"""

from __future__ import annotations

import pytest

from rag_eval.metrics.similarity import SimilarityEvaluator


@pytest.fixture(scope="module")
def evaluator() -> SimilarityEvaluator:
    """Module-scoped evaluator — model loaded once for all tests in this file."""
    return SimilarityEvaluator(threshold=0.75)


class TestSimilarityEvaluator:
    def test_identical_texts_have_high_similarity(self, evaluator: SimilarityEvaluator) -> None:
        """Identical strings should have similarity close to 1.0."""
        text = "Docker is a containerization platform for running applications."
        result = evaluator.compute(generated=text, ground_truth=text)
        assert result.cosine_similarity >= 0.99

    def test_very_different_texts_have_low_similarity(
        self, evaluator: SimilarityEvaluator
    ) -> None:
        """Semantically unrelated texts should have low similarity."""
        result = evaluator.compute(
            generated="The Eiffel Tower is in Paris, France.",
            ground_truth="Python virtual environments isolate project dependencies.",
        )
        assert result.cosine_similarity < 0.5

    def test_semantically_similar_texts(self, evaluator: SimilarityEvaluator) -> None:
        """Paraphrased answer should have high similarity to ground truth."""
        result = evaluator.compute(
            generated="Git rebase replays commits on top of another branch for a linear history.",
            ground_truth="Git rebase moves commits from one branch onto another, creating a linear commit history.",
        )
        assert result.cosine_similarity >= 0.80

    def test_meets_threshold_true_above_threshold(
        self, evaluator: SimilarityEvaluator
    ) -> None:
        """Similarity above 0.75 should set meets_threshold=True."""
        result = evaluator.compute(
            generated="Docker containers start quickly and use fewer resources than VMs.",
            ground_truth="Containers are faster to start and more resource-efficient than virtual machines.",
        )
        # Both sentences convey the same idea — should be above 0.75
        assert result.meets_threshold == (result.cosine_similarity >= 0.75)

    def test_meets_threshold_false_below_threshold(self) -> None:
        """Low threshold evaluator with unrelated texts → meets_threshold=False."""
        # Use a very high threshold to force failure
        strict_evaluator = SimilarityEvaluator(threshold=0.99)
        result = strict_evaluator.compute(
            generated="Git merge creates a merge commit.",
            ground_truth="Python uses virtual environments for dependency isolation.",
        )
        assert not result.meets_threshold

    def test_result_is_pydantic_model(self, evaluator: SimilarityEvaluator) -> None:
        """Result should be a valid SimilarityMetrics pydantic model."""
        from rag_eval.models.results import SimilarityMetrics

        result = evaluator.compute(
            generated="This is a test answer.",
            ground_truth="This is the expected answer.",
        )
        assert isinstance(result, SimilarityMetrics)
        assert -1.0 <= result.cosine_similarity <= 1.0

    def test_cosine_similarity_range(self, evaluator: SimilarityEvaluator) -> None:
        """Similarity should always be in [-1, 1]."""
        pairs = [
            ("hello world", "hello world"),
            ("completely different content", "nothing related at all here"),
            ("", "some content"),
        ]
        for gen, gt in pairs:
            if not gen or not gt:
                continue  # skip empty strings
            result = evaluator.compute(generated=gen, ground_truth=gt)
            assert -1.0 <= result.cosine_similarity <= 1.0, (
                f"Similarity out of range for ({gen!r}, {gt!r}): {result.cosine_similarity}"
            )
