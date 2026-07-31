"""Unit tests for retrieval metrics (precision@K and recall@K).

All tests are deterministic and require no external services.
"""

from __future__ import annotations

import pytest

from rag_eval.metrics.retrieval import RetrievalEvaluator


@pytest.fixture
def evaluator() -> RetrievalEvaluator:
    return RetrievalEvaluator()


# ── precision_at_k ────────────────────────────────────────────────────────────


class TestPrecisionAtK:
    def test_perfect_precision(self, evaluator: RetrievalEvaluator) -> None:
        """All retrieved docs are relevant."""
        score = evaluator.precision_at_k(
            retrieved_ids=["a", "b", "c"],
            relevant_ids=["a", "b", "c", "d"],
            k=3,
        )
        assert score == pytest.approx(1.0)

    def test_zero_precision(self, evaluator: RetrievalEvaluator) -> None:
        """No retrieved docs are relevant."""
        score = evaluator.precision_at_k(
            retrieved_ids=["x", "y", "z"],
            relevant_ids=["a", "b", "c"],
            k=3,
        )
        assert score == pytest.approx(0.0)

    def test_partial_precision(self, evaluator: RetrievalEvaluator) -> None:
        """2 out of 5 retrieved docs are relevant → 0.4."""
        score = evaluator.precision_at_k(
            retrieved_ids=["a", "b", "x", "y", "z"],
            relevant_ids=["a", "b", "c"],
            k=5,
        )
        assert score == pytest.approx(2 / 5)

    def test_k_zero_returns_zero(self, evaluator: RetrievalEvaluator) -> None:
        """Edge case: k=0 returns 0."""
        score = evaluator.precision_at_k(["a", "b"], ["a"], k=0)
        assert score == 0.0

    def test_k_larger_than_retrieved(self, evaluator: RetrievalEvaluator) -> None:
        """Only first K items are considered even if fewer are retrieved."""
        # K=5 but only 2 docs returned; precision uses actual K for denominator
        score = evaluator.precision_at_k(
            retrieved_ids=["a", "b"],
            relevant_ids=["a"],
            k=5,
        )
        # 1 hit / 5 (k denominator) = 0.2
        assert score == pytest.approx(0.2)

    def test_order_matters_for_k(self, evaluator: RetrievalEvaluator) -> None:
        """Only the first K retrieved docs count."""
        # Relevant doc "a" is at position 3, beyond K=2
        score = evaluator.precision_at_k(
            retrieved_ids=["x", "y", "a"],
            relevant_ids=["a"],
            k=2,
        )
        assert score == pytest.approx(0.0)


# ── recall_at_k ───────────────────────────────────────────────────────────────


class TestRecallAtK:
    def test_perfect_recall(self, evaluator: RetrievalEvaluator) -> None:
        """All relevant docs are retrieved."""
        score = evaluator.recall_at_k(
            retrieved_ids=["a", "b", "c", "x"],
            relevant_ids=["a", "b", "c"],
            k=4,
        )
        assert score == pytest.approx(1.0)

    def test_zero_recall(self, evaluator: RetrievalEvaluator) -> None:
        """No relevant docs are retrieved."""
        score = evaluator.recall_at_k(
            retrieved_ids=["x", "y", "z"],
            relevant_ids=["a", "b", "c"],
            k=3,
        )
        assert score == pytest.approx(0.0)

    def test_partial_recall(self, evaluator: RetrievalEvaluator) -> None:
        """2 of 3 relevant docs retrieved → recall=2/3."""
        score = evaluator.recall_at_k(
            retrieved_ids=["a", "b", "x"],
            relevant_ids=["a", "b", "c"],
            k=3,
        )
        assert score == pytest.approx(2 / 3)

    def test_empty_relevant_returns_one(self, evaluator: RetrievalEvaluator) -> None:
        """Edge case: no relevant docs → recall=1.0 (nothing to miss)."""
        score = evaluator.recall_at_k(["a", "b"], [], k=2)
        assert score == pytest.approx(1.0)

    def test_recall_k_cutoff(self, evaluator: RetrievalEvaluator) -> None:
        """Docs beyond K are not counted."""
        # "c" is at position 4 (0-indexed 3), beyond K=3
        score = evaluator.recall_at_k(
            retrieved_ids=["a", "b", "x", "c"],
            relevant_ids=["a", "b", "c"],
            k=3,
        )
        assert score == pytest.approx(2 / 3)


# ── evaluate (combined) ───────────────────────────────────────────────────────


class TestEvaluateCombined:
    def test_evaluate_returns_correct_types(self, evaluator: RetrievalEvaluator) -> None:
        result = evaluator.evaluate(
            retrieved_ids=["doc-1", "doc-2", "doc-3"],
            relevant_ids=["doc-1", "doc-2"],
            k=3,
        )
        assert isinstance(result.precision_at_k, float)
        assert isinstance(result.recall_at_k, float)
        assert result.k == 3
        assert result.retrieved_count == 3
        assert result.relevant_count == 2

    def test_evaluate_perfect_results(self, evaluator: RetrievalEvaluator) -> None:
        result = evaluator.evaluate(
            retrieved_ids=["a", "b"],
            relevant_ids=["a", "b"],
            k=2,
        )
        assert result.precision_at_k == pytest.approx(1.0)
        assert result.recall_at_k == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "retrieved,relevant,k,expected_p,expected_r",
        [
            (["a", "b", "c"], ["a"], 3, 1 / 3, 1.0),
            (["x", "y", "z"], ["a", "b"], 3, 0.0, 0.0),
            (["a", "x"], ["a", "b", "c"], 2, 0.5, 1 / 3),
        ],
    )
    def test_evaluate_parametrized(
        self,
        evaluator: RetrievalEvaluator,
        retrieved: list[str],
        relevant: list[str],
        k: int,
        expected_p: float,
        expected_r: float,
    ) -> None:
        result = evaluator.evaluate(retrieved, relevant, k=k)
        assert result.precision_at_k == pytest.approx(expected_p, abs=1e-4)
        assert result.recall_at_k == pytest.approx(expected_r, abs=1e-4)
