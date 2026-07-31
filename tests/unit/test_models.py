"""Unit tests for all Pydantic v2 data models.

Tests validation, computed fields, custom validators, and serialization.
No external services required.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from rag_eval.models.dataset import GoldenDataset, GoldenSample
from rag_eval.models.results import (
    AggregateMetrics,
    ConfidenceRoutingResult,
    EvalReport,
    HallucinationResult,
    RetrievalMetrics,
    SampleEvalResult,
    SimilarityMetrics,
)
from rag_eval.models.thresholds import MetricThresholds


# ── GoldenSample ─────────────────────────────────────────────────────────────


class TestGoldenSample:
    def test_valid_sample(self) -> None:
        sample = GoldenSample(
            id="test-001",
            query="What is Docker?",
            ground_truth_answer="Docker is a containerization platform.",
            relevant_doc_ids=["docker-001"],
        )
        assert sample.id == "test-001"
        assert sample.category == "general"  # default

    def test_query_min_length(self) -> None:
        with pytest.raises(ValidationError):
            GoldenSample(
                id="x",
                query="Hi",  # too short (< 5 chars)
                ground_truth_answer="Long enough answer for validation.",
                relevant_doc_ids=["doc-1"],
            )

    def test_ground_truth_min_length(self) -> None:
        with pytest.raises(ValidationError):
            GoldenSample(
                id="x",
                query="What is this?",
                ground_truth_answer="Short",  # too short (< 10 chars)
                relevant_doc_ids=["doc-1"],
            )

    def test_empty_relevant_doc_ids_fails(self) -> None:
        with pytest.raises(ValidationError):
            GoldenSample(
                id="x",
                query="What is Docker?",
                ground_truth_answer="Docker is a containerization platform.",
                relevant_doc_ids=[],  # must have at least 1
            )


# ── GoldenDataset ─────────────────────────────────────────────────────────────


class TestGoldenDataset:
    def test_valid_dataset(self) -> None:
        dataset = GoldenDataset(
            samples=[
                GoldenSample(
                    id="test-001",
                    query="What is Docker?",
                    ground_truth_answer="Docker is a containerization platform.",
                    relevant_doc_ids=["docker-001"],
                )
            ]
        )
        assert len(dataset) == 1

    def test_duplicate_ids_fail(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate"):
            GoldenDataset(
                samples=[
                    GoldenSample(
                        id="same-id",
                        query="First query here?",
                        ground_truth_answer="First answer that is long enough.",
                        relevant_doc_ids=["doc-1"],
                    ),
                    GoldenSample(
                        id="same-id",  # duplicate
                        query="Second query here?",
                        ground_truth_answer="Second answer that is long enough.",
                        relevant_doc_ids=["doc-2"],
                    ),
                ]
            )

    def test_by_category_groups_correctly(self) -> None:
        dataset = GoldenDataset(
            samples=[
                GoldenSample(
                    id="git-001",
                    query="What is Git Flow?",
                    ground_truth_answer="Git Flow is a branching strategy.",
                    relevant_doc_ids=["git-001"],
                    category="git",
                ),
                GoldenSample(
                    id="docker-001",
                    query="What is Docker anyway?",
                    ground_truth_answer="Docker is a containerization platform.",
                    relevant_doc_ids=["docker-001"],
                    category="docker",
                ),
            ]
        )
        groups = dataset.by_category()
        assert "git" in groups
        assert "docker" in groups
        assert len(groups["git"]) == 1

    def test_empty_samples_fails(self) -> None:
        with pytest.raises(ValidationError):
            GoldenDataset(samples=[])


# ── RetrievalMetrics ──────────────────────────────────────────────────────────


class TestRetrievalMetrics:
    def test_valid_metrics(self) -> None:
        m = RetrievalMetrics(
            precision_at_k=0.8, recall_at_k=0.6, k=5, retrieved_count=5, relevant_count=3
        )
        assert m.precision_at_k == 0.8

    def test_precision_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalMetrics(precision_at_k=1.5, recall_at_k=0.5, k=5, retrieved_count=5, relevant_count=3)

    def test_recall_negative(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalMetrics(precision_at_k=0.5, recall_at_k=-0.1, k=5, retrieved_count=5, relevant_count=3)


# ── SampleEvalResult.passed computed field ───────────────────────────────────


class TestSampleEvalResultPassedField:
    def _make_result(
        self,
        is_hallucination: bool,
        meets_threshold: bool,
        routing_correct: bool,
    ) -> SampleEvalResult:
        return SampleEvalResult(
            sample_id="test",
            query="test query here",
            ground_truth_answer="ground truth answer here is long enough",
            generated_answer="generated answer",
            retrieved_doc_ids=["doc-1"],
            retrieval=RetrievalMetrics(
                precision_at_k=0.8, recall_at_k=0.8, k=5,
                retrieved_count=5, relevant_count=2
            ),
            similarity=SimilarityMetrics(
                cosine_similarity=0.9 if meets_threshold else 0.5,
                meets_threshold=meets_threshold,
            ),
            hallucination=HallucinationResult(
                is_hallucination=is_hallucination,
                judge_reasoning="test",
                judge_confidence=0.9,
            ),
            routing=ConfidenceRoutingResult(
                answer_confidence=0.9,
                confidence_source="logprobs",
                routed_to_human=False,
                routing_correct=routing_correct,
                threshold_used=0.7,
            ),
            latency_ms=100.0,
        )

    def test_passed_all_good(self) -> None:
        result = self._make_result(
            is_hallucination=False, meets_threshold=True, routing_correct=True
        )
        assert result.passed is True

    def test_fails_when_hallucinated(self) -> None:
        result = self._make_result(
            is_hallucination=True, meets_threshold=True, routing_correct=True
        )
        assert result.passed is False

    def test_fails_when_similarity_below_threshold(self) -> None:
        result = self._make_result(
            is_hallucination=False, meets_threshold=False, routing_correct=True
        )
        assert result.passed is False

    def test_fails_when_routing_incorrect(self) -> None:
        result = self._make_result(
            is_hallucination=False, meets_threshold=True, routing_correct=False
        )
        assert result.passed is False


# ── MetricThresholds ──────────────────────────────────────────────────────────


class TestMetricThresholds:
    def test_defaults(self) -> None:
        t = MetricThresholds()
        assert t.min_precision_at_k == 0.60
        assert t.max_hallucination_rate == 0.15

    def test_describe_returns_strings(self) -> None:
        t = MetricThresholds()
        desc = t.describe()
        assert isinstance(desc, dict)
        assert "Precision@K" in desc
        assert ">=" in desc["Precision@K"]

    def test_out_of_range_threshold_fails(self) -> None:
        with pytest.raises(ValidationError):
            MetricThresholds(min_precision_at_k=1.5)


# ── EvalReport.has_alerts ─────────────────────────────────────────────────────


class TestEvalReportAlerts:
    def test_no_alerts_has_alerts_false(self, sample_eval_report: EvalReport) -> None:
        assert sample_eval_report.has_alerts is False

    def test_with_alerts_has_alerts_true(self, sample_eval_report: EvalReport) -> None:
        report_with_alerts = sample_eval_report.model_copy(
            update={"alerts": ["Precision@K below threshold"]}
        )
        assert report_with_alerts.has_alerts is True


# ── Serialization ─────────────────────────────────────────────────────────────


class TestSerialization:
    def test_golden_sample_round_trip(self) -> None:
        sample = GoldenSample(
            id="test-001",
            query="What is Docker?",
            ground_truth_answer="Docker is a containerization platform.",
            relevant_doc_ids=["docker-001"],
        )
        data = sample.model_dump()
        restored = GoldenSample.model_validate(data)
        assert restored == sample

    def test_eval_report_json_serialization(self, sample_eval_report: EvalReport) -> None:
        """EvalReport should serialize to valid JSON (datetimes → ISO strings)."""
        import json
        data = sample_eval_report.model_dump(mode="json")
        json_str = json.dumps(data)  # should not raise
        assert "run_id" in json_str
        assert "aggregate" in json_str
