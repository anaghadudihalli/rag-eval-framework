"""Shared pytest fixtures for the RAG eval test suite.

Fixtures are organized by scope:
- session: expensive setup (model loading) — created once per test session
- function (default): created fresh for each test

Test categories:
- unit: No external services. Fast. Run with `pytest tests/unit/`.
- integration: May require ChromaDB or OpenSearch. Run with `pytest tests/integration/`.
- opensearch: Requires a live OpenSearch Docker container.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rag_eval.config import Settings
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


# ── Settings ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Settings configured for testing (ChromaDB, no real API key required)."""
    return Settings(
        openai_api_key="sk-test-placeholder",  # type: ignore[arg-type]
        vector_store_backend="chroma",
        opensearch_url="http://localhost:9200",
        opensearch_index="rag-eval-test",
        top_k=3,
        confidence_threshold=0.7,
        similarity_threshold=0.75,
        golden_dataset_path="data/golden_dataset.json",
    )


# ── Golden dataset ───────────────────────────────────────────────────────────


@pytest.fixture
def sample_golden_sample() -> GoldenSample:
    """A single golden sample for unit testing."""
    return GoldenSample(
        id="test-001",
        query="What is Docker?",
        ground_truth_answer="Docker is a containerization platform that uses OS-level virtualization to deliver software in packages called containers.",
        relevant_doc_ids=["docker-001", "docker-002"],
        category="docker",
    )


@pytest.fixture
def sample_golden_dataset(sample_golden_sample: GoldenSample) -> GoldenDataset:
    """A minimal golden dataset for unit testing."""
    return GoldenDataset(
        version="1.0",
        description="Test dataset",
        created_at=datetime.now(timezone.utc),
        samples=[sample_golden_sample],
    )


@pytest.fixture
def three_sample_dataset() -> GoldenDataset:
    """A 3-sample dataset for aggregate metric testing."""
    return GoldenDataset(
        version="1.0",
        description="3-sample test dataset",
        created_at=datetime.now(timezone.utc),
        samples=[
            GoldenSample(
                id=f"test-{i:03d}",
                query=f"Test query {i}",
                ground_truth_answer=f"Ground truth answer {i} with sufficient length.",
                relevant_doc_ids=[f"doc-{i}"],
                category="test",
            )
            for i in range(3)
        ],
    )


# ── Metric result fixtures ───────────────────────────────────────────────────


@pytest.fixture
def perfect_retrieval_metrics() -> RetrievalMetrics:
    return RetrievalMetrics(precision_at_k=1.0, recall_at_k=1.0, k=5, retrieved_count=5, relevant_count=2)


@pytest.fixture
def poor_retrieval_metrics() -> RetrievalMetrics:
    return RetrievalMetrics(precision_at_k=0.2, recall_at_k=0.2, k=5, retrieved_count=5, relevant_count=5)


@pytest.fixture
def good_similarity_metrics() -> SimilarityMetrics:
    return SimilarityMetrics(cosine_similarity=0.90, meets_threshold=True)


@pytest.fixture
def poor_similarity_metrics() -> SimilarityMetrics:
    return SimilarityMetrics(cosine_similarity=0.50, meets_threshold=False)


@pytest.fixture
def no_hallucination() -> HallucinationResult:
    return HallucinationResult(
        is_hallucination=False,
        judge_reasoning="The answer is fully supported by the context.",
        judge_confidence=0.95,
    )


@pytest.fixture
def is_hallucination() -> HallucinationResult:
    return HallucinationResult(
        is_hallucination=True,
        judge_reasoning="The answer claims X but context does not mention X.",
        judge_confidence=0.85,
    )


@pytest.fixture
def high_confidence_routing() -> ConfidenceRoutingResult:
    return ConfidenceRoutingResult(
        answer_confidence=0.92,
        confidence_source="logprobs",
        routed_to_human=False,
        routing_correct=True,
        threshold_used=0.7,
    )


@pytest.fixture
def low_confidence_routing() -> ConfidenceRoutingResult:
    return ConfidenceRoutingResult(
        answer_confidence=0.45,
        confidence_source="similarity_fallback",
        routed_to_human=True,
        routing_correct=True,
        threshold_used=0.7,
    )


# ── Mock OpenAI client ───────────────────────────────────────────────────────


@pytest.fixture
def mock_hallucination_response_no_hallucination() -> dict[str, Any]:
    """JSON response dict for a clean (non-hallucinated) judge verdict."""
    return {
        "is_hallucination": False,
        "reasoning": "All claims are supported by the provided context.",
        "confidence": 0.92,
    }


@pytest.fixture
def mock_hallucination_response_hallucination() -> dict[str, Any]:
    """JSON response dict for a hallucination verdict."""
    return {
        "is_hallucination": True,
        "reasoning": "The answer mentions a specific version number not found in the context.",
        "confidence": 0.88,
    }


# ── Full report fixture ──────────────────────────────────────────────────────


@pytest.fixture
def sample_eval_result(
    sample_golden_sample: GoldenSample,
    perfect_retrieval_metrics: RetrievalMetrics,
    good_similarity_metrics: SimilarityMetrics,
    no_hallucination: HallucinationResult,
    high_confidence_routing: ConfidenceRoutingResult,
) -> SampleEvalResult:
    return SampleEvalResult(
        sample_id=sample_golden_sample.id,
        query=sample_golden_sample.query,
        ground_truth_answer=sample_golden_sample.ground_truth_answer,
        generated_answer="Docker is a platform that uses OS-level virtualization for containers.",
        retrieved_doc_ids=["docker-001", "docker-002"],
        retrieval=perfect_retrieval_metrics,
        similarity=good_similarity_metrics,
        hallucination=no_hallucination,
        routing=high_confidence_routing,
        latency_ms=350.0,
    )


@pytest.fixture
def sample_eval_report(
    sample_eval_result: SampleEvalResult,
) -> EvalReport:
    return EvalReport(
        run_id="test-run-001",
        git_sha="abc1234",
        vector_store_backend="chroma",
        embedding_model="all-MiniLM-L6-v2",
        judge_model="gpt-3.5-turbo",
        top_k=5,
        aggregate=AggregateMetrics(
            sample_count=1,
            mean_precision_at_k=1.0,
            mean_recall_at_k=1.0,
            mean_similarity=0.90,
            hallucination_rate=0.0,
            routing_accuracy=1.0,
            mean_latency_ms=350.0,
            pass_rate=1.0,
        ),
        samples=[sample_eval_result],
        alerts=[],
    )
