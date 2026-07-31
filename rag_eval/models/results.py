"""Eval result models — typed outputs for every metric and the full report."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field


# ── Per-metric result models ────────────────────────────────────────────────


class RetrievalMetrics(BaseModel):
    """Precision and recall for a single retrieval.

    Attributes:
        precision_at_k: Fraction of retrieved docs that were relevant.
        recall_at_k: Fraction of relevant docs that were retrieved.
        k: The K used in retrieval (matches settings.top_k).
        retrieved_count: Actual number of docs returned (may be < k if index is small).
        relevant_count: Total number of relevant docs for this query.
    """

    precision_at_k: float = Field(ge=0.0, le=1.0)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    k: int
    retrieved_count: int
    relevant_count: int


class SimilarityMetrics(BaseModel):
    """Semantic similarity between generated and ground-truth answer.

    Attributes:
        cosine_similarity: [-1, 1] cosine similarity from sentence-transformers.
        meets_threshold: True if similarity >= configured threshold.
    """

    cosine_similarity: float = Field(ge=-1.0, le=1.0)
    meets_threshold: bool


class HallucinationResult(BaseModel):
    """LLM-as-judge hallucination verdict.

    Attributes:
        is_hallucination: True if the answer introduces unsupported claims.
        judge_reasoning: The judge's explanation (verbatim from GPT-3.5-turbo).
        judge_confidence: The judge's self-reported confidence [0, 1].
        judge_model: Model used for this judgment (for audit trail).
    """

    is_hallucination: bool
    judge_reasoning: str
    judge_confidence: float = Field(ge=0.0, le=1.0)
    judge_model: str = "gpt-3.5-turbo"


class ConfidenceRoutingResult(BaseModel):
    """Confidence-based routing decision for a single answer.

    Confidence is computed from OpenAI token logprobs (exp(mean(logprobs))).
    Falls back to cosine similarity score if logprobs are unavailable.

    Attributes:
        answer_confidence: Scalar [0, 1] confidence in the generated answer.
        confidence_source: "logprobs" or "similarity_fallback".
        routed_to_human: True if confidence < threshold → escalated.
        routing_correct: True if routing decision matches expected behavior.
            (high-confidence answers should NOT be routed, low-confidence should be).
        threshold_used: The confidence threshold applied.
    """

    answer_confidence: float = Field(ge=0.0, le=1.0)
    confidence_source: str  # "logprobs" | "similarity_fallback"
    routed_to_human: bool
    routing_correct: bool
    threshold_used: float


# ── Per-sample result model ─────────────────────────────────────────────────


class SampleEvalResult(BaseModel):
    """Complete evaluation result for one GoldenSample.

    Aggregates all metric results for a single query-answer pair.
    """

    sample_id: str
    query: str
    ground_truth_answer: str
    generated_answer: str
    retrieved_doc_ids: list[str]
    retrieval: RetrievalMetrics
    similarity: SimilarityMetrics
    hallucination: HallucinationResult
    routing: ConfidenceRoutingResult
    latency_ms: float = Field(ge=0.0)

    @computed_field
    @property
    def passed(self) -> bool:
        """True if this sample passes all quality checks."""
        return (
            not self.hallucination.is_hallucination
            and self.similarity.meets_threshold
            and self.routing.routing_correct
        )


# ── Aggregate metrics ───────────────────────────────────────────────────────


class AggregateMetrics(BaseModel):
    """Aggregated metrics across all samples in an eval run."""

    sample_count: int
    mean_precision_at_k: float = Field(ge=0.0, le=1.0)
    mean_recall_at_k: float = Field(ge=0.0, le=1.0)
    mean_similarity: float = Field(ge=-1.0, le=1.0)
    hallucination_rate: float = Field(ge=0.0, le=1.0)
    routing_accuracy: float = Field(ge=0.0, le=1.0)
    mean_latency_ms: float = Field(ge=0.0)
    pass_rate: float = Field(ge=0.0, le=1.0)


# ── Full eval report ────────────────────────────────────────────────────────


class EvalReport(BaseModel):
    """The complete output of one evaluation run.

    This is the root object serialized to reports/<run_id>.json.
    """

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    git_sha: Optional[str] = None
    vector_store_backend: str
    embedding_model: str
    judge_model: str
    top_k: int
    aggregate: AggregateMetrics
    samples: List[SampleEvalResult]
    alerts: List[str] = Field(default_factory=list)

    @computed_field
    @property
    def has_alerts(self) -> bool:
        """True if any metric threshold was breached."""
        return len(self.alerts) > 0
