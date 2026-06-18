"""Metric degradation thresholds.

These define the baseline performance floor for the RAG system.
If any aggregate metric falls below (or above) its threshold, the
AlertEngine fires and CI fails.

Override defaults by subclassing or patching in tests.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MetricThresholds(BaseModel):
    """Pass/fail thresholds for eval run aggregates.

    All values represent the MINIMUM acceptable performance except
    max_hallucination_rate, which is a MAXIMUM cap.

    Attributes:
        min_precision_at_k: Minimum acceptable mean precision@K.
        min_recall_at_k: Minimum acceptable mean recall@K.
        min_similarity: Minimum acceptable mean cosine similarity.
        max_hallucination_rate: Maximum acceptable fraction of hallucinated answers.
        min_routing_accuracy: Minimum fraction of correctly routed responses.
    """

    min_precision_at_k: float = Field(default=0.60, ge=0.0, le=1.0)
    min_recall_at_k: float = Field(default=0.60, ge=0.0, le=1.0)
    min_similarity: float = Field(default=0.75, ge=0.0, le=1.0)
    max_hallucination_rate: float = Field(default=0.15, ge=0.0, le=1.0)
    min_routing_accuracy: float = Field(default=0.90, ge=0.0, le=1.0)

    def describe(self) -> dict[str, str]:
        """Human-readable description of each threshold for reporting."""
        return {
            "Precision@K": f">= {self.min_precision_at_k:.0%}",
            "Recall@K": f">= {self.min_recall_at_k:.0%}",
            "Semantic Similarity": f">= {self.min_similarity:.0%}",
            "Hallucination Rate": f"<= {self.max_hallucination_rate:.0%}",
            "Routing Accuracy": f">= {self.min_routing_accuracy:.0%}",
        }
