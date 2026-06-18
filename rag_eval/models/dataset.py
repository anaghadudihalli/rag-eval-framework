"""Golden dataset models.

A GoldenDataset is the ground-truth eval set: a collection of queries
paired with expected answers and the doc IDs that *should* be retrieved.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, model_validator


class GoldenSample(BaseModel):
    """A single ground-truth evaluation sample.

    Attributes:
        id: Unique identifier for this sample (e.g. "git-001").
        query: The natural-language question posed to the RAG system.
        ground_truth_answer: The authoritative answer to compare against.
        relevant_doc_ids: IDs of documents that contain the answer.
            Used to compute precision@K and recall@K.
        category: Optional topic tag (e.g. "git", "docker") for grouped reporting.
    """

    id: str
    query: str = Field(min_length=5)
    ground_truth_answer: str = Field(min_length=10)
    relevant_doc_ids: list[str] = Field(min_length=1)
    category: str = "general"


class GoldenDataset(BaseModel):
    """A versioned collection of golden evaluation samples.

    Attributes:
        version: Schema version string for forward-compatibility.
        description: Human-readable description of the dataset.
        created_at: UTC timestamp of dataset creation.
        samples: The list of GoldenSample objects.
    """

    version: str = "1.0"
    description: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    samples: list[GoldenSample] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> GoldenDataset:
        """Ensure all sample IDs are unique within the dataset."""
        ids = [s.id for s in self.samples]
        if len(ids) != len(set(ids)):
            duplicates = [id_ for id_ in ids if ids.count(id_) > 1]
            raise ValueError(f"Duplicate sample IDs found: {set(duplicates)}")
        return self

    def __len__(self) -> int:
        return len(self.samples)

    def by_category(self) -> dict[str, list[GoldenSample]]:
        """Group samples by category for filtered reporting."""
        groups: dict[str, list[GoldenSample]] = {}
        for sample in self.samples:
            groups.setdefault(sample.category, []).append(sample)
        return groups
