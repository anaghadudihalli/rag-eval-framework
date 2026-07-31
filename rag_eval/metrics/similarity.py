"""Semantic similarity evaluator using sentence-transformers.

Uses the all-MiniLM-L6-v2 model to compute cosine similarity between
the generated answer and the ground-truth answer. This model was chosen
because it's fast (22M params), widely benchmarked, and the same model
used in the EA production RAG embedding pipeline.

The SimilarityEvaluator is a singleton — the model is loaded once on
first use and shared across the entire eval run.
"""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from rag_eval.config import get_settings
from rag_eval.models.results import SimilarityMetrics

logger = logging.getLogger(__name__)

# Module-level singleton — avoid reloading the model for every sample
_model_instance: SentenceTransformer | None = None


def _get_model(model_name: str) -> SentenceTransformer:
    """Load the sentence-transformers model once (singleton pattern)."""
    global _model_instance
    if _model_instance is None:
        logger.info("Loading sentence-transformers model: %s", model_name)
        _model_instance = SentenceTransformer(model_name)
        logger.info("Model loaded.")
    return _model_instance


class SimilarityEvaluator:
    """Computes cosine semantic similarity between two text strings.

    Args:
        model_name: HuggingFace model ID (default: settings.embedding_model).
        threshold: Similarity score below which meets_threshold=False.
    """

    def __init__(
        self,
        model_name: str | None = None,
        threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.embedding_model
        self._threshold = threshold if threshold is not None else settings.similarity_threshold

    def _model(self) -> SentenceTransformer:
        return _get_model(self._model_name)

    def compute(self, generated: str, ground_truth: str) -> SimilarityMetrics:
        """Compute cosine similarity between generated and ground-truth answers.

        Args:
            generated: The RAG system's generated answer.
            ground_truth: The gold-standard answer from GoldenSample.

        Returns:
            SimilarityMetrics with cosine_similarity ∈ [-1, 1] and threshold flag.
        """
        model = self._model()
        embeddings = model.encode(
            [generated, ground_truth],
            convert_to_tensor=True,
            normalize_embeddings=True,  # ensures cosine via dot product
        )
        # cos_sim returns a 2D tensor; extract the scalar
        similarity_tensor = cos_sim(embeddings[0], embeddings[1])
        score = float(similarity_tensor.item())

        logger.debug(
            "Similarity — score=%.4f (threshold=%.2f) | generated='%s...' | gt='%s...'",
            score,
            self._threshold,
            generated[:60],
            ground_truth[:60],
        )

        return SimilarityMetrics(
            cosine_similarity=round(score, 4),
            meets_threshold=score >= self._threshold,
        )
