"""Retrieval quality metrics: Precision@K and Recall@K.

These are the core retrieval metrics used in information retrieval research
and mirrored from the EA production RAG monitoring approach.

Definitions:
    Precision@K = |retrieved_top_k ∩ relevant| / K
    Recall@K    = |retrieved_top_k ∩ relevant| / |relevant|

where K is the number of documents retrieved per query.
"""

from __future__ import annotations

import logging

from rag_eval.models.results import RetrievalMetrics

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """Computes precision@K and recall@K for retrieved document sets.

    This is a stateless evaluator — all state is passed per-call.
    No embeddings or API calls involved.
    """

    @staticmethod
    def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
        """Fraction of the top-K retrieved documents that are relevant.

        Args:
            retrieved_ids: Ordered list of retrieved doc IDs (position matters for K).
            relevant_ids: Set of ground-truth relevant doc IDs.
            k: Cutoff rank (only the first K retrieved IDs are considered).

        Returns:
            Float in [0, 1]. Returns 0.0 if k == 0.
        """
        if k == 0:
            return 0.0
        top_k = retrieved_ids[:k]
        relevant_set = set(relevant_ids)
        hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
        return hits / k

    @staticmethod
    def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
        """Fraction of all relevant documents that appear in the top-K results.

        Args:
            retrieved_ids: Ordered list of retrieved doc IDs.
            relevant_ids: Set of ground-truth relevant doc IDs.
            k: Cutoff rank.

        Returns:
            Float in [0, 1]. Returns 1.0 if relevant_ids is empty (nothing to miss).
        """
        if not relevant_ids:
            logger.warning("recall_at_k called with empty relevant_ids — returning 1.0.")
            return 1.0
        top_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        hits = len(top_k & relevant_set)
        return hits / len(relevant_set)

    def evaluate(
        self,
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int,
    ) -> RetrievalMetrics:
        """Compute both precision and recall for a single retrieval.

        Args:
            retrieved_ids: Ordered list of retrieved doc IDs (metadata["doc_id"]).
            relevant_ids: Ground-truth relevant doc IDs from GoldenSample.
            k: The K used for retrieval (settings.top_k).

        Returns:
            RetrievalMetrics with precision_at_k, recall_at_k, and counts.
        """
        # Note: retrieved_ids may contain chunk-level IDs (e.g., "git-001")
        # that match the document-level relevant_ids. The ingestion pipeline
        # ensures all chunks inherit their parent's doc_id in metadata.
        actual_k = min(k, len(retrieved_ids))

        precision = self.precision_at_k(retrieved_ids, relevant_ids, k=actual_k)
        recall = self.recall_at_k(retrieved_ids, relevant_ids, k=actual_k)

        logger.debug(
            "Retrieval — P@%d=%.3f, R@%d=%.3f | retrieved=%s, relevant=%s",
            k, precision, k, recall, retrieved_ids[:k], relevant_ids,
        )

        return RetrievalMetrics(
            precision_at_k=round(precision, 4),
            recall_at_k=round(recall, 4),
            k=k,
            retrieved_count=len(retrieved_ids),
            relevant_count=len(relevant_ids),
        )
