"""Unit tests for EvalRunner.

Covers:
- _compute_aggregates() with known inputs
- _evaluate_sample() with fully mocked sub-evaluators
- run() end-to-end with mocked dataset, vector store, and all metric evaluators
- Error handling: failed samples are skipped, not crashed
- All-samples-failed raises RuntimeError
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

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
from rag_eval.pipeline.evaluator import EvalRunner, _get_git_sha, _load_golden_dataset


# ── Shared fixtures ──────────────────────────────────────────────────────────


def _make_sample(idx: int = 0) -> GoldenSample:
    return GoldenSample(
        id=f"test-{idx:03d}",
        query=f"What is container isolation? ({idx})",
        ground_truth_answer="Docker uses OS-level virtualization to isolate containers.",
        relevant_doc_ids=["docker-001"],
        category="docker",
    )


def _make_sample_result(
    idx: int = 0,
    precision: float = 0.8,
    recall: float = 1.0,
    similarity: float = 0.90,
    is_hallucination: bool = False,
    routing_correct: bool = True,
    latency_ms: float = 100.0,
) -> SampleEvalResult:
    return SampleEvalResult(
        sample_id=f"test-{idx:03d}",
        query=f"Query {idx}",
        ground_truth_answer="Ground truth.",
        generated_answer="Generated answer.",
        retrieved_doc_ids=["docker-001"],
        retrieval=RetrievalMetrics(
            precision_at_k=precision,
            recall_at_k=recall,
            k=5,
            retrieved_count=5,
            relevant_count=1,
        ),
        similarity=SimilarityMetrics(
            cosine_similarity=similarity,
            meets_threshold=similarity >= 0.75,
        ),
        hallucination=HallucinationResult(
            is_hallucination=is_hallucination,
            judge_reasoning="Test reasoning.",
            judge_confidence=0.9,
        ),
        routing=ConfidenceRoutingResult(
            answer_confidence=0.85,
            confidence_source="logprobs",
            routed_to_human=False,
            routing_correct=routing_correct,
            threshold_used=0.7,
        ),
        latency_ms=latency_ms,
    )


def _make_test_settings(tmpdir: str) -> Settings:
    return Settings(
        openai_api_key="sk-test-placeholder",  # type: ignore[arg-type]
        vector_store_backend="chroma",
        top_k=3,
        confidence_threshold=0.7,
        similarity_threshold=0.75,
        golden_dataset_path="data/golden_dataset.json",
        documents_dir="data/documents",
        reports_dir=tmpdir,
    )


# ── _get_git_sha ─────────────────────────────────────────────────────────────


class TestGetGitSha:
    def test_returns_string_or_none(self) -> None:
        sha = _get_git_sha()
        assert sha is None or isinstance(sha, str)

    def test_returns_none_when_git_fails(self) -> None:
        with patch("rag_eval.pipeline.evaluator.subprocess.run", side_effect=Exception("no git")):
            sha = _get_git_sha()
        assert sha is None


# ── _load_golden_dataset ─────────────────────────────────────────────────────


class TestLoadGoldenDataset:
    def test_loads_real_dataset(self) -> None:
        dataset = _load_golden_dataset("data/golden_dataset.json")
        assert len(dataset) > 0
        assert all(hasattr(s, "query") for s in dataset.samples)

    def test_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            _load_golden_dataset("data/nonexistent.json")


# ── _compute_aggregates ──────────────────────────────────────────────────────


class TestComputeAggregates:
    def _runner(self) -> EvalRunner:
        settings = Settings(openai_api_key="sk-test-placeholder")  # type: ignore[arg-type]
        with patch("rag_eval.pipeline.evaluator.get_settings", return_value=settings), \
             patch("rag_eval.metrics.similarity.SentenceTransformer"), \
             patch("rag_eval.llm_factory.get_chat_model"), \
             patch("rag_eval.llm_factory.get_chat_model"):
            return EvalRunner(settings=settings)

    def test_empty_results_raises(self) -> None:
        runner = self._runner()
        with pytest.raises(ValueError, match="empty"):
            runner._compute_aggregates([])

    def test_single_result_aggregates_correctly(self) -> None:
        runner = self._runner()
        result = _make_sample_result(precision=0.8, recall=1.0, similarity=0.9, latency_ms=200.0)
        agg = runner._compute_aggregates([result])
        assert agg.sample_count == 1
        assert agg.mean_precision_at_k == pytest.approx(0.8, abs=1e-4)
        assert agg.mean_recall_at_k == pytest.approx(1.0, abs=1e-4)
        assert agg.mean_similarity == pytest.approx(0.9, abs=1e-4)
        assert agg.hallucination_rate == pytest.approx(0.0, abs=1e-4)
        assert agg.routing_accuracy == pytest.approx(1.0, abs=1e-4)
        assert agg.mean_latency_ms == pytest.approx(200.0, abs=0.1)

    def test_multiple_results_averages_correctly(self) -> None:
        runner = self._runner()
        results = [
            _make_sample_result(0, precision=1.0, recall=1.0, similarity=0.9, latency_ms=100.0),
            _make_sample_result(1, precision=0.0, recall=0.0, similarity=0.5, latency_ms=300.0),
        ]
        agg = runner._compute_aggregates(results)
        assert agg.sample_count == 2
        assert agg.mean_precision_at_k == pytest.approx(0.5, abs=1e-4)
        assert agg.mean_recall_at_k == pytest.approx(0.5, abs=1e-4)
        assert agg.mean_similarity == pytest.approx(0.7, abs=1e-4)
        assert agg.mean_latency_ms == pytest.approx(200.0, abs=0.1)

    def test_hallucination_rate_counts_flagged_samples(self) -> None:
        runner = self._runner()
        results = [
            _make_sample_result(0, is_hallucination=False),
            _make_sample_result(1, is_hallucination=True),
            _make_sample_result(2, is_hallucination=True),
        ]
        agg = runner._compute_aggregates(results)
        assert agg.hallucination_rate == pytest.approx(2 / 3, abs=1e-4)

    def test_routing_accuracy_counts_correct_routing(self) -> None:
        runner = self._runner()
        results = [
            _make_sample_result(0, routing_correct=True),
            _make_sample_result(1, routing_correct=False),
        ]
        agg = runner._compute_aggregates(results)
        assert agg.routing_accuracy == pytest.approx(0.5, abs=1e-4)

    def test_pass_rate_uses_sample_passed_property(self) -> None:
        runner = self._runner()
        # Passed sample: no hallucination, similarity meets threshold, routing correct
        passed = _make_sample_result(0, similarity=0.90, is_hallucination=False, routing_correct=True)
        # Failed sample: has hallucination
        failed = _make_sample_result(1, similarity=0.90, is_hallucination=True, routing_correct=True)
        agg = runner._compute_aggregates([passed, failed])
        assert agg.pass_rate == pytest.approx(0.5, abs=1e-4)


# ── _evaluate_sample ─────────────────────────────────────────────────────────


class TestEvaluateSample:
    def _make_runner_with_mocks(
        self,
        answer: str = "Docker uses OS-level virtualization.",
        logprobs: list[float] | None = None,
        is_hallucination: bool = False,
    ) -> tuple[EvalRunner, MagicMock]:
        settings = Settings(
            openai_api_key="sk-test-placeholder",  # type: ignore[arg-type]
            top_k=3,
            confidence_threshold=0.7,
            similarity_threshold=0.75,
        )

        retrieved_docs = [
            Document(
                page_content="Docker uses OS-level virtualization for container isolation.",
                metadata={"doc_id": "docker-001"},
            )
        ]
        ai_message = AIMessage(
            content=answer,
            response_metadata=(
                {"logprobs": {"content": [{"token": t, "logprob": lp} for t, lp in
                               zip(answer.split()[:5], logprobs or [])]}}
                if logprobs else {}
            ),
        )

        mock_rag_chain = MagicMock()
        mock_rag_chain.run.return_value = (answer, retrieved_docs, logprobs or [], 150.0)

        hallucination_json = json.dumps({
            "is_hallucination": is_hallucination,
            "reasoning": "Test reasoning.",
            "confidence": 0.9,
        })
        mock_judge = MagicMock()
        mock_judge.invoke.return_value = MagicMock(content=hallucination_json)

        with patch("rag_eval.pipeline.evaluator.get_settings", return_value=settings), \
             patch("rag_eval.llm_factory.get_chat_model", return_value=mock_judge), \
             patch("rag_eval.llm_factory.get_chat_model"):
            runner = EvalRunner(settings=settings)

        return runner, mock_rag_chain

    def test_evaluate_sample_returns_sample_eval_result(self) -> None:
        runner, mock_chain = self._make_runner_with_mocks()
        sample = _make_sample(0)
        result = runner._evaluate_sample(sample, mock_chain)
        assert isinstance(result, SampleEvalResult)

    def test_sample_id_matches_input(self) -> None:
        runner, mock_chain = self._make_runner_with_mocks()
        sample = _make_sample(0)
        result = runner._evaluate_sample(sample, mock_chain)
        assert result.sample_id == sample.id

    def test_query_matches_input(self) -> None:
        runner, mock_chain = self._make_runner_with_mocks()
        sample = _make_sample(0)
        result = runner._evaluate_sample(sample, mock_chain)
        assert result.query == sample.query

    def test_latency_is_non_negative(self) -> None:
        runner, mock_chain = self._make_runner_with_mocks()
        result = runner._evaluate_sample(_make_sample(0), mock_chain)
        assert result.latency_ms >= 0.0

    def test_uses_logprobs_when_available(self) -> None:
        """Routing should use 'logprobs' source when logprobs are returned."""
        runner, mock_chain = self._make_runner_with_mocks(logprobs=[-0.1, -0.2, -0.05])
        result = runner._evaluate_sample(_make_sample(0), mock_chain)
        assert result.routing.confidence_source == "logprobs"

    def test_falls_back_to_similarity_when_no_logprobs(self) -> None:
        """Routing should use 'similarity_fallback' when logprobs list is empty."""
        runner, mock_chain = self._make_runner_with_mocks(logprobs=None)
        result = runner._evaluate_sample(_make_sample(0), mock_chain)
        assert result.routing.confidence_source == "similarity_fallback"

    def test_retrieved_doc_ids_extracted_from_metadata(self) -> None:
        runner, mock_chain = self._make_runner_with_mocks()
        result = runner._evaluate_sample(_make_sample(0), mock_chain)
        assert "docker-001" in result.retrieved_doc_ids

    def test_doc_without_doc_id_gets_unknown_placeholder(self) -> None:
        """Docs missing doc_id metadata get 'unknown-N' as fallback."""
        settings = Settings(openai_api_key="sk-test-placeholder")  # type: ignore[arg-type]
        doc_no_id = Document(page_content="Some content.", metadata={})
        hallucination_json = json.dumps(
            {"is_hallucination": False, "reasoning": "ok", "confidence": 0.9}
        )
        mock_judge = MagicMock()
        mock_judge.invoke.return_value = MagicMock(content=hallucination_json)
        mock_chain = MagicMock()
        mock_chain.run.return_value = ("answer", [doc_no_id], [], 50.0)

        with patch("rag_eval.pipeline.evaluator.get_settings", return_value=settings), \
             patch("rag_eval.llm_factory.get_chat_model", return_value=mock_judge), \
             patch("rag_eval.llm_factory.get_chat_model"):
            runner = EvalRunner(settings=settings)

        result = runner._evaluate_sample(_make_sample(0), mock_chain)
        assert result.retrieved_doc_ids[0].startswith("unknown-")


# ── run() ────────────────────────────────────────────────────────────────────


class TestEvalRunnerRun:
    def _patched_run(
        self,
        tmpdir: str,
        num_samples: int = 2,
        make_sample_fail: bool = False,
    ) -> EvalReport:
        settings = _make_test_settings(tmpdir)
        dataset = GoldenDataset(
            version="1.0",
            description="Test",
            samples=[_make_sample(i) for i in range(num_samples)],
        )
        retrieved_docs = [
            Document(
                page_content="Docker uses namespaces for isolation.",
                metadata={"doc_id": "docker-001"},
            )
        ]
        answer = "Docker uses OS-level virtualization."
        hallucination_json = json.dumps(
            {"is_hallucination": False, "reasoning": "ok", "confidence": 0.9}
        )
        mock_judge = MagicMock()
        mock_judge.invoke.return_value = MagicMock(content=hallucination_json)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content=answer, response_metadata={})

        mock_retriever = MagicMock()
        if make_sample_fail:
            mock_retriever.invoke.side_effect = RuntimeError("retriever exploded")
        else:
            mock_retriever.invoke.return_value = retrieved_docs

        mock_store = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever

        with patch("rag_eval.pipeline.evaluator.get_settings", return_value=settings), \
             patch("rag_eval.pipeline.evaluator._load_golden_dataset", return_value=dataset), \
             patch("rag_eval.pipeline.evaluator.get_vector_store", return_value=mock_store), \
             patch("rag_eval.llm_factory.get_chat_model", return_value=mock_llm), \
             patch("rag_eval.llm_factory.get_chat_model", return_value=mock_judge):
            runner = EvalRunner(settings=settings)
            return runner.run()

    def test_run_returns_eval_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = self._patched_run(tmpdir)
        assert isinstance(report, EvalReport)

    def test_run_sample_count_matches_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = self._patched_run(tmpdir, num_samples=3)
        assert report.aggregate.sample_count == 3

    def test_run_creates_json_report_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = self._patched_run(tmpdir)
            report_files = list(__import__("pathlib").Path(tmpdir).glob("*.json"))
        assert len(report_files) == 1
        assert report_files[0].stem == report.run_id

    def test_run_report_has_correct_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = self._patched_run(tmpdir)
        assert report.vector_store_backend == "chroma"

    def test_run_skips_failed_samples_gracefully(self) -> None:
        """If all but one sample fails, run() should still produce a report."""
        settings = _make_test_settings("/tmp")
        dataset = GoldenDataset(
            version="1.0",
            description="Test",
            samples=[_make_sample(i) for i in range(3)],
        )
        retrieved_docs = [
            Document(page_content="Docker content.", metadata={"doc_id": "docker-001"})
        ]
        hallucination_json = json.dumps(
            {"is_hallucination": False, "reasoning": "ok", "confidence": 0.9}
        )
        mock_judge = MagicMock()
        mock_judge.invoke.return_value = MagicMock(content=hallucination_json)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Answer.", response_metadata={})

        call_count = 0

        def retriever_side_effect(query: str) -> list[Document]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Simulated failure")
            return retrieved_docs

        mock_retriever = MagicMock()
        mock_retriever.invoke.side_effect = retriever_side_effect
        mock_store = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_test_settings(tmpdir)
            with patch("rag_eval.pipeline.evaluator.get_settings", return_value=settings), \
                 patch("rag_eval.pipeline.evaluator._load_golden_dataset", return_value=dataset), \
                 patch("rag_eval.pipeline.evaluator.get_vector_store", return_value=mock_store), \
                 patch("rag_eval.llm_factory.get_chat_model", return_value=mock_llm), \
                 patch("rag_eval.llm_factory.get_chat_model", return_value=mock_judge):
                runner = EvalRunner(settings=settings)
                report = runner.run()

        # Only 1 sample succeeded — report should still be produced
        assert report.aggregate.sample_count == 1

    def test_run_raises_when_all_samples_fail(self) -> None:
        settings = Settings(openai_api_key="sk-test-placeholder")  # type: ignore[arg-type]
        dataset = GoldenDataset(
            version="1.0",
            description="Test",
            samples=[_make_sample(0)],
        )
        mock_retriever = MagicMock()
        mock_retriever.invoke.side_effect = RuntimeError("always fails")
        mock_store = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_test_settings(tmpdir)
            with patch("rag_eval.pipeline.evaluator.get_settings", return_value=settings), \
                 patch("rag_eval.pipeline.evaluator._load_golden_dataset", return_value=dataset), \
                 patch("rag_eval.pipeline.evaluator.get_vector_store", return_value=mock_store), \
                 patch("rag_eval.llm_factory.get_chat_model"), \
                 patch("rag_eval.llm_factory.get_chat_model"):
                runner = EvalRunner(settings=settings)
                with pytest.raises(RuntimeError, match="All samples failed"):
                    runner.run()

    def test_run_report_has_alerts_when_thresholds_breached(self) -> None:
        """Force very strict thresholds so alerts fire."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_test_settings(tmpdir)
            dataset = GoldenDataset(
                version="1.0",
                description="Test",
                samples=[_make_sample(0)],
            )
            retrieved_docs = [
                Document(page_content="Unrelated content.", metadata={"doc_id": "other-999"})
            ]
            hallucination_json = json.dumps(
                {"is_hallucination": False, "reasoning": "ok", "confidence": 0.9}
            )
            mock_judge = MagicMock()
            mock_judge.invoke.return_value = MagicMock(content=hallucination_json)
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = AIMessage(content="Answer.", response_metadata={})
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = retrieved_docs
            mock_store = MagicMock()
            mock_store.as_retriever.return_value = mock_retriever

            # Very strict thresholds — precision will be 0.0, should fire alert
            strict_thresholds = MetricThresholds(min_precision_at_k=0.99)

            with patch("rag_eval.pipeline.evaluator.get_settings", return_value=settings), \
                 patch("rag_eval.pipeline.evaluator._load_golden_dataset", return_value=dataset), \
                 patch("rag_eval.pipeline.evaluator.get_vector_store", return_value=mock_store), \
                 patch("rag_eval.llm_factory.get_chat_model", return_value=mock_llm), \
                 patch("rag_eval.llm_factory.get_chat_model", return_value=mock_judge):
                runner = EvalRunner(settings=settings, thresholds=strict_thresholds)
                report = runner.run()

        assert report.has_alerts
        assert any("Precision" in alert for alert in report.alerts)
