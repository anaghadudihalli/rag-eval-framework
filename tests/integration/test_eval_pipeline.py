"""Integration tests for the full eval pipeline.

Uses ChromaDB backend (no Docker required) with mocked OpenAI calls.
Tests the end-to-end flow: load dataset → ingest → retrieve → evaluate → report.

Run: pytest tests/integration/ -v -m "not opensearch"
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_eval.models.dataset import GoldenDataset, GoldenSample
from rag_eval.models.results import EvalReport
from rag_eval.models.thresholds import MetricThresholds
from rag_eval.reporting.alerts import AlertEngine
from rag_eval.reporting.json_reporter import JSONReporter


pytestmark = pytest.mark.integration


# ── Alert engine integration ──────────────────────────────────────────────────


class TestAlertEngineIntegration:
    def test_no_alerts_when_all_metrics_pass(self) -> None:
        from rag_eval.models.results import AggregateMetrics

        metrics = AggregateMetrics(
            sample_count=10,
            mean_precision_at_k=0.8,
            mean_recall_at_k=0.7,
            mean_similarity=0.85,
            hallucination_rate=0.05,
            routing_accuracy=0.95,
            mean_latency_ms=300.0,
            pass_rate=0.9,
        )
        engine = AlertEngine()
        alerts = engine.check(metrics)
        assert alerts == []

    def test_alerts_fire_for_all_failing_metrics(self) -> None:
        from rag_eval.models.results import AggregateMetrics

        metrics = AggregateMetrics(
            sample_count=10,
            mean_precision_at_k=0.3,  # below 0.60
            mean_recall_at_k=0.3,     # below 0.60
            mean_similarity=0.5,      # below 0.75
            hallucination_rate=0.4,   # above 0.15
            routing_accuracy=0.5,     # below 0.90
            mean_latency_ms=300.0,
            pass_rate=0.1,
        )
        engine = AlertEngine()
        alerts = engine.check(metrics)
        assert len(alerts) == 5

    def test_only_failing_metrics_produce_alerts(self) -> None:
        from rag_eval.models.results import AggregateMetrics

        metrics = AggregateMetrics(
            sample_count=10,
            mean_precision_at_k=0.3,  # FAIL
            mean_recall_at_k=0.8,     # pass
            mean_similarity=0.85,     # pass
            hallucination_rate=0.05,  # pass
            routing_accuracy=0.95,    # pass
            mean_latency_ms=300.0,
            pass_rate=0.9,
        )
        engine = AlertEngine()
        alerts = engine.check(metrics)
        assert len(alerts) == 1
        assert "Precision" in alerts[0]

    def test_custom_thresholds(self) -> None:
        from rag_eval.models.results import AggregateMetrics

        # Very strict thresholds — everything fails
        strict = MetricThresholds(
            min_precision_at_k=0.99,
            min_recall_at_k=0.99,
            min_similarity=0.99,
            max_hallucination_rate=0.01,
            min_routing_accuracy=0.99,
        )
        metrics = AggregateMetrics(
            sample_count=10,
            mean_precision_at_k=0.8,
            mean_recall_at_k=0.8,
            mean_similarity=0.8,
            hallucination_rate=0.1,
            routing_accuracy=0.8,
            mean_latency_ms=300.0,
            pass_rate=0.5,
        )
        engine = AlertEngine(strict)
        alerts = engine.check(metrics)
        assert len(alerts) == 5


# ── JSON reporter integration ─────────────────────────────────────────────────


class TestJSONReporterIntegration:
    def test_save_and_load_round_trip(self, sample_eval_report: EvalReport) -> None:
        """Save an EvalReport to disk and reload it — should be identical."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = JSONReporter(reports_dir=tmpdir)
            path = reporter.save(sample_eval_report)

            assert path.exists()
            loaded = reporter.load(path)

        assert loaded.run_id == sample_eval_report.run_id
        assert loaded.aggregate.sample_count == sample_eval_report.aggregate.sample_count
        assert len(loaded.samples) == len(sample_eval_report.samples)

    def test_saved_file_is_valid_json(self, sample_eval_report: EvalReport) -> None:
        """The saved file should be parseable JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = JSONReporter(reports_dir=tmpdir)
            path = reporter.save(sample_eval_report)

            with path.open() as f:
                data = json.load(f)  # should not raise

        assert "run_id" in data
        assert "aggregate" in data
        assert "samples" in data
        assert "alerts" in data

    def test_report_filename_uses_run_id(self, sample_eval_report: EvalReport) -> None:
        """Report filename should include the run_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = JSONReporter(reports_dir=tmpdir)
            path = reporter.save(sample_eval_report)

        assert sample_eval_report.run_id in path.name

    def test_datetime_serialized_as_iso_string(self, sample_eval_report: EvalReport) -> None:
        """Datetime fields should serialize to ISO 8601 strings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = JSONReporter(reports_dir=tmpdir)
            path = reporter.save(sample_eval_report)

            with path.open() as f:
                data = json.load(f)

        # timestamp should be a string, not a dict
        assert isinstance(data["timestamp"], str)


# ── Document loader integration ───────────────────────────────────────────────


class TestDocumentLoaderIntegration:
    def test_load_documents_from_real_data_dir(self) -> None:
        """Load the actual knowledge base documents."""
        from rag_eval.ingest.loader import load_documents_from_dir

        docs = load_documents_from_dir("data/documents")
        assert len(docs) >= 10
        for doc in docs:
            assert "doc_id" in doc.metadata
            assert len(doc.page_content) > 50

    def test_documents_have_required_metadata(self) -> None:
        """Each document should have doc_id, title, category."""
        from rag_eval.ingest.loader import load_documents_from_dir

        docs = load_documents_from_dir("data/documents")
        for doc in docs:
            assert doc.metadata.get("doc_id"), f"Missing doc_id in {doc.metadata}"
            assert doc.metadata.get("title"), f"Missing title in {doc.metadata}"


# ── Golden dataset integration ────────────────────────────────────────────────


class TestGoldenDatasetIntegration:
    def test_load_real_golden_dataset(self) -> None:
        """Load and validate the actual golden dataset file."""
        with open("data/golden_dataset.json") as f:
            data = json.load(f)
        dataset = GoldenDataset.model_validate(data)
        assert len(dataset) == 31  # 31 samples: 3 git, 3 docker, 3 python, 3 cicd-001, 3 cicd-002, 3 testing, 4 api

    def test_all_relevant_doc_ids_exist_in_documents(self) -> None:
        """Every relevant_doc_id referenced in the dataset should match a document."""
        import os
        from rag_eval.ingest.loader import load_documents_from_dir

        with open("data/golden_dataset.json") as f:
            data = json.load(f)
        dataset = GoldenDataset.model_validate(data)

        docs = load_documents_from_dir("data/documents")
        available_doc_ids = {doc.metadata["doc_id"] for doc in docs}

        for sample in dataset.samples:
            for doc_id in sample.relevant_doc_ids:
                assert doc_id in available_doc_ids, (
                    f"Sample {sample.id} references doc_id '{doc_id}' "
                    f"which is not in data/documents/"
                )
