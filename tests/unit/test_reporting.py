"""Unit tests for the reporting module.

Covers:
- AlertEngine.check(): all threshold breach combinations
- ConsoleReporter.render(): smoke tests — no crashes, output contains expected strings
- JSONReporter.save() / load(): file creation, round-trip, schema integrity
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

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
from rag_eval.reporting.alerts import AlertEngine
from rag_eval.reporting.console import ConsoleReporter
from rag_eval.reporting.json_reporter import JSONReporter


# ── Shared fixtures ──────────────────────────────────────────────────────────


def _make_aggregate(
    precision: float = 0.80,
    recall: float = 0.80,
    similarity: float = 0.85,
    hallucination_rate: float = 0.05,
    routing_accuracy: float = 0.95,
    mean_latency_ms: float = 250.0,
    pass_rate: float = 0.90,
    sample_count: int = 5,
) -> AggregateMetrics:
    return AggregateMetrics(
        sample_count=sample_count,
        mean_precision_at_k=precision,
        mean_recall_at_k=recall,
        mean_similarity=similarity,
        hallucination_rate=hallucination_rate,
        routing_accuracy=routing_accuracy,
        mean_latency_ms=mean_latency_ms,
        pass_rate=pass_rate,
    )


def _make_sample_result(
    sample_id: str = "test-001",
    is_hallucination: bool = False,
    similarity: float = 0.90,
    routing_correct: bool = True,
) -> SampleEvalResult:
    return SampleEvalResult(
        sample_id=sample_id,
        query="What is Docker?",
        ground_truth_answer="Docker is a containerization platform.",
        generated_answer="Docker uses OS-level virtualization for containers.",
        retrieved_doc_ids=["docker-001"],
        retrieval=RetrievalMetrics(
            precision_at_k=0.8,
            recall_at_k=1.0,
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
            judge_reasoning="Answer is fully supported by context.",
            judge_confidence=0.92,
        ),
        routing=ConfidenceRoutingResult(
            answer_confidence=0.85,
            confidence_source="logprobs",
            routed_to_human=False,
            routing_correct=routing_correct,
            threshold_used=0.7,
        ),
        latency_ms=300.0,
    )


def _make_report(
    alerts: list[str] | None = None,
    samples: list[SampleEvalResult] | None = None,
    aggregate: AggregateMetrics | None = None,
) -> EvalReport:
    return EvalReport(
        run_id="test-run-abc123",
        timestamp=datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        git_sha="abc1234",
        vector_store_backend="chroma",
        embedding_model="all-MiniLM-L6-v2",
        judge_model="gpt-3.5-turbo",
        top_k=5,
        aggregate=aggregate or _make_aggregate(),
        samples=samples or [_make_sample_result()],
        alerts=alerts or [],
    )


# ── AlertEngine ──────────────────────────────────────────────────────────────


class TestAlertEngine:
    def _engine(self, **threshold_overrides) -> AlertEngine:
        return AlertEngine(MetricThresholds(**threshold_overrides))

    def test_no_alerts_when_all_metrics_pass(self) -> None:
        engine = self._engine()
        metrics = _make_aggregate(
            precision=0.80, recall=0.80, similarity=0.85,
            hallucination_rate=0.05, routing_accuracy=0.95
        )
        alerts = engine.check(metrics)
        assert alerts == []

    def test_precision_alert_fires_when_below_threshold(self) -> None:
        engine = self._engine(min_precision_at_k=0.70)
        metrics = _make_aggregate(precision=0.50)
        alerts = engine.check(metrics)
        assert any("Precision" in a for a in alerts)

    def test_precision_alert_does_not_fire_when_at_threshold(self) -> None:
        engine = self._engine(min_precision_at_k=0.70)
        metrics = _make_aggregate(precision=0.70)
        alerts = engine.check(metrics)
        assert not any("Precision" in a for a in alerts)

    def test_recall_alert_fires_when_below_threshold(self) -> None:
        engine = self._engine(min_recall_at_k=0.70)
        metrics = _make_aggregate(recall=0.40)
        alerts = engine.check(metrics)
        assert any("Recall" in a for a in alerts)

    def test_similarity_alert_fires_when_below_threshold(self) -> None:
        engine = self._engine(min_similarity=0.80)
        metrics = _make_aggregate(similarity=0.60)
        alerts = engine.check(metrics)
        assert any("Similarity" in a for a in alerts)

    def test_hallucination_alert_fires_when_above_threshold(self) -> None:
        engine = self._engine(max_hallucination_rate=0.10)
        metrics = _make_aggregate(hallucination_rate=0.30)
        alerts = engine.check(metrics)
        assert any("Hallucination" in a for a in alerts)

    def test_hallucination_alert_does_not_fire_when_at_threshold(self) -> None:
        engine = self._engine(max_hallucination_rate=0.15)
        metrics = _make_aggregate(hallucination_rate=0.15)
        alerts = engine.check(metrics)
        assert not any("Hallucination" in a for a in alerts)

    def test_routing_alert_fires_when_below_threshold(self) -> None:
        engine = self._engine(min_routing_accuracy=0.90)
        metrics = _make_aggregate(routing_accuracy=0.70)
        alerts = engine.check(metrics)
        assert any("Routing" in a for a in alerts)

    def test_multiple_alerts_fire_simultaneously(self) -> None:
        engine = self._engine(
            min_precision_at_k=0.90,
            min_recall_at_k=0.90,
            max_hallucination_rate=0.01,
        )
        metrics = _make_aggregate(
            precision=0.50,
            recall=0.50,
            hallucination_rate=0.30,
        )
        alerts = engine.check(metrics)
        assert len(alerts) >= 3

    def test_alert_message_contains_actual_value(self) -> None:
        engine = self._engine(min_precision_at_k=0.90)
        metrics = _make_aggregate(precision=0.45)
        alerts = engine.check(metrics)
        assert any("45.0%" in a for a in alerts)

    def test_alert_message_contains_threshold_value(self) -> None:
        engine = self._engine(min_precision_at_k=0.90)
        metrics = _make_aggregate(precision=0.45)
        alerts = engine.check(metrics)
        assert any("90.0%" in a for a in alerts)

    def test_returns_list_type(self) -> None:
        engine = self._engine()
        result = engine.check(_make_aggregate())
        assert isinstance(result, list)

    def test_uses_default_thresholds_when_none_provided(self) -> None:
        engine = AlertEngine()  # no thresholds arg
        metrics = _make_aggregate(
            precision=0.80, recall=0.80, similarity=0.85,
            hallucination_rate=0.05, routing_accuracy=0.95,
        )
        alerts = engine.check(metrics)
        assert isinstance(alerts, list)


# ── ConsoleReporter ──────────────────────────────────────────────────────────


class TestConsoleReporter:
    def _capture_output(self, report: EvalReport, **threshold_overrides) -> str:
        """Render report to a string buffer and return the captured text."""
        buffer = StringIO()
        console = Console(file=buffer, highlight=False, markup=False)
        reporter = ConsoleReporter(
            console=console,
            thresholds=MetricThresholds(**threshold_overrides) if threshold_overrides else None,
        )
        reporter.render(report)
        return buffer.getvalue()

    def test_render_does_not_raise(self) -> None:
        report = _make_report()
        output = self._capture_output(report)
        assert output  # non-empty output

    def test_output_contains_run_id(self) -> None:
        report = _make_report()
        output = self._capture_output(report)
        assert "test-run-abc123" in output

    def test_output_contains_sample_count(self) -> None:
        report = _make_report(aggregate=_make_aggregate(sample_count=10))
        output = self._capture_output(report)
        assert "10" in output

    def test_output_contains_metric_names(self) -> None:
        report = _make_report()
        output = self._capture_output(report)
        for metric in ["Precision", "Recall", "Similarity", "Hallucination", "Routing"]:
            assert metric in output, f"'{metric}' not found in console output"

    def test_output_contains_vector_store_backend(self) -> None:
        report = _make_report()
        output = self._capture_output(report)
        assert "chroma" in output

    def test_output_contains_git_sha(self) -> None:
        report = _make_report()
        output = self._capture_output(report)
        assert "abc1234" in output

    def test_hallucination_detail_shown_when_flagged(self) -> None:
        hallucinated_sample = _make_sample_result("hall-001", is_hallucination=True)
        report = _make_report(
            samples=[hallucinated_sample],
            aggregate=_make_aggregate(hallucination_rate=1.0),
        )
        output = self._capture_output(report)
        assert "Hallucination" in output

    def test_no_hallucination_message_when_clean(self) -> None:
        clean_sample = _make_sample_result("clean-001", is_hallucination=False)
        report = _make_report(samples=[clean_sample])
        output = self._capture_output(report)
        assert "No hallucinations" in output

    def test_alert_rendered_when_present(self) -> None:
        report = _make_report(alerts=["Precision@K BELOW threshold: 40.0% < 60.0%"])
        output = self._capture_output(report)
        assert "Precision@K BELOW threshold" in output

    def test_no_alert_rendered_when_none(self) -> None:
        report = _make_report(alerts=[])
        output = self._capture_output(report)
        # Should NOT contain the alert banner header
        assert "ALERT" not in output

    def test_pass_footer_shown_when_no_alerts(self) -> None:
        report = _make_report(alerts=[])
        output = self._capture_output(report)
        assert "PASSED" in output

    def test_fail_footer_shown_when_alerts_present(self) -> None:
        report = _make_report(alerts=["Some metric alert"])
        output = self._capture_output(report)
        assert "FAILED" in output

    def test_render_with_multiple_samples(self) -> None:
        samples = [_make_sample_result(f"s-{i:03d}") for i in range(5)]
        report = _make_report(samples=samples, aggregate=_make_aggregate(sample_count=5))
        # Should not raise with multiple samples
        output = self._capture_output(report)
        assert output

    def test_render_with_none_git_sha(self) -> None:
        report = EvalReport(
            run_id="test-run",
            git_sha=None,
            vector_store_backend="chroma",
            embedding_model="all-MiniLM-L6-v2",
            judge_model="gpt-3.5-turbo",
            top_k=5,
            aggregate=_make_aggregate(),
            samples=[_make_sample_result()],
        )
        output = self._capture_output(report)
        assert "N/A" in output


# ── JSONReporter ─────────────────────────────────────────────────────────────


class TestJSONReporter:
    def test_save_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = JSONReporter(reports_dir=tmpdir)
            report = _make_report()
            path = reporter.save(report)
            assert path.exists()

    def test_saved_filename_uses_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = JSONReporter(reports_dir=tmpdir)
            report = _make_report()
            path = reporter.save(report)
            assert path.stem == report.run_id

    def test_saved_file_is_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = JSONReporter(reports_dir=tmpdir)
            path = reporter.save(_make_report())
            with path.open() as f:
                data = json.load(f)
            assert isinstance(data, dict)

    def test_saved_json_contains_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = JSONReporter(reports_dir=tmpdir)
            report = _make_report()
            path = reporter.save(report)
            data = json.loads(path.read_text())
            assert data["run_id"] == report.run_id

    def test_saved_json_contains_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = JSONReporter(reports_dir=tmpdir)
            path = reporter.save(_make_report())
            data = json.loads(path.read_text())
            assert "aggregate" in data
            assert "mean_precision_at_k" in data["aggregate"]

    def test_saved_json_contains_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = JSONReporter(reports_dir=tmpdir)
            path = reporter.save(_make_report())
            data = json.loads(path.read_text())
            assert "samples" in data
            assert len(data["samples"]) == 1

    def test_datetime_serialized_as_iso_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = JSONReporter(reports_dir=tmpdir)
            path = reporter.save(_make_report())
            data = json.loads(path.read_text())
            # Pydantic mode="json" serializes datetime as ISO string
            assert isinstance(data["timestamp"], str)
            assert "2026" in data["timestamp"]

    def test_load_round_trips_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = JSONReporter(reports_dir=tmpdir)
            original = _make_report()
            path = reporter.save(original)
            loaded = reporter.load(path)

            assert loaded.run_id == original.run_id
            assert loaded.aggregate.sample_count == original.aggregate.sample_count
            assert loaded.aggregate.mean_precision_at_k == original.aggregate.mean_precision_at_k
            assert len(loaded.samples) == len(original.samples)

    def test_load_preserves_alerts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = JSONReporter(reports_dir=tmpdir)
            report = _make_report(alerts=["Alert A", "Alert B"])
            path = reporter.save(report)
            loaded = reporter.load(path)
            assert loaded.alerts == ["Alert A", "Alert B"]

    def test_save_creates_reports_dir_if_missing(self) -> None:
        with tempfile.TemporaryDirectory() as base:
            new_subdir = Path(base) / "reports" / "nested"
            reporter = JSONReporter(reports_dir=str(new_subdir))
            report = _make_report()
            path = reporter.save(report)
            assert path.exists()

    def test_multiple_saves_produce_separate_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = JSONReporter(reports_dir=tmpdir)
            # Two reports with different run_ids
            r1 = EvalReport(
                run_id="run-aaa",
                vector_store_backend="chroma",
                embedding_model="all-MiniLM-L6-v2",
                judge_model="gpt-3.5-turbo",
                top_k=5,
                aggregate=_make_aggregate(),
                samples=[_make_sample_result("s-001")],
            )
            r2 = EvalReport(
                run_id="run-bbb",
                vector_store_backend="chroma",
                embedding_model="all-MiniLM-L6-v2",
                judge_model="gpt-3.5-turbo",
                top_k=5,
                aggregate=_make_aggregate(),
                samples=[_make_sample_result("s-002")],
            )
            p1 = reporter.save(r1)
            p2 = reporter.save(r2)
            assert p1 != p2
            assert p1.exists() and p2.exists()

    def test_has_alerts_field_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = JSONReporter(reports_dir=tmpdir)
            report_with_alerts = _make_report(alerts=["Some alert"])
            path = reporter.save(report_with_alerts)
            loaded = reporter.load(path)
            assert loaded.has_alerts is True
