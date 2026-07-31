"""End-to-end smoke test with fully mocked OpenAI calls.

Verifies the complete pipeline wiring — from golden dataset load through
EvalRunner orchestration, all 4 metric evaluators, report assembly,
JSON serialization, alert engine, and console reporting — without
making any real API calls or requiring a running vector store.

Run with:
    python tests/smoke_test_mocked_eval.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from rich.console import Console

# ── Make sure the package root is on the path ───────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_eval.config import Settings
from rag_eval.models.dataset import GoldenDataset, GoldenSample
from rag_eval.models.thresholds import MetricThresholds
from rag_eval.pipeline.evaluator import EvalRunner
from rag_eval.reporting.alerts import AlertEngine
from rag_eval.reporting.console import ConsoleReporter
from rag_eval.reporting.json_reporter import JSONReporter

console = Console()


# ── Mock helpers ─────────────────────────────────────────────────────────────

def _fake_ai_message(content: str, logprob: float = -0.1) -> AIMessage:
    """Build a mock AIMessage with logprob metadata."""
    return AIMessage(
        content=content,
        response_metadata={
            "logprobs": {
                "content": [
                    {"token": t, "logprob": logprob}
                    for t in content.split()[:5]
                ]
            }
        },
    )


def _make_mock_store(docs: list[Document]) -> MagicMock:
    """Return a mock VectorStoreBackend whose retriever returns fixed docs."""
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = docs

    mock_store = MagicMock()
    mock_store.as_retriever.return_value = mock_retriever
    return mock_store


# ── Smoke test ───────────────────────────────────────────────────────────────

def run_smoke_test() -> None:
    console.rule("[bold cyan]End-to-End Smoke Test (Mocked OpenAI)[/bold cyan]")

    # 1. Settings pointing at a temp reports dir
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(
            openai_api_key="sk-test-placeholder",  # type: ignore[arg-type]
            vector_store_backend="chroma",
            top_k=3,
            confidence_threshold=0.7,
            similarity_threshold=0.75,
            golden_dataset_path="data/golden_dataset.json",
            documents_dir="data/documents",
            reports_dir=tmpdir,
        )

        # 2. Build a minimal golden dataset (3 samples to keep it fast)
        dataset = GoldenDataset(
            version="1.0",
            description="Smoke test dataset",
            samples=[
                GoldenSample(
                    id=f"smoke-{i:03d}",
                    query=f"What is Docker container isolation? ({i})",
                    ground_truth_answer=(
                        "Docker uses OS-level virtualization to isolate containers. "
                        "Each container runs in its own namespace with isolated resources."
                    ),
                    relevant_doc_ids=["docker-001"],
                    category="docker",
                )
                for i in range(3)
            ],
        )

        # 3. Fixed retrieved docs (matching relevant_doc_ids for good precision/recall)
        retrieved_docs = [
            Document(
                page_content=(
                    "Docker uses OS-level virtualization to isolate containers. "
                    "Containers share the host OS kernel but run in isolated namespaces."
                ),
                metadata={"doc_id": "docker-001"},
            ),
            Document(
                page_content="Each container has its own filesystem, network, and process space.",
                metadata={"doc_id": "docker-002"},
            ),
        ]

        # 4. Mock OpenAI LLM response
        fake_answer = (
            "Docker uses OS-level virtualization to isolate containers using namespaces."
        )
        fake_ai_msg = _fake_ai_message(fake_answer, logprob=-0.05)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_ai_msg

        mock_store = _make_mock_store(retrieved_docs)

        # 5. Patch the hallucination judge to return clean JSON
        hallucination_json = json.dumps({
            "is_hallucination": False,
            "reasoning": "Answer is fully supported by context.",
            "confidence": 0.95,
        })
        mock_judge_response = MagicMock()
        mock_judge_response.content = hallucination_json

        mock_judge_llm = MagicMock()
        mock_judge_llm.invoke.return_value = mock_judge_response

        # 6. Run the EvalRunner with patches
        with (
            patch("rag_eval.pipeline.evaluator.get_settings", return_value=settings),
            patch("rag_eval.pipeline.evaluator._load_golden_dataset", return_value=dataset),
            patch("rag_eval.pipeline.evaluator.get_vector_store", return_value=mock_store),
            patch("rag_eval.pipeline.rag_chain.ChatOpenAI", return_value=mock_llm),
            patch("rag_eval.metrics.hallucination.ChatOpenAI", return_value=mock_judge_llm),
        ):
            runner = EvalRunner(settings=settings)
            report = runner.run()

        # 7. Assertions
        console.print("\n[bold]Verifying report structure...[/bold]")

        assert report.aggregate.sample_count == 3, "Expected 3 samples"
        assert report.aggregate.mean_precision_at_k >= 0.0
        assert report.aggregate.mean_recall_at_k >= 0.0
        assert 0.0 <= report.aggregate.mean_similarity <= 1.0
        assert 0.0 <= report.aggregate.hallucination_rate <= 1.0
        assert report.aggregate.routing_accuracy == 1.0, "Routing should always be correct (deterministic)"
        assert report.aggregate.mean_latency_ms >= 0.0
        assert len(report.samples) == 3
        assert report.run_id  # non-empty UUID
        assert report.timestamp

        # Check JSON serialization round-trip
        reporter = JSONReporter(reports_dir=tmpdir)
        saved_path = reporter.save(report)
        assert saved_path.exists(), "JSON report file was not created"

        loaded = reporter.load(saved_path)
        assert loaded.run_id == report.run_id
        assert loaded.aggregate.sample_count == 3
        console.print(f"[green]✅ JSON report saved and round-tripped: {saved_path.name}[/green]")

        # Check alert engine
        alert_engine = AlertEngine(MetricThresholds())
        alerts = alert_engine.check(report.aggregate)
        console.print(f"[green]✅ AlertEngine ran — {len(alerts)} alert(s)[/green]")

        # Check console reporter renders without crashing
        console_reporter = ConsoleReporter(console=console)
        console_reporter.render(report)

        # Summary
        console.print()
        console.rule("[bold green]✅ All smoke test assertions passed[/bold green]")
        console.print(f"  Samples evaluated : {report.aggregate.sample_count}")
        console.print(f"  Mean Precision@K  : {report.aggregate.mean_precision_at_k:.3f}")
        console.print(f"  Mean Recall@K     : {report.aggregate.mean_recall_at_k:.3f}")
        console.print(f"  Mean Similarity   : {report.aggregate.mean_similarity:.3f}")
        console.print(f"  Hallucination Rate: {report.aggregate.hallucination_rate:.3f}")
        console.print(f"  Routing Accuracy  : {report.aggregate.routing_accuracy:.3f}")
        console.print(f"  Mean Latency      : {report.aggregate.mean_latency_ms:.1f} ms")
        console.print(f"  Alerts            : {len(alerts)}")
        console.print(f"  Report file       : {saved_path.name}")


if __name__ == "__main__":
    run_smoke_test()
