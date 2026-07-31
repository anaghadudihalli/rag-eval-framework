"""Eval runner — orchestrates the full evaluation loop.

Iterates over every sample in the GoldenDataset, runs the RAG pipeline,
computes all four metrics, assembles SampleEvalResult objects, and produces
a final EvalReport with aggregates and alerts.

Can be run as a module: `python -m rag_eval.pipeline.evaluator`
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from rag_eval.config import get_settings
from rag_eval.metrics.confidence_routing import ConfidenceRouter
from rag_eval.metrics.hallucination import HallucinationEvaluator
from rag_eval.metrics.retrieval import RetrievalEvaluator
from rag_eval.metrics.similarity import SimilarityEvaluator
from rag_eval.models.dataset import GoldenDataset, GoldenSample
from rag_eval.models.results import (
    AggregateMetrics,
    EvalReport,
    SampleEvalResult,
)
from rag_eval.models.thresholds import MetricThresholds
from rag_eval.pipeline.rag_chain import RAGChain
from rag_eval.reporting.alerts import AlertEngine
from rag_eval.reporting.console import ConsoleReporter
from rag_eval.reporting.json_reporter import JSONReporter
from rag_eval.store.factory import get_vector_store

logger = logging.getLogger(__name__)
console = Console()


def _get_git_sha() -> str | None:
    """Return the current git commit SHA for audit trail in reports."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _load_golden_dataset(path: str) -> GoldenDataset:
    """Load and validate the golden dataset from JSON."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return GoldenDataset.model_validate(data)


class EvalRunner:
    """Orchestrates the full evaluation pipeline.

    Args:
        thresholds: MetricThresholds for pass/fail decisions.
        settings: Optional Settings override for testing.
    """

    def __init__(
        self,
        thresholds: MetricThresholds | None = None,
        settings=None,
    ) -> None:
        self._settings = settings or get_settings()
        self._thresholds = thresholds or MetricThresholds()

        # Initialize metric evaluators once (model loading is expensive)
        self._retrieval_eval = RetrievalEvaluator()
        self._similarity_eval = SimilarityEvaluator()
        self._hallucination_eval = HallucinationEvaluator()
        self._confidence_router = ConfidenceRouter()

    def _evaluate_sample(
        self,
        sample: GoldenSample,
        rag_chain: RAGChain,
    ) -> SampleEvalResult:
        """Run the full eval pipeline for a single golden sample.

        Steps:
        1. Run RAG chain → answer + retrieved docs + logprobs + latency
        2. Compute retrieval metrics (precision@K, recall@K)
        3. Compute semantic similarity
        4. Evaluate hallucination via LLM judge
        5. Compute confidence routing
        6. Assemble SampleEvalResult
        """
        # Step 1: Run RAG
        answer, retrieved_docs, logprobs, latency_ms = rag_chain.run(sample.query)

        # Extract doc IDs from retrieved chunks' metadata
        retrieved_ids = [
            doc.metadata.get("doc_id", f"unknown-{i}")
            for i, doc in enumerate(retrieved_docs)
        ]

        # Step 2: Retrieval metrics
        retrieval_metrics = self._retrieval_eval.evaluate(
            retrieved_ids=retrieved_ids,
            relevant_ids=sample.relevant_doc_ids,
            k=self._settings.top_k,
        )

        # Step 3: Semantic similarity
        similarity_metrics = self._similarity_eval.compute(
            generated=answer,
            ground_truth=sample.ground_truth_answer,
        )

        # Step 4: Hallucination detection
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        hallucination_result = self._hallucination_eval.evaluate(
            query=sample.query,
            context=context_text,
            generated_answer=answer,
        )

        # Step 5: Confidence routing
        if logprobs:
            routing_result = self._confidence_router.evaluate_from_logprobs(logprobs)
        else:
            # Fallback: use similarity as confidence proxy
            routing_result = self._confidence_router.evaluate_from_similarity(
                similarity_metrics.cosine_similarity
            )

        return SampleEvalResult(
            sample_id=sample.id,
            query=sample.query,
            ground_truth_answer=sample.ground_truth_answer,
            generated_answer=answer,
            retrieved_doc_ids=retrieved_ids,
            retrieval=retrieval_metrics,
            similarity=similarity_metrics,
            hallucination=hallucination_result,
            routing=routing_result,
            latency_ms=round(latency_ms, 2),
        )

    def _compute_aggregates(self, results: list[SampleEvalResult]) -> AggregateMetrics:
        """Compute mean metrics across all sample results."""
        n = len(results)
        if n == 0:
            raise ValueError("Cannot compute aggregates from empty results list.")

        return AggregateMetrics(
            sample_count=n,
            mean_precision_at_k=round(
                sum(r.retrieval.precision_at_k for r in results) / n, 4
            ),
            mean_recall_at_k=round(
                sum(r.retrieval.recall_at_k for r in results) / n, 4
            ),
            mean_similarity=round(
                sum(r.similarity.cosine_similarity for r in results) / n, 4
            ),
            hallucination_rate=round(
                sum(1 for r in results if r.hallucination.is_hallucination) / n, 4
            ),
            routing_accuracy=round(
                sum(1 for r in results if r.routing.routing_correct) / n, 4
            ),
            mean_latency_ms=round(
                sum(r.latency_ms for r in results) / n, 2
            ),
            pass_rate=round(
                sum(1 for r in results if r.passed) / n, 4
            ),
        )

    def run(self) -> EvalReport:
        """Execute the full evaluation pipeline and return the EvalReport.

        Loads the golden dataset, builds the RAG chain, evaluates every sample,
        computes aggregates, fires alerts, and saves the report.
        """
        settings = self._settings
        console.rule("[bold cyan]RAG Evaluation Pipeline[/bold cyan]")

        # Load dataset
        console.print(f"Loading golden dataset: [cyan]{settings.golden_dataset_path}[/cyan]")
        dataset = _load_golden_dataset(settings.golden_dataset_path)
        console.print(f"[green]✓ Loaded {len(dataset)} samples[/green]")

        # Build vector store and RAG chain
        console.print("Connecting to vector store...")
        store = get_vector_store(settings)
        retriever = store.as_retriever(k=settings.top_k)
        rag_chain = RAGChain(
            retriever=retriever,
            model_name=settings.judge_model,
        )
        console.print(f"[green]✓ Vector store ready ({settings.vector_store_backend})[/green]")

        # Evaluate all samples with a rich progress bar
        results: list[SampleEvalResult] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Evaluating samples...", total=len(dataset.samples)
            )
            for sample in dataset.samples:
                progress.update(task, description=f"[cyan]Evaluating: {sample.id}")
                try:
                    result = self._evaluate_sample(sample, rag_chain)
                    results.append(result)
                except Exception as exc:
                    logger.error("Failed to evaluate sample %s: %s", sample.id, exc)
                    console.print(f"[yellow]⚠ Skipped sample {sample.id}: {exc}[/yellow]")
                finally:
                    progress.advance(task)

        if not results:
            raise RuntimeError("All samples failed evaluation. Cannot produce report.")

        # Compute aggregates and alerts
        aggregates = self._compute_aggregates(results)
        alert_engine = AlertEngine(self._thresholds)
        alerts = alert_engine.check(aggregates)

        # Assemble report
        report = EvalReport(
            git_sha=_get_git_sha(),
            vector_store_backend=settings.vector_store_backend,
            embedding_model=settings.embedding_model,
            judge_model=settings.judge_model,
            top_k=settings.top_k,
            aggregate=aggregates,
            samples=results,
            alerts=alerts,
        )

        # Save JSON report
        Path(settings.reports_dir).mkdir(parents=True, exist_ok=True)
        json_reporter = JSONReporter(reports_dir=settings.reports_dir)
        report_path = json_reporter.save(report)
        console.print(f"[green]✓ Report saved: {report_path}[/green]")

        # Render console report
        reporter = ConsoleReporter(console=console, thresholds=self._thresholds)
        reporter.render(report)

        return report


def main() -> None:
    """CLI entry point for `python -m rag_eval.pipeline.evaluator`."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run RAG evaluation pipeline.")
    parser.add_argument(
        "--fail-on-alert",
        action="store_true",
        help="Exit with code 1 if any metric alert fires (for CI/CD).",
    )
    args = parser.parse_args()

    runner = EvalRunner()
    try:
        report = runner.run()
        if args.fail_on_alert and report.has_alerts:
            console.print("\n[bold red]❌ Eval FAILED — metric thresholds breached.[/bold red]")
            sys.exit(1)
        else:
            console.print("\n[bold green]✅ Eval PASSED[/bold green]")
            sys.exit(0)
    except Exception as exc:
        console.print(f"\n[bold red]Fatal error: {exc}[/bold red]")
        logger.exception("Eval runner crashed")
        sys.exit(1)


if __name__ == "__main__":
    main()
