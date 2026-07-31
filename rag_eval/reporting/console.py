"""Rich terminal report renderer.

Produces a structured, color-coded terminal report after each eval run:
- Header panel with run metadata
- Summary table: metric | value | threshold | status
- Hallucination detail panel for any flagged samples
- Alert banners in red/yellow

All output goes through a Rich Console for consistent formatting.
"""

from __future__ import annotations

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rag_eval.models.results import EvalReport, SampleEvalResult
from rag_eval.models.thresholds import MetricThresholds


def _status_icon(passed: bool) -> str:
    return "[bold green]✅ PASS[/bold green]" if passed else "[bold red]❌ FAIL[/bold red]"


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


class ConsoleReporter:
    """Renders a rich terminal eval report.

    Args:
        console: Rich Console instance (defaults to a new Console).
        thresholds: MetricThresholds for determining pass/fail coloring.
    """

    def __init__(
        self,
        console: Console | None = None,
        thresholds: MetricThresholds | None = None,
    ) -> None:
        self._console = console or Console()
        self._thresholds = thresholds or MetricThresholds()

    def render(self, report: EvalReport) -> None:
        """Render the complete eval report to the terminal."""
        self._render_header(report)
        self._render_summary_table(report)
        self._render_hallucinations(report)
        self._render_alerts(report)
        self._render_footer(report)

    def _render_header(self, report: EvalReport) -> None:
        """Run metadata panel."""
        meta = Table.grid(padding=(0, 2))
        meta.add_column(style="dim")
        meta.add_column()

        meta.add_row("Run ID:", report.run_id)
        meta.add_row("Timestamp:", report.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"))
        meta.add_row("Git SHA:", report.git_sha or "N/A")
        meta.add_row("Vector Store:", report.vector_store_backend)
        meta.add_row("Embedding Model:", report.embedding_model)
        meta.add_row("Judge Model:", report.judge_model)
        meta.add_row("Top-K:", str(report.top_k))
        meta.add_row("Samples:", str(report.aggregate.sample_count))

        self._console.print()
        self._console.print(Panel(meta, title="[bold cyan]📊 RAG Eval Report[/bold cyan]", border_style="cyan"))

    def _render_summary_table(self, report: EvalReport) -> None:
        """Main metrics summary table."""
        agg = report.aggregate
        t = self._thresholds

        table = Table(
            title="Aggregate Metrics",
            show_header=True,
            header_style="bold magenta",
            border_style="dim",
            expand=False,
        )
        table.add_column("Metric", style="bold", min_width=22)
        table.add_column("Value", justify="right", min_width=10)
        table.add_column("Threshold", justify="right", min_width=12, style="dim")
        table.add_column("Status", justify="center", min_width=12)

        rows = [
            (
                "Precision@K",
                _format_pct(agg.mean_precision_at_k),
                f">= {_format_pct(t.min_precision_at_k)}",
                agg.mean_precision_at_k >= t.min_precision_at_k,
            ),
            (
                "Recall@K",
                _format_pct(agg.mean_recall_at_k),
                f">= {_format_pct(t.min_recall_at_k)}",
                agg.mean_recall_at_k >= t.min_recall_at_k,
            ),
            (
                "Semantic Similarity",
                _format_pct(agg.mean_similarity),
                f">= {_format_pct(t.min_similarity)}",
                agg.mean_similarity >= t.min_similarity,
            ),
            (
                "Hallucination Rate",
                _format_pct(agg.hallucination_rate),
                f"<= {_format_pct(t.max_hallucination_rate)}",
                agg.hallucination_rate <= t.max_hallucination_rate,
            ),
            (
                "Routing Accuracy",
                _format_pct(agg.routing_accuracy),
                f">= {_format_pct(t.min_routing_accuracy)}",
                agg.routing_accuracy >= t.min_routing_accuracy,
            ),
        ]

        for metric, value, threshold, passed in rows:
            value_style = "green" if passed else "red bold"
            table.add_row(
                metric,
                Text(value, style=value_style),
                threshold,
                _status_icon(passed),
            )

        # Separator row
        table.add_section()
        table.add_row(
            "Overall Pass Rate",
            Text(_format_pct(agg.pass_rate), style="bold"),
            "",
            _status_icon(not report.has_alerts),
        )
        table.add_row(
            "Mean Latency",
            f"{agg.mean_latency_ms:.1f} ms",
            "",
            "",
        )

        self._console.print()
        self._console.print(table)

    def _render_hallucinations(self, report: EvalReport) -> None:
        """Show details for any hallucinated samples."""
        hallucinated = [
            r for r in report.samples if r.hallucination.is_hallucination
        ]
        if not hallucinated:
            self._console.print(
                "\n[green]✅ No hallucinations detected across all samples.[/green]"
            )
            return

        self._console.print(
            f"\n[bold red]⚠ {len(hallucinated)} Hallucination(s) Detected[/bold red]"
        )
        for result in hallucinated:
            self._console.print(
                Panel(
                    f"[bold]Query:[/bold] {result.query}\n"
                    f"[bold]Generated:[/bold] {result.generated_answer[:200]}...\n"
                    f"[bold]Judge Reasoning:[/bold] {result.hallucination.judge_reasoning}\n"
                    f"[bold]Judge Confidence:[/bold] {result.hallucination.judge_confidence:.2f}",
                    title=f"[red]🚨 Hallucination — Sample {result.sample_id}[/red]",
                    border_style="red",
                )
            )

    def _render_alerts(self, report: EvalReport) -> None:
        """Render alert banners for threshold violations."""
        if not report.alerts:
            return
        self._console.print()
        for alert in report.alerts:
            self._console.print(
                Panel(
                    f"[bold]{alert}[/bold]",
                    title="[bold red]🔔 ALERT[/bold red]",
                    border_style="red",
                )
            )

    def _render_footer(self, report: EvalReport) -> None:
        """Final pass/fail banner."""
        self._console.print()
        if report.has_alerts:
            self._console.rule("[bold red]❌ EVAL FAILED — Metric thresholds breached[/bold red]")
        else:
            self._console.rule("[bold green]✅ EVAL PASSED — All metrics within thresholds[/bold green]")
        self._console.print()
