"""Alert engine — compares aggregate metrics against thresholds.

Generates human-readable alert messages for any metric that breaches
its threshold. The EvalRunner includes these in the EvalReport.

The CI/CD step checks report.has_alerts and exits non-zero to block
the pipeline when quality degrades.

Can also be run as a module for CI gating:
    python -m rag_eval.reporting.alerts --file reports/<id>.json --fail-on-alert
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rag_eval.models.results import AggregateMetrics
from rag_eval.models.thresholds import MetricThresholds


class AlertEngine:
    """Compares AggregateMetrics against MetricThresholds and produces alerts.

    Args:
        thresholds: The pass/fail thresholds to apply.
    """

    def __init__(self, thresholds: MetricThresholds | None = None) -> None:
        self._thresholds = thresholds or MetricThresholds()

    def check(self, metrics: AggregateMetrics) -> list[str]:
        """Check all metrics against thresholds and return alert messages.

        Returns:
            List of alert strings (empty if all metrics pass).
        """
        t = self._thresholds
        alerts: list[str] = []

        if metrics.mean_precision_at_k < t.min_precision_at_k:
            alerts.append(
                f"Precision@K BELOW threshold: "
                f"{metrics.mean_precision_at_k:.1%} < {t.min_precision_at_k:.1%}"
            )

        if metrics.mean_recall_at_k < t.min_recall_at_k:
            alerts.append(
                f"Recall@K BELOW threshold: "
                f"{metrics.mean_recall_at_k:.1%} < {t.min_recall_at_k:.1%}"
            )

        if metrics.mean_similarity < t.min_similarity:
            alerts.append(
                f"Semantic Similarity BELOW threshold: "
                f"{metrics.mean_similarity:.1%} < {t.min_similarity:.1%}"
            )

        if metrics.hallucination_rate > t.max_hallucination_rate:
            alerts.append(
                f"Hallucination Rate ABOVE threshold: "
                f"{metrics.hallucination_rate:.1%} > {t.max_hallucination_rate:.1%}"
            )

        if metrics.routing_accuracy < t.min_routing_accuracy:
            alerts.append(
                f"Routing Accuracy BELOW threshold: "
                f"{metrics.routing_accuracy:.1%} < {t.min_routing_accuracy:.1%}"
            )

        return alerts


def main() -> None:
    """CLI entry point for CI/CD alert gating."""
    import argparse
    from rich.console import Console

    console = Console()
    parser = argparse.ArgumentParser(
        description="Check a saved report against thresholds and exit non-zero if alerts fire."
    )
    parser.add_argument("--file", required=True, help="Path to the eval report JSON.")
    parser.add_argument(
        "--fail-on-alert",
        action="store_true",
        help="Exit code 1 if any alert fires.",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        console.print(f"[red]Report not found: {path}[/red]")
        sys.exit(1)

    with path.open() as f:
        data = json.load(f)

    alerts = data.get("alerts", [])
    if alerts:
        console.print(f"[bold red]⚠ {len(alerts)} alert(s) detected:[/bold red]")
        for alert in alerts:
            console.print(f"  • {alert}")
        if args.fail_on_alert:
            sys.exit(1)
    else:
        console.print("[green]✅ No alerts — all metrics within thresholds.[/green]")
        sys.exit(0)


if __name__ == "__main__":
    main()
