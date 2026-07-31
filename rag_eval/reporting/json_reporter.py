"""JSON report serialization.

Saves EvalReport to reports/<run_id>.json with full Pydantic v2 model
serialization including datetime ISO formatting.

Can also be run as a module to pretty-print an existing report:
    python -m rag_eval.reporting.json_reporter --file reports/<id>.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.syntax import Syntax

from rag_eval.models.results import EvalReport

console = Console()


class JSONReporter:
    """Serializes EvalReport to disk as a timestamped JSON file.

    Args:
        reports_dir: Directory to write report files to.
    """

    def __init__(self, reports_dir: str = "reports") -> None:
        self._dir = Path(reports_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, report: EvalReport) -> Path:
        """Serialize the report and write to disk.

        Args:
            report: The completed EvalReport.

        Returns:
            Path to the written JSON file.
        """
        filename = f"{report.run_id}.json"
        output_path = self._dir / filename

        # model_dump with mode="json" ensures datetime → ISO string serialization
        report_dict = report.model_dump(mode="json")

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        return output_path

    def load(self, path: Path | str) -> EvalReport:
        """Load and validate a previously saved report.

        Args:
            path: Path to the JSON report file.

        Returns:
            EvalReport parsed and validated by Pydantic.
        """
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return EvalReport.model_validate(data)


def main() -> None:
    """Pretty-print a JSON report file to the terminal."""
    import argparse

    parser = argparse.ArgumentParser(description="Display a saved eval report.")
    parser.add_argument("--file", required=True, help="Path to the report JSON file.")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        sys.exit(1)

    with path.open() as f:
        content = f.read()

    syntax = Syntax(content, "json", theme="monokai", line_numbers=True)
    console.print(syntax)


if __name__ == "__main__":
    main()
