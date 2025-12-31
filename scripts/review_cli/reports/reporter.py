"""Report generation for code review results."""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from scripts.review_cli.providers.base import ReviewResult

logger = logging.getLogger(__name__)


class Reporter:
    """Generates reports from code review results."""

    def __init__(
        self, output_file: str = "review_report.json", project_root: Path | None = None
    ):
        self.output_file = output_file
        self.project_root = project_root or Path.cwd()
        self.console = Console()

    def save_json(self, results: list[ReviewResult], summary: dict) -> Path:
        """Save results to a JSON file."""
        # Deduplicate issues by description
        all_issues = []
        seen_descriptions: set[str] = set()

        for result in results:
            if not result.is_success:
                continue
            for issue in result.issues:
                desc_key = f"{result.file}:{issue.get('description', '')[:100]}"
                if desc_key not in seen_descriptions:
                    seen_descriptions.add(desc_key)
                    all_issues.append(
                        {
                            "file": result.file,
                            "category": issue.get("category", "UNKNOWN"),
                            "line": issue.get("line"),
                            "description": issue.get("description", ""),
                            "language": result.language,
                            "found_by": result.model,
                        }
                    )

        results_summary = []
        for result in results:
            if result.skipped:
                status = "skipped"
            elif result.error:
                status = "failed"
            else:
                status = "success"
            results_summary.append(
                {
                    "file": result.file,
                    "status": status,
                    "issues_count": len(result.issues),
                    "summary": result.summary,
                    "model": result.model,
                    "language": result.language,
                    "error": result.error,
                    "skip_reason": result.skip_reason,
                }
            )

        report_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "summary": summary,
            "issues": all_issues,
            "results": results_summary,
        }

        report_path = self.project_root / self.output_file
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        logger.info("Report saved to %s", report_path)
        return report_path

    def print_summary(self, results: list[ReviewResult], summary: dict) -> None:
        """Print a summary to the console."""
        self.console.print()
        self.console.rule("[bold blue]Code Review Summary")

        # Summary table
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Files Reviewed", str(summary.get("reviewed", 0)))
        table.add_row("Skipped", str(summary.get("skipped", 0)))
        table.add_row("Failed", str(summary.get("failed", 0)))
        table.add_row("Files with Issues", str(summary.get("files_with_issues", 0)))
        table.add_row("Total Issues", str(summary.get("total_issues", 0)))

        self.console.print(table)

        # Issues by category
        if summary.get("total_issues", 0) > 0:
            self.console.print()
            self.console.rule("[bold yellow]Issues by Category")

            # Collect all issues
            category_counts: dict[str, int] = {}
            for result in results:
                if result.is_success:
                    for issue in result.issues:
                        cat = issue.get("category", "UNKNOWN")
                        category_counts[cat] = category_counts.get(cat, 0) + 1

            cat_table = Table(show_header=True)
            cat_table.add_column("Category", style="magenta")
            cat_table.add_column("Count", style="yellow", justify="right")

            for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
                cat_table.add_row(cat, str(count))

            self.console.print(cat_table)

            # Show sample issues
            self.console.print()
            self.console.rule("[bold]Sample Issues (max 10)")

            shown = 0
            for result in results:
                if not result.is_success:
                    continue
                for issue in result.issues:
                    if shown >= 10:
                        break
                    line_str = f":L{issue.get('line')}" if issue.get("line") else ""
                    self.console.print(
                        f"  [{issue.get('category', '?')[:4]}] "
                        f"[cyan]{result.file}{line_str}[/cyan]: "
                        f"{issue.get('description', '')[:80]}"
                    )
                    shown += 1
                if shown >= 10:
                    break
        else:
            self.console.print()
            self.console.print("[bold green]✅ No issues found!")

        self.console.print()
