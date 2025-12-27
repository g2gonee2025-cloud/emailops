"""
Pipeline CLI Commands.

Exposes the unified pipeline orchestration via `cortex pipeline`.
"""

import json
import logging

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


def cmd_pipeline_run(
    source_prefix: str,
    tenant_id: str = "default",
    limit: int | None = None,
    concurrency: int = 4,
    auto_embed: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    json_output: bool = False,
) -> None:
    """
    Run the unified ingestion pipeline.
    """
    # Lazy import to avoid eager config loading
    from cortex.orchestrator import PipelineOrchestrator

    # Configure logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not json_output:
        console.print("[bold blue]🚀 Cortex Unified Pipeline[/bold blue]")
        console.print(f"   Source:      [cyan]{source_prefix}[/cyan]")
        console.print(f"   Tenant:      [cyan]{tenant_id}[/cyan]")
        console.print(f"   Concurrency: [cyan]{concurrency}[/cyan]")
        console.print(f"   Auto-Embed:  [cyan]{auto_embed}[/cyan]")
        if dry_run:
            console.print("   [yellow]DRY RUN MODE - No changes will be made[/yellow]")
            if auto_embed:
                console.print("   [dim](--auto-embed ignored in dry-run mode)[/dim]")
        console.print()

    orchestrator = PipelineOrchestrator(
        tenant_id=tenant_id,
        auto_embed=auto_embed,
        concurrency=concurrency,
        dry_run=dry_run,
    )

    # Run the pipeline (now enqueues jobs)
    stats = orchestrator.run(source_prefix=source_prefix, limit=limit)

    # Output results
    if not json_output:
        console.print()
        table = Table(title="Pipeline Enqueueing Results")
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="bold")

        table.add_row("Duration", f"{stats.duration_seconds:.2f}s")
        table.add_row("Folders Found", str(stats.folders_found))
        table.add_row("Jobs Enqueued", f"[green]{stats.folders_processed}[/green]")
        table.add_row("Failed to Enqueue", f"[red]{stats.folders_failed}[/red]")

        console.print(table)

        if stats.folders_failed > 0:
            console.print(
                f"\n[yellow]⚠ {stats.folders_failed} folder(s) failed to enqueue. Run with --verbose for details.[/yellow]"
            )
        elif stats.folders_processed > 0:
            console.print(
                f"\n[green]✓ {stats.folders_processed} ingestion job(s) enqueued successfully![/green]"
            )
            console.print("[dim]  Run worker processes to handle the jobs.[/dim]")
        else:
            console.print("\n[yellow]No new folders found to process.[/yellow]")
    else:
        output = {
            "success": stats.folders_failed == 0,
            "dry_run": dry_run,
            "duration_seconds": stats.duration_seconds,
            "folders_found": stats.folders_found,
            "jobs_enqueued": stats.folders_processed,
            "enqueue_failures": stats.folders_failed,
            "errors": getattr(stats, "errors_list", []),
        }
        print(json.dumps(output))
