"""
Pipeline CLI Commands.

Exposes the unified pipeline orchestration via `cortex pipeline`.
"""

import json
import logging
import sys
import traceback

from rich.console import Console
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
    if concurrency < 1:
        print("ERROR: --concurrency must be >= 1", file=sys.stderr)
        return
    if limit is not None and limit < 1:
        print("ERROR: --limit must be >= 1 when provided", file=sys.stderr)
        return

    # Lazy import to avoid eager config loading
    try:
        from cortex.orchestrator import PipelineOrchestrator
    except Exception as exc:
        print(f"ERROR: Failed to import pipeline orchestrator: {exc}", file=sys.stderr)
        traceback.print_exc()
        return

    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(levelname)s: %(message)s", force=True
    )

    effective_auto_embed = auto_embed if not dry_run else False

    if not json_output:
        console.print("[bold blue]🚀 Cortex Unified Pipeline[/bold blue]")
        console.print(f"   Source:      [cyan]{source_prefix}[/cyan]")
        console.print(f"   Tenant:      [cyan]{tenant_id}[/cyan]")
        console.print(f"   Concurrency: [cyan]{concurrency}[/cyan]")
        console.print(f"   Auto-Embed:  [cyan]{effective_auto_embed}[/cyan]")
        if dry_run:
            console.print("   [yellow]DRY RUN MODE - No changes will be made[/yellow]")
            if auto_embed:
                console.print("   [dim](--auto-embed ignored in dry-run mode)[/dim]")
        console.print()

    orchestrator = PipelineOrchestrator(
        tenant_id=tenant_id,
        auto_embed=effective_auto_embed,
        concurrency=concurrency,
        dry_run=dry_run,
    )

    # Run the pipeline (now enqueues jobs)
    try:
        stats = orchestrator.run(source_prefix=source_prefix, limit=limit)
    except Exception as exc:
        logger.error("Pipeline run failed: %s", exc, exc_info=True)
        print(f"ERROR: Pipeline run failed: {exc}", file=sys.stderr)
        return
    if stats is None:
        print("ERROR: Pipeline returned no stats.", file=sys.stderr)
        return
    try:
        duration_seconds = float(getattr(stats, "duration_seconds", 0.0))
    except (TypeError, ValueError):
        duration_seconds = 0.0

    # Output results
    if not json_output:
        console.print()
        table = Table(title="Pipeline Enqueueing Results")
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="bold")

        table.add_row("Duration", f"{duration_seconds:.2f}s")
        table.add_row("Folders Found", str(stats.folders_found))
        table.add_row("Jobs Enqueued", f"[green]{stats.folders_enqueued}[/green]")
        table.add_row("Failed to Enqueue", f"[red]{stats.folders_failed}[/red]")

        console.print(table)

        if stats.folders_failed > 0:
            console.print(
                f"\n[yellow]⚠ {stats.folders_failed} folder(s) failed to enqueue. Run with --verbose for details.[/yellow]"
            )
        elif stats.folders_enqueued > 0:
            console.print(
                f"\n[green]✓ {stats.folders_enqueued} ingestion job(s) enqueued successfully![/green]"
            )
            console.print("[dim]  Run worker processes to handle the jobs.[/dim]")
        else:
            console.print("\n[yellow]No new folders found to process.[/yellow]")
    else:
        output = {
            "success": stats.folders_failed == 0,
            "dry_run": dry_run,
            "duration_seconds": duration_seconds,
            "folders_found": stats.folders_found,
            "jobs_enqueued": stats.folders_enqueued,
            "enqueue_failures": stats.folders_failed,
            "errors": getattr(stats, "errors_list", []),
        }
        print(json.dumps(output, default=str))
