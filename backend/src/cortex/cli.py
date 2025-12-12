"""
Cortex CLI Entry Point.

Provides command-line interface for Cortex operations including
doctor checks, ingestion triggers, and system management.
"""

import structlog
import typer
from cortex.cmd_doctor import CortexDoctor

# Configure stuctlog for CLI
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)

app = typer.Typer(help="Outlook Cortex CLI")
logger = structlog.get_logger()


@app.command()
def doctor(check_all: bool = typer.Option(False, "--check-all", help="Run all checks")):
    """
    Run system health checks (Doctor).
    """
    doc = CortexDoctor()
    logger.info("running_cortex_doctor")
    success = doc.run_all()
    if not success:
        logger.error("doctor_checks_failed")
        raise typer.Exit(code=1)
    logger.info("doctor_checks_passed")


@app.command()
def ingest(
    source: str = typer.Option(..., help="Source to ingest from (e.g. s3)"),
    dry_run: bool = typer.Option(False, help="Verify without processing"),
):
    """
    Trigger ingestion process.
    """
    logger.info("ingestion_trigger", source=source, dry_run=dry_run)
    # Placeholder for connecting to ingestion routes or logic
    print(f"Triggering ingestion from {source} (Dry Run: {dry_run})")


if __name__ == "__main__":
    app()
