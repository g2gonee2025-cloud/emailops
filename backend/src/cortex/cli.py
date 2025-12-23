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
    tenant: str = typer.Option("default", help="Tenant ID"),
):
    """
    Trigger ingestion process.
    """
    logger.info("ingestion_trigger", source=source, dry_run=dry_run)

    if source.lower() == "s3":
        from cortex.ingestion.processor import IngestionProcessor

        # Use defaults from config/env
        processor = IngestionProcessor()
        processor.run_full_ingestion()
    elif source == "backfill-embeddings":
        from cortex.ingestion.backfill import backfill_embeddings

        backfill_embeddings(tenant_id=tenant)
    else:
        logger.error("unsupported_source", source=source)
        print(
            f"Error: Source '{source}' not supported. Use 's3' or 'backfill-embeddings'."
        )


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-k", help="Number of results"),
    tenant: str = typer.Option("default", "--tenant", "-t", help="Tenant ID"),
    fusion: str = typer.Option(
        "rrf", "--fusion", help="Fusion method: rrf or weighted_sum"
    ),
    debug: bool = typer.Option(False, "--debug", help="Show detailed score breakdown"),
):
    """
    Search the Knowledge Base (Hybrid = Vector + FTS).
    """
    from cortex.retrieval.hybrid_search import KBSearchInput, tool_kb_search_hybrid
    from cortex.retrieval.query_classifier import QueryClassification

    logger.info("search_command_start", query=query, tenant=tenant)

    try:
        # Construct input
        # We default to 'semantic' classification for CLI unless specialized
        classification = QueryClassification(query=query, type="semantic")

        args = KBSearchInput(
            tenant_id=tenant,
            user_id="cli-user",
            query=query,
            k=limit,
            fusion_method=fusion,  # type: ignore
            classification=classification,
        )

        # Execute search
        results = tool_kb_search_hybrid(args)

        # Output results
        print(f"\n🔎 Search Results for: '{query}' ({len(results.results)} hits)\n")

        for i, item in enumerate(results.results, 1):
            score_display = f"{item.score:.4f}"
            if debug:
                score_display += f" (Fusion: {item.fusion_score:.4f} | Vec: {item.vector_score or 0:.4f} | Lex: {item.lexical_score or 0:.4f})"

            print(
                f"{i}. [{score_display}] {item.content[:200].replace(chr(10), ' ')}..."
            )
            if debug:
                print(
                    f"    Source: {item.chunk_id or item.message_id} | Type: {item.metadata.get('chunk_type', 'unknown')}"
                )
                if item.highlights:
                    print(f"    Highlights: {item.highlights[:2]}")
            print("")

    except Exception as e:
        logger.error("search_failed", error=str(e))
        print(f"\n❌ Search failed: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
