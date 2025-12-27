"""CLI for graph-related commands."""

import typer
from cortex.intelligence.graph_discovery import discover_graph_schema as discover_schema_logic

app = typer.Typer(
    name="graph",
    help="Commands for interacting with the Knowledge Graph.",
    no_args_is_help=True,
)


@app.command("discover-schema")
def discover_schema(
    tenant_id: str = typer.Option(
        ...,
        "--tenant-id",
        "-t",
        help="Tenant ID to search within.",
    ),
    sample_size: int = typer.Option(
        20,
        "--sample-size",
        "-n",
        help="Number of conversations to sample.",
    ),
) -> None:
    """
    Discover the graph schema from a sample of conversations.
    """
    discover_schema_logic(tenant_id=tenant_id, sample_size=sample_size)
