from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from cortex.audit import get_audit_trail

app = typer.Typer()


@app.command()
def show(
    tenant_id: str = typer.Option(..., help="Tenant ID to query audit logs for"),
    limit: int = typer.Option(100, help="Maximum number of audit logs to retrieve"),
    since: Optional[datetime] = typer.Option(None, help="Show logs since a specific ISO 8601 timestamp"),
    user_or_agent: Optional[str] = typer.Option(None, help="Filter by user or agent"),
    action: Optional[str] = typer.Option(None, help="Filter by a specific action"),
    correlation_id: Optional[str] = typer.Option(None, help="Filter by correlation ID"),
):
    """
    Query and display audit trail logs with powerful filtering capabilities.
    """
    console = Console()

    async def _get_and_print_logs():
        try:
            results = await get_audit_trail(
                tenant_id=tenant_id,
                action=action,
                user_or_agent=user_or_agent,
                correlation_id=correlation_id,
                since=since,
                limit=limit,
            )

            if not results:
                console.print("No audit events found for the specified criteria.")
                return

            table = Table(
                "Timestamp",
                "Action",
                "User/Agent",
                "Risk",
                "Input Hash",
                "Output Hash",
                "Correlation ID",
            )
            for r in results:
                correlation_id_str = r.metadata_.get("correlation_id") if r.metadata_ else "N/A"
                table.add_row(
                    str(r.ts),
                    r.action,
                    r.user_or_agent,
                    r.risk_level,
                    r.input_hash[:12] if r.input_hash else "N/A",
                    r.output_hash[:12] if r.output_hash else "N/A",
                    str(correlation_id_str),
                )
            console.print(table)

        except Exception as e:
            console.print(f"[red]Error querying audit log: {e}[/red]")

    asyncio.run(_get_and_print_logs())


if __name__ == "__main__":
    app()
