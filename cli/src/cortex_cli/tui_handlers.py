"""
TUI handlers for database, embeddings, S3, config, imports, and RAG.
Moves complex logic out of the main loop.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import questionary
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()

PROMPT_DRY_RUN = "Dry run?"


# --- Database ---
def _handle_db() -> None:
    action = questionary.select(
        "Database Management:", choices=["Stats", "Migrate", "Back"]
    ).ask()

    if not action or "Back" in action:
        return

    from cortex_cli.cmd_db import cmd_db_stats

    @dataclass
    class DbArgs:
        json: bool = False
        verbose: bool = False
        dry_run: bool = False

    if "Stats" in action:
        console.print("[dim]Fetching database stats...[/dim]")
        cmd_db_stats(DbArgs())
    elif "Migrate" in action:
        dry_run = questionary.confirm(PROMPT_DRY_RUN).ask()

        args = DbArgs(dry_run=dry_run)

        try:
            # Import dynamically to avoid circular imports if any
            from cortex_cli.cmd_db import cmd_db_migrate

            console.print("[dim]Running migrations...[/dim]")
            cmd_db_migrate(args)
        except Exception as e:
            console.print(f"[red]Error running migrations: {e}[/red]")

    # SECTION 4: View Ledger
    # ----------------------
    # This section seems to be misplaced here, but added as per instruction.
    # It refers to an 's' object which is not defined in this scope.
    # Assuming 's' would be available in a larger context or is a placeholder.
    # For now, commenting out the 's' related lines to avoid NameError.
    # if s.view_ledger_req:
    #     s.view_ledger_req = False
    #     try:
    #         from cortex_cli import cmd_db
    #         ledger = cmd_db.get_ledger()
    #         if not ledger:
    #             print("No ledger found.")
    #         else:
    #             # We can iterate the pydantic model dump
    #             data = ledger.model_dump() if hasattr(ledger, "model_dump") else ledger

    #             # Simple display of the dict/model
    #             # The original instruction ended here, assuming this is where the display logic would go.
    #             # For example:
    #             # console.print(data)

    questionary.press_any_key_to_continue().ask()


# --- Embeddings ---
def _handle_embeddings() -> None:
    action = questionary.select(
        "Embeddings Management:", choices=["Stats", "Backfill", "Back"]
    ).ask()

    if not action or "Back" in action:
        return

    from cortex_cli.cmd_embeddings import cmd_embeddings_backfill, cmd_embeddings_stats

    @dataclass
    class EmbeddingsArgs:
        json: bool = False
        verbose: bool = False
        limit: int | None = None
        batch_size: int = 64
        dry_run: bool = False

    if "Stats" in action:
        cmd_embeddings_stats(EmbeddingsArgs())
    elif "Backfill" in action:
        limit = questionary.text("Limit (optional):").ask()
        dry_run = questionary.confirm(PROMPT_DRY_RUN, default=False).ask()
        limit_int = int(limit) if limit and limit.isdigit() else None

        args = EmbeddingsArgs(limit=limit_int, dry_run=dry_run)
        cmd_embeddings_backfill(args)

    questionary.press_any_key_to_continue().ask()


# --- S3 / Storage ---
def _handle_s3() -> None:
    action = questionary.select(
        "S3/Spaces Storage:",
        choices=["List Buckets/Prefixes", "Ingest from S3", "Back"],
    ).ask()

    if not action or "Back" in action:
        return

    from cortex_cli.cmd_s3 import cmd_s3_ingest, cmd_s3_list

    @dataclass
    class S3Args:
        json: bool = False
        verbose: bool = False
        prefix: str = ""
        bucket: str | None = None
        limit: int = 50
        dry_run: bool = False
        tenant: str = "default"

    if "List" in action:
        prefix = questionary.text("Prefix (optional):").ask()
        args = S3Args(prefix=prefix or "")
        cmd_s3_list(args)
    elif "Ingest" in action:
        prefix = questionary.text("Prefix to ingest:").ask()
        if not prefix:
            return

        dry_run = questionary.confirm(PROMPT_DRY_RUN).ask()
        args = S3Args(prefix=prefix, dry_run=dry_run)
        cmd_s3_ingest(args)

    questionary.press_any_key_to_continue().ask()


# --- Import Data (Unified) ---
def _handle_import_data(cli_main) -> None:
    """Unified import handler for local and S3 sources."""
    source_type = questionary.select(
        "Select data source:",
        choices=["📁 Local Filesystem", "☁️ S3/DigitalOcean Spaces", "🔙 Back"],
    ).ask()

    if not source_type or "Back" in source_type:
        return

    if "Local" in source_type:
        _handle_local_import(cli_main)
    elif "S3" in source_type:
        _handle_s3_import_flow()

    questionary.press_any_key_to_continue().ask()


def _handle_local_import(cli_main) -> None:
    def validate_path(path):
        if not path:
            return "Please enter a path"
        if not Path(path).is_dir():
            return "Path must be a directory"
        return True

    source = questionary.path(
        "Path to export folder:", only_directories=True, validate=validate_path
    ).ask()
    if not source:
        return

    dry_run = questionary.confirm("Dry run? (No changes)", default=False).ask()

    console.print(f"[dim]Running ingest on {source}...[/dim]")
    cli_main._run_ingest(source_path=str(source), dry_run=dry_run)


def _handle_s3_import_flow() -> None:
    from cortex_cli.cmd_s3 import cmd_s3_ingest, cmd_s3_list

    # First list available prefixes
    list_first = questionary.confirm(
        "List available S3 prefixes first?", default=True
    ).ask()

    if list_first:

        @dataclass
        class ListArgs:
            json: bool = False
            prefix: str = ""
            limit: int = 20

        cmd_s3_list(ListArgs())
        print()

    prefix = questionary.text("Prefix to ingest:").ask()
    if not prefix:
        return

    dry_run = questionary.confirm(PROMPT_DRY_RUN, default=False).ask()

    @dataclass
    class IngestArgs:
        json: bool = False
        verbose: bool = False
        tenant: str = "default"
        dry_run: bool = False
        prefix: str = ""

    args = IngestArgs()
    args.prefix = prefix
    args.dry_run = dry_run

    cmd_s3_ingest(args)


# --- Config ---
def _view_config_section(cli_main) -> None:
    section = questionary.select(
        "Select Section:",
        choices=[
            "Core",
            "Embeddings",
            "Search",
            "Database",
            "Storage",
            "Processing",
            "GCP",
            "DigitalOcean LLM",
            "Retry",
            "Limits",
        ],
    ).ask()

    if not section:
        return

    cli_main._show_config(section=section)
    questionary.press_any_key_to_continue().ask()


# --- RAG / Search ---
def _handle_rag_menu() -> None:
    while True:
        action = questionary.select(
            "Select AI Capability:",
            choices=[
                "🔎 Search (Hybrid)",
                "💬 Answer (QA)",
                "📝 Draft Email",
                "📋 Summarize Thread",
                "🔙 Back",
            ],
        ).ask()

        if not action or "Back" in action:
            return

        if "Search" in action:
            _interactive_search()
        elif "Answer" in action:
            _interactive_answer()
        elif "Draft" in action:
            _interactive_draft()
        elif "Summarize" in action:
            _interactive_summarize()


def _interactive_search() -> None:
    query = questionary.text("Enter search query:").ask()
    if not query:
        return

    top_k_str = questionary.text("Top K results:", default="10").ask()
    try:
        top_k = int(top_k_str)
    except ValueError:
        console.print("[red]Invalid integer for Top K, defaulting to 10.[/red]")
        top_k = 10

    console.print(f"[dim]Searching for '{query}'...[/dim]")

    try:
        from cortex.retrieval.hybrid_search import KBSearchInput, tool_kb_search_hybrid

        search_input = KBSearchInput(
            query=query, k=top_k, tenant_id="default", user_id="cli-user"
        )
        results = tool_kb_search_hybrid(search_input)

        if not results.results:
            console.print("[yellow]No results found.[/yellow]")
        else:
            table = Table(box=box.ROUNDED, show_lines=True)
            table.add_column("Score", style="cyan", width=8)
            table.add_column("Source", style="dim", width=20)
            table.add_column("Snippet", style="white")

            for res in results.results:
                raw_text = getattr(res, "text", "") or ""
                if not isinstance(raw_text, str):
                    raw_text = str(raw_text)

                snippet = (raw_text[:150] + "...") if len(raw_text) > 150 else raw_text
                table.add_row(
                    f"{res.score:.3f}", str(res.source), snippet.replace("\n", " ")
                )

            console.print(table)

    except Exception as e:
        console.print(f"[bold red]Search failed:[/bold red] {e}")

    questionary.press_any_key_to_continue().ask()


def _interactive_answer() -> None:
    query = questionary.text("Enter your question:").ask()
    if not query:
        return

    console.print("[dim]Thinking...[/dim]")

    try:
        from cortex_cli.api_client import get_api_client

        api_client = get_api_client()
        response = api_client.answer(
            query=query, tenant_id="default", user_id="cli-user"
        )

        answer = response.get("answer")

        if not answer:
            console.print("[yellow]No answer generated.[/yellow]")
        else:
            console.print(
                Panel(
                    Markdown(answer.get("answer_markdown", "")),
                    title="Answer",
                    border_style="green",
                )
            )

            evidence = answer.get("evidence", [])
            if evidence:
                table = Table(title="Sources", box=box.SIMPLE)
                table.add_column("Ref", style="cyan")
                table.add_column("Evidence")
                for i, ev in enumerate(evidence, 1):
                    txt = ev.get("snippet") or ev.get("text") or "No text"
                    table.add_row(str(i), txt[:100] + "...")
                console.print(table)

    except Exception as e:
        console.print(f"[bold red]QA failed:[/bold red] {e}")

    questionary.press_any_key_to_continue().ask()


def _interactive_draft() -> None:
    instructions = questionary.text("Drafting instructions:").ask()
    if not instructions:
        return
    thread_id = questionary.text("Thread ID (optional):").ask()

    console.print("[dim]Drafting...[/dim]")

    try:
        from cortex.orchestration.graphs import build_draft_graph

        graph = build_draft_graph().compile()

        async def _run():
            return await graph.ainvoke(
                {
                    "explicit_query": instructions,
                    "thread_id": thread_id if thread_id else None,
                    "tenant_id": "default",
                    "user_id": "cli-user",
                    "draft_query": None,
                    "retrieval_results": None,
                    "assembled_context": None,
                    "draft": None,
                    "critique": None,
                    "iteration_count": 0,
                    "error": None,
                }
            )

        state = asyncio.run(_run())

        if state.get("error"):
            console.print(f"[bold red]Error:[/bold red] {state['error']}")
        elif state.get("draft"):
            d = state["draft"]
            console.print(
                Panel(
                    f"Subject: {d.subject}\nTo: {d.to}",
                    title="Draft Metadata",
                    style="dim",
                )
            )
            console.print(
                Panel(
                    Markdown(d.body_markdown), title="Email Body", border_style="blue"
                )
            )
        else:
            console.print("[yellow]No draft generated.[/yellow]")

    except Exception as e:
        console.print(f"[bold red]Drafting failed:[/bold red] {e}")

    questionary.press_any_key_to_continue().ask()


def _interactive_summarize() -> None:
    thread_id = questionary.text("Thread ID:").ask()
    if not thread_id:
        return

    console.print("[dim]Summarizing...[/dim]")

    try:
        from cortex.orchestration.graphs import build_summarize_graph

        graph = build_summarize_graph().compile()

        async def _run():
            return await graph.ainvoke(
                {
                    "thread_id": thread_id,
                    "tenant_id": "default",
                    "user_id": "cli-user",
                    "thread_context": None,
                    "facts_ledger": None,
                    "critique": None,
                    "iteration_count": 0,
                    "summary": None,
                    "error": None,
                }
            )

        state = asyncio.run(_run())

        if state.get("error"):
            console.print(f"[bold red]Error:[/bold red] {state['error']}")
        elif state.get("summary"):
            s = state["summary"]
            console.print(
                Panel(
                    Markdown(s.summary_markdown),
                    title="Summary",
                    border_style="magenta",
                )
            )

            if s.facts_ledger:
                console.print("[bold]Facts Ledger:[/bold]")
                # We can iterate the pydantic model dump
                ledger = s.facts_ledger
                data = ledger.model_dump() if hasattr(ledger, "model_dump") else ledger

                # Simple display of the dict/model
                from rich.pretty import pprint

                pprint(data, max_depth=2)

        else:
            console.print("[yellow]No summary generated.[/yellow]")

    except Exception as e:
        console.print(f"[bold red]Summarization failed:[/bold red] {e}")

    questionary.press_any_key_to_continue().ask()
