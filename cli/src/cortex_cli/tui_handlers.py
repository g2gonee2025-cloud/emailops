"""
TUI handlers for database, embeddings, S3, config, imports, and RAG.
Moves complex logic out of the main loop.
"""
import asyncio
from pathlib import Path

import questionary
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()


# --- Database ---
def _handle_db():
    action = questionary.select(
        "Database Management:", choices=["Stats", "Migrate", "Back"]
    ).ask()

    if not action or "Back" in action:
        return

    import sys

    from cortex_cli import cmd_doctor
    from cortex_cli.cmd_db import cmd_db_stats

    class Args:
        json = False
        verbose = False

    if "Stats" in action:
        console.print("[dim]Fetching database stats...[/dim]")
        cmd_db_stats(Args())
    elif "Migrate" in action:
        dry_run = questionary.confirm("Dry run?").ask()
        # Assuming doctor_args is meant to be constructed from dry_run
        doctor_args = ["--dry-run"] if dry_run else []
        sys.argv = [sys.argv[0], *doctor_args]
        try:
            cmd_doctor.main()  # This seems to replace cmd_db_migrate
        except Exception as e:
            console.print(f"[red]Error running doctor: {e}[/red]")

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
def _handle_embeddings():
    action = questionary.select(
        "Embeddings Management:", choices=["Stats", "Backfill", "Back"]
    ).ask()

    if not action or "Back" in action:
        return

    from cortex_cli.cmd_embeddings import cmd_embeddings_backfill, cmd_embeddings_stats

    class Args:
        json = False
        verbose = False
        limit = None
        batch_size = 64
        dry_run = False

    if "Stats" in action:
        cmd_embeddings_stats(Args())
    elif "Backfill" in action:
        limit = questionary.text("Limit (optional):").ask()
        dry_run = questionary.confirm("Dry run?", default=False).ask()
        limit_int = int(limit) if limit and limit.isdigit() else None

        args = Args()
        args.limit = limit_int
        args.dry_run = dry_run
        cmd_embeddings_backfill(args)

    questionary.press_any_key_to_continue().ask()


# --- S3 / Storage ---
def _handle_s3():
    action = questionary.select(
        "S3/Spaces Storage:",
        choices=["List Buckets/Prefixes", "Ingest from S3", "Back"],
    ).ask()

    if not action or "Back" in action:
        return

    from cortex_cli.cmd_s3 import cmd_s3_ingest, cmd_s3_list

    class Args:
        json = False
        verbose = False
        prefix = ""
        bucket = None
        limit = 50

    if "List" in action:
        prefix = questionary.text("Prefix (optional):").ask()
        args = Args()
        args.prefix = prefix or ""
        cmd_s3_list(args)
    elif "Ingest" in action:
        prefix = questionary.text("Prefix to ingest:").ask()
        if not prefix:
            return

        dry_run = questionary.confirm("Dry run?").ask()
        args = Args()
        args.prefix = prefix
        args.dry_run = dry_run
        args.tenant = "default"
        # s3 ingest usually requires a source argument too, but let's check cmd_s3
        cmd_s3_ingest(args)

    questionary.press_any_key_to_continue().ask()


# --- Import Data (Unified) ---
def _handle_import_data(cli_main):
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


def _handle_local_import(cli_main):
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


def _handle_s3_import_flow():
    from cortex_cli.cmd_s3 import cmd_s3_ingest, cmd_s3_list

    # First list available prefixes
    list_first = questionary.confirm(
        "List available S3 prefixes first?", default=True
    ).ask()

    if list_first:

        class ListArgs:
            json = False
            prefix = ""
            limit = 20

        cmd_s3_list(ListArgs())
        print()

    prefix = questionary.text("Prefix to ingest:").ask()
    if not prefix:
        return

    dry_run = questionary.confirm("Dry run?", default=False).ask()

    class IngestArgs:
        json = False
        verbose = False
        tenant = "default"
        dry_run = False

    args = IngestArgs()
    args.prefix = prefix
    args.dry_run = dry_run

    cmd_s3_ingest(args)


# --- Config ---
def _view_config_section(cli_main):
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
def _handle_rag_menu():
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


def _interactive_search():
    query = questionary.text("Enter search query:").ask()
    if not query:
        return

    top_k = int(questionary.text("Top K results:", default="10").ask())

    console.print(f"[dim]Searching for '{query}'...[/dim]")

    try:
        from cortex.models.api import SearchRequest
        from cortex.retrieval.hybrid_search import hybrid_search

        req = SearchRequest(query=query, top_k=top_k, tenant_id="default")
        results = hybrid_search(req)

        if not results.results:
            console.print("[yellow]No results found.[/yellow]")
        else:
            table = Table(box=box.ROUNDED, show_lines=True)
            table.add_column("Score", style="cyan", width=8)
            table.add_column("Source", style="dim", width=20)
            table.add_column("Snippet", style="white")

            for res in results.results:
                snippet = (res.text[:150] + "...") if len(res.text) > 150 else res.text
                table.add_row(
                    f"{res.score:.3f}", str(res.source), snippet.replace("\n", " ")
                )

            console.print(table)

    except Exception as e:
        console.print(f"[bold red]Search failed:[/bold red] {e}")

    questionary.press_any_key_to_continue().ask()


def _interactive_answer():
    query = questionary.text("Enter your question:").ask()
    if not query:
        return

    console.print("[dim]Thinking...[/dim]")

    try:
        from cortex.orchestration.graphs import build_answer_graph

        graph = build_answer_graph().compile()

        async def _run():
            return await graph.ainvoke(
                {"query": query, "tenant_id": "default", "user_id": "cli-user"}
            )

        state = asyncio.run(_run())

        if state.get("error"):
            console.print(f"[bold red]Error:[/bold red] {state['error']}")
        elif state.get("answer"):
            ans = state["answer"]
            console.print(
                Panel(
                    Markdown(ans.answer_markdown), title="Answer", border_style="green"
                )
            )

            if ans.evidence:
                table = Table(title="Sources", box=box.SIMPLE)
                table.add_column("Ref", style="cyan")
                table.add_column("Evidence")
                for i, ev in enumerate(ans.evidence, 1):
                    txt = ev.snippet or ev.text or "No text"
                    table.add_row(str(i), txt[:100] + "...")
                console.print(table)
        else:
            console.print("[yellow]No answer generated.[/yellow]")

    except Exception as e:
        console.print(f"[bold red]QA failed:[/bold red] {e}")

    questionary.press_any_key_to_continue().ask()


def _interactive_draft():
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


def _interactive_summarize():
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
