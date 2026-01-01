"""
Interactive TUI for Cortex CLI.
"""

import sys

try:
    import questionary
except ImportError:
    questionary = None
from rich.console import Console
from rich.panel import Panel

# Lazy imports are handled inside functions to ensure fast startup

console = Console()

STYLE_HIGHLIGHT = "fg:cyan bold"


def _print_header() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]EmailOps Cortex[/bold cyan] [dim]Interactive Mode[/dim]",
            border_style="cyan",
            padding=(0, 2),
        )
    )


def _require_questionary() -> None:
    if questionary is None:
        console.print("[bold red]Error:[/bold red] questionary is not installed.")
        console.print("Install it with: pip install questionary")
        raise SystemExit(1)


def interactive_loop() -> None:
    """Main interactive loop."""
    _require_questionary()
    # We don't import main here anymore unless needed for specific delegated tasks (like status)
    from cortex_cli import main as cli_main
    from cortex_cli.tui_handlers import (
        _handle_db,
        _handle_embeddings,
        _handle_import_data,
        _handle_rag_menu,
    )

    _print_header()

    while True:
        choice = questionary.select(
            "What would you like to do?",
            choices=[
                # 1. Setup & Health (first step)
                "🏥 Doctor (System Health Check)",
                "📊 Status (Environment Overview)",
                # 2. Data Import (second step)
                "📥 Import Data (Local or S3)",
                # 3. Processing (third step)
                "⚙️  Build Index (Embeddings)",
                "🗄️ Database (Stats, Migrate)",
                "🧠 Embeddings (Stats, Backfill)",
                # 4. Query & Use (fourth step)
                "🔍 Search & RAG (Query Your Data)",
                # 5. Admin
                "🔧 Configuration",
                "❌ Exit",
            ],
            style=questionary.Style(
                [
                    ("qmark", STYLE_HIGHLIGHT),
                    ("question", "fg:white bold"),
                    ("answer", STYLE_HIGHLIGHT),
                    ("pointer", STYLE_HIGHLIGHT),
                    ("highlighted", STYLE_HIGHLIGHT),
                    ("selected", STYLE_HIGHLIGHT),
                    ("separator", "fg:black"),
                    ("instruction", "fg:white"),
                    ("text", "fg:white"),
                    ("disabled", "fg:gray"),
                ]
            ),
        ).ask()

        if not choice or "Exit" in choice:
            console.print("[yellow]Goodbye![/yellow]")
            sys.exit(0)

        try:
            if "Doctor" in choice:
                _handle_doctor()
            elif "Status" in choice:
                cli_main._show_status()
                questionary.press_any_key_to_continue().ask()
            elif "Import Data" in choice:
                _handle_import_data(cli_main)
            elif "Build Index" in choice:
                _handle_index(cli_main)
            elif "Database" in choice:
                _handle_db()
            elif "Embeddings" in choice:
                _handle_embeddings()
            elif "Search & RAG" in choice:
                _handle_rag_menu()
            elif "Configuration" in choice:
                _handle_config(cli_main)

            print()  # Spacer
        except KeyboardInterrupt:
            continue
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            questionary.press_any_key_to_continue().ask()


def _handle_index(cli_main) -> None:
    _require_questionary()
    provider = questionary.select(
        "Embedding Provider:",
        choices=["vertex", "openai", "local"],
        default="vertex",
    ).ask()

    workers = questionary.text(
        "Number of workers:",
        default="4",
        validate=lambda x: x.isdigit() or "Must be integer",
    ).ask()
    force = questionary.confirm("Force Re-index (Ignoring cache)?", default=False).ask()

    console.print("[dim]Starting indexer...[/dim]")
    cli_main._run_index(root=".", provider=provider, workers=int(workers), force=force)
    questionary.press_any_key_to_continue().ask()


def _handle_doctor() -> None:
    _require_questionary()
    checks = questionary.checkbox(
        "Select diagnostics to run:",
        choices=[
            questionary.Choice("Check Database", checked=True, value="check_db"),
            questionary.Choice("Check S3/Spaces", checked=True, value="check_exports"),
            questionary.Choice(
                "Check Embeddings API", checked=True, value="check_embeddings"
            ),
            questionary.Choice(
                "Check Reranker API", checked=True, value="check_reranker"
            ),
            questionary.Choice("Check Index", value="check_index"),
            questionary.Choice("Check Redis", value="check_redis"),
        ],
    ).ask()

    if not checks:
        return

    # Call doctor main via sys.argv injection (robust fallback)
    from cortex_cli import cmd_doctor

    original_argv = sys.argv
    # We DO NOT add "doctor" as the first argument because cmd_doctor.py uses a flat ArgumentParser
    # that doesn't expect a subcommand when run as __main__.
    doctor_args = []
    for c in checks:
        doctor_args.append(f"--{c.replace('_', '-')}")

    sys.argv = [sys.argv[0], *doctor_args]
    try:
        cmd_doctor.main()
    except SystemExit:
        pass
    finally:
        sys.argv = original_argv

    questionary.press_any_key_to_continue().ask()


def _handle_config(cli_main) -> None:
    """Show configuration directly without nested submenu."""
    _require_questionary()
    action = questionary.select(
        "Configuration:",
        choices=["View All (Tree)", "View as JSON", "Validate", "Back"],
    ).ask()

    if not action or "Back" in action:
        return

    if "Tree" in action:
        _view_rich_config()
    elif "JSON" in action:
        cli_main._show_config(export_format="json")
        questionary.press_any_key_to_continue().ask()
    elif "Validate" in action:
        cli_main._show_config(validate=True)
        questionary.press_any_key_to_continue().ask()


def _view_rich_config() -> None:
    try:
        from cortex.config.loader import get_config
        from rich.tree import Tree

        config = get_config()

        # Build tree
        tree = Tree(f"[bold cyan]Configuration ({config.core.env})[/bold cyan]")

        # Core
        core = tree.add("[bold]Core[/bold]")
        core.add(f"Provider: {config.core.provider}")
        core.add(f"Persona: {config.core.persona}")
        core.add(f"Env: {config.core.env}")

        # Embeddings
        emb = tree.add("[bold]Embeddings[/bold]")
        emb.add(f"Model: {config.embedding.model_name}")
        emb.add(f"Dim: {config.embedding.output_dimensionality}")
        emb.add(f"Provider: {config.embedding.provider}")

        # Search
        search = tree.add("[bold]Search[/bold]")
        search.add(f"Fusion: {config.search.fusion_strategy}")
        search.add(f"K: {config.search.k}")
        search.add(f"Reranker: {config.search.reranker_endpoint or 'None'}")

        # Database
        db = tree.add("[bold]Database[/bold]")
        db.add(
            f"URL: {config.database.url.split('@')[-1] if '@' in config.database.url else 'local'}"
        )
        db.add(f"Pool Size: {config.database.pool_size}")

        # Storage
        store = tree.add("[bold]Storage[/bold]")
        store.add(f"Endpoint: {config.storage.endpoint_url}")
        store.add(f"Bucket: {config.storage.bucket_raw}")
        store.add(f"Region: {config.storage.region}")

        # Processing
        proc = tree.add("[bold]Processing[/bold]")
        proc.add(f"Chunk Size: {config.processing.chunk_size}")
        proc.add(f"Overlap: {config.processing.chunk_overlap}")
        proc.add(f"Workers: {config.processing.max_workers}")

        # GCP
        if config.gcp:
            gcp = tree.add("[bold]GCP[/bold]")
            gcp.add(f"Project: {config.gcp.gcp_project}")
            gcp.add(f"Region: {config.gcp.gcp_region}")
            gcp.add(f"Vertex Loc: {config.gcp.vertex_location}")

        # DO LLM
        if config.digitalocean_llm and hasattr(config.digitalocean_llm, "model"):
            do = tree.add("[bold]DigitalOcean LLM[/bold]")
            do.add(f"Model: {config.digitalocean_llm.model.name}")
            do.add(f"Endpoint: {config.digitalocean_llm.endpoint.BASE_URL}")

        # Retry
        retry = tree.add("[bold]Retry[/bold]")
        retry.add(f"Max Retries: {config.retry.max_retries}")
        retry.add(
            f"Backoff: {config.retry.min_backoff_seconds}-{config.retry.max_backoff_seconds}s"
        )

        # Limits
        limits = tree.add("[bold]Limits[/bold]")
        limits.add(f"Max Attach Chars: {config.limits.max_attachment_text_chars}")

        console.print(tree)

    except Exception as e:
        console.print(f"[red]Failed to load config: {e}[/red]")

    questionary.press_any_key_to_continue().ask()
