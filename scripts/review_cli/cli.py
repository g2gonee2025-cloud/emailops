"""Interactive CLI for code review."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from .config import (
    DEFAULT_SCAN_DIRS,
    EXTENSION_GROUPS,
    PROVIDER_MODELS,
    Config,
    ReviewConfig,
    ScanConfig,
)
from .providers.base import ReviewResult
from .providers.jules import JulesProvider
from .providers.openai_compat import OpenAICompatProvider
from .reports.reporter import Reporter
from .reviewers.code_reviewer import CodeReviewer
from .scanners.file_scanner import FileScanner

console = Console()


def select_provider() -> str:
    """Prompt user to select a provider."""
    console.print("\n[bold cyan]Select LLM Provider:[/bold cyan]")
    console.print("  1. OpenAI-compatible (DO Inference, OpenRouter, etc.)")
    console.print("  2. Jules API (GitHub PR automation)")

    choice = Prompt.ask("Enter choice", choices=["1", "2"], default="1")
    return "openai" if choice == "1" else "jules"


def select_model(provider: str) -> str:
    """Prompt user to select a model."""
    models = PROVIDER_MODELS.get(provider, [])
    if not models:
        raise ValueError(f"No models configured for provider: {provider}")

    console.print(f"\n[bold cyan]Select Model for {provider}:[/bold cyan]")
    for i, model in enumerate(models, 1):
        console.print(f"  {i}. {model}")

    choice = IntPrompt.ask("Enter choice", default=1)
    idx = max(0, min(choice - 1, len(models) - 1))
    return models[idx]


def select_directories(project_root: Path) -> list[Path]:
    """Prompt user to select directories to scan."""
    console.print("\n[bold cyan]Select Directories to Scan:[/bold cyan]")

    available_dirs = []
    for d in DEFAULT_SCAN_DIRS:
        full_path = project_root / d
        if full_path.exists():
            available_dirs.append(d)
            console.print(f"  {len(available_dirs)}. {d}")

    if not available_dirs:
        console.print(
            "[yellow]No default directories found. Using project root.[/yellow]"
        )
        return [project_root]

    console.print(f"  {len(available_dirs) + 1}. [All directories]")

    response = Prompt.ask(
        "Enter numbers (comma-separated) or 'all'",
        default="all",
    )

    if response.lower() == "all" or response == str(len(available_dirs) + 1):
        return [project_root / d for d in available_dirs]

    selected = []
    for part in response.split(","):
        try:
            idx = int(part.strip()) - 1
            if 0 <= idx < len(available_dirs):
                selected.append(project_root / available_dirs[idx])
        except ValueError:
            continue

    return selected if selected else [project_root / d for d in available_dirs]


def select_extensions() -> set[str]:
    """Prompt user to select file extensions."""
    console.print("\n[bold cyan]Select File Types:[/bold cyan]")

    groups = list(EXTENSION_GROUPS.items())
    for i, (name, exts) in enumerate(groups, 1):
        console.print(f"  {i}. {name} ({', '.join(exts)})")
    console.print(f"  {len(groups) + 1}. [All types]")

    response = Prompt.ask(
        "Enter numbers (comma-separated) or 'all'",
        default="all",
    )

    if response.lower() == "all" or response == str(len(groups) + 1):
        all_exts: set[str] = set()
        for exts in EXTENSION_GROUPS.values():
            all_exts.update(exts)
        return all_exts

    selected: set[str] = set()
    for part in response.split(","):
        try:
            idx = int(part.strip()) - 1
            if 0 <= idx < len(groups):
                selected.update(groups[idx][1])
        except ValueError:
            continue

    if not selected:
        # Default to Python and TypeScript if no selection
        return set(EXTENSION_GROUPS["Python"] + EXTENSION_GROUPS["TypeScript"])

    return selected


def get_provider_instance(
    provider: str, model: str
) -> OpenAICompatProvider | JulesProvider:
    """Create provider instance based on selection."""
    if provider == "jules":
        return JulesProvider()
    return OpenAICompatProvider(model=model)


def run_interactive() -> Config:
    """Run interactive configuration prompts."""
    console.rule("[bold blue]Cortex Code Review CLI")
    console.print("Interactive configuration mode\n")

    project_root = Path.cwd()

    # Provider selection
    provider = select_provider()
    model = select_model(provider)

    # Directory selection
    directories = select_directories(project_root)

    # Extension selection
    extensions = select_extensions()

    # Additional options
    console.print("\n[bold cyan]Additional Options:[/bold cyan]")
    skip_tests = Confirm.ask("Skip test files?", default=False)
    dry_run = Confirm.ask("Dry run (preview only)?", default=False)
    workers = max(1, IntPrompt.ask("Concurrent workers", default=4))

    incremental_save = False
    output_file = "review_report.json"  # Default value
    if not dry_run:
        incremental_save = Confirm.ask("Save results incrementally?", default=True)
        output_file = Prompt.ask("Output report file", default="review_report.json")

    # Build config
    return Config(
        project_root=project_root,
        scan=ScanConfig(
            directories=directories,
            extensions=extensions,
            skip_tests=skip_tests,
        ),
        review=ReviewConfig(
            provider=provider,
            model=model,
            max_workers=workers,
            dry_run=dry_run,
            incremental_save=incremental_save,
            output_file=output_file,
        ),
    )


def print_config_summary(config: Config) -> None:
    """Print configuration summary."""
    console.print()
    console.rule("[bold]Configuration Summary")

    table = Table(show_header=False, box=None)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Provider", config.review.provider)
    table.add_row("Model", config.review.model)
    table.add_row("Directories", ", ".join(str(d) for d in config.scan.directories))
    table.add_row("Extensions", ", ".join(sorted(config.scan.extensions)))
    table.add_row("Skip Tests", str(config.scan.skip_tests))
    table.add_row("Workers", str(config.review.max_workers))
    table.add_row("Dry Run", str(config.review.dry_run))
    if not config.review.dry_run:
        table.add_row("Output File", config.review.output_file)
        table.add_row("Incremental Save", str(config.review.incremental_save))

    console.print(table)
    console.print()


async def run_review(config: Config) -> None:
    """Execute the code review."""
    # Create components
    scanner = FileScanner(config.project_root, config.scan)
    provider = get_provider_instance(config.review.provider, config.review.model)
    reviewer = CodeReviewer(provider, scanner, config)
    reporter = Reporter(config.review.output_file, config.project_root)

    try:

        async def _save_incremental_result(_: ReviewResult) -> None:
            try:
                summary = reviewer.get_summary()
                reporter.save_json(reviewer.results, summary)
            except OSError as e:
                console.print(
                    f"[yellow]Warning: Incremental save failed: {e}[/yellow]"
                )

        on_result = None
        if not config.review.dry_run and config.review.incremental_save:
            on_result = _save_incremental_result

        # Run review
        results = await reviewer.run(on_result=on_result)

        if not config.review.dry_run and results:
            try:
                summary = reviewer.get_summary()
                reporter.save_json(results, summary)
                reporter.print_summary(results, summary)
            except OSError as e:
                console.print(f"[red]Error saving report: {e}[/red]")
            except Exception as e:
                console.print(f"[red]Error generating summary: {e}[/red]")
    finally:
        try:
            await provider.close()
        except Exception as e:
            console.print(f"[yellow]Warning: Provider cleanup failed: {e}[/yellow]")


def main() -> None:
    """Main entry point."""
    from dotenv import load_dotenv

    load_dotenv()
    try:
        config = run_interactive()
        print_config_summary(config)

        if not Confirm.ask("Proceed with review?", default=True):
            console.print("[yellow]Cancelled.[/yellow]")
            return

        # Handle nested asyncio loops
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            console.print("[yellow]Running within an existing event loop.[/yellow]")
            loop.create_task(run_review(config))
        else:
            asyncio.run(run_review(config))

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
