"""CLI entry point for `python -m scripts.review_cli`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console

def _setup_sys_path() -> None:
    """Add project root to sys.path to allow for absolute imports."""
    _project_root = Path(__file__).resolve().parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))


try:
    from scripts.review_cli.cli import main as interactive_main
    from scripts.review_cli.cli import print_config_summary, run_review
    from scripts.review_cli.config import (
        PROVIDER_MODELS,
        Config,
        ReviewConfig,
        ScanConfig,
    )
except ImportError:
    _setup_sys_path()
    from scripts.review_cli.cli import main as interactive_main
    from scripts.review_cli.cli import print_config_summary, run_review
    from scripts.review_cli.config import (
        PROVIDER_MODELS,
        Config,
        ReviewConfig,
        ScanConfig,
    )

console = Console()


def _normalize_extensions(extensions: list[str]) -> set[str]:
    normalized: set[str] = set()
    for ext in extensions:
        cleaned = ext.strip().lower()
        if not cleaned:
            continue
        if not cleaned.startswith("."):
            cleaned = f".{cleaned}"
        normalized.add(cleaned)
    return normalized


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cortex Code Review CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python -m scripts.review_cli

  # Direct invocation
  python -m scripts.review_cli --provider openai --model openai-gpt-5 --dirs backend/src --ext .py

  # Dry run
  python -m scripts.review_cli --dry-run --dirs frontend/src --ext .ts .tsx

  # Jules provider
  JULES_API_KEY=xxx python -m scripts.review_cli --provider jules
        """,
    )

    parser.add_argument(
        "--provider",
        choices=["openai", "jules"],
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model",
        help="Model name (provider-specific)",
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        help="Directories to scan (relative to project root)",
    )
    parser.add_argument(
        "--ext",
        nargs="+",
        help="File extensions to include (e.g., .py .ts)",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip test files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent workers (default: 8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview files without sending to API",
    )
    parser.add_argument(
        "--output",
        default="review_report.json",
        help="Output file for JSON report (default: review_report.json)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> Config:
    """Build configuration from CLI arguments."""
    project_root = Path.cwd()

    if args.workers < 1:
        raise ValueError("Workers must be >= 1")

    # Resolve and validate directories
    directories = []
    if args.dirs:
        for d in args.dirs:
            path = Path(d)
            if not path.is_absolute():
                path = project_root / path
            if not path.exists():
                raise FileNotFoundError(f"Directory not found: {path}")
            if not path.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {path}")
            directories.append(path)
    else:
        # Default directories
        for default in ["backend/src", "frontend/src", "cli/src", "scripts"]:
            path = project_root / default
            if path.exists() and path.is_dir():
                directories.append(path)

    if not directories:
        raise ValueError("No valid directories to scan.")

    # Resolve extensions
    extensions = (
        _normalize_extensions(args.ext)
        if args.ext
        else {".py", ".ts", ".tsx", ".js", ".jsx"}
    )

    # Model defaults and validation
    provider = args.provider or "openai"
    model = args.model
    if not model:
        models = PROVIDER_MODELS.get(provider, [])
        if not models:
            raise ValueError(
                f"No default models found for provider '{provider}'. Please specify a model."
            )
        model = models[0]

    return Config(
        project_root=project_root,
        scan=ScanConfig(
            directories=directories,
            extensions=extensions,
            skip_tests=args.skip_tests,
        ),
        review=ReviewConfig(
            provider=provider,
            model=model,
            max_workers=args.workers,
            dry_run=args.dry_run,
            output_file=args.output,
        ),
    )


def main() -> None:
    """Main entry point."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        # python-dotenv is not installed, which is fine.
        # It's an optional dependency for this script.
        pass

    # If no CLI args are provided (other than the script name), run interactive mode.
    if len(sys.argv) == 1:
        interactive_main()
        return

    try:
        args = parse_args()

        # Configure logging
        log_level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )

        config = build_config_from_args(args)
        print_config_summary(config)
        asyncio.run(run_review(config))

    except (ValueError, FileNotFoundError, NotADirectoryError) as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        logging.exception("Unhandled exception in main")
        sys.exit(1)


if __name__ == "__main__":
    main()
