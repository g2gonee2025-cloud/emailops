"""Grounding check CLI command."""

import argparse
import sys
import json
from rich.console import Console
from rich.table import Table

# Correcting the import path requires modifying sys.path since this is a script
# This is a common pattern in this codebase.
try:
    from cortex.safety.grounding import (
        GroundingCheckInput,
        tool_check_grounding,
        ClaimAnalysis,
    )
except ImportError:
    # Add backend to sys.path to resolve import
    from pathlib import Path

    backend_path = Path(__file__).resolve().parent.parent.parent.parent / "backend/src"
    sys.path.insert(0, str(backend_path))
    from cortex.safety.grounding import (
        GroundingCheckInput,
        tool_check_grounding,
        ClaimAnalysis,
    )


def setup_grounding_parser(parser: argparse.ArgumentParser):
    """Add grounding command to parser."""
    grounding_parser = parser.add_parser(
        "grounding", help="Tools for checking answer grounding"
    )
    grounding_subparsers = grounding_parser.add_subparsers(
        dest="subcommand", required=True
    )

    check_parser = grounding_subparsers.add_parser(
        "check", help="Check if an answer is grounded in the provided facts"
    )
    check_parser.add_argument(
        "--answer", required=True, help="The answer candidate to verify"
    )
    check_parser.add_argument(
        "--facts",
        required=True,
        nargs="+",
        help="One or more facts to check against",
    )
    check_parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use a Large Language Model for verification (more accurate, requires API access)",
    )
    check_parser.set_defaults(func=run_grounding_check)


def run_grounding_check(args: argparse.Namespace):
    """Run the grounding check tool and display results."""
    console = Console()

    try:
        input_data = GroundingCheckInput(
            answer_candidate=args.answer, facts=args.facts, use_llm=args.use_llm
        )
        result = tool_check_grounding(input_data)

        console.print("\n[bold]Grounding Check Results[/bold]")
        console.print("---")

        status = "[bold green]GROUNDED[/bold green]" if result.is_grounded else "[bold red]NOT GROUNDED[/bold red]"
        console.print(f"Overall Status: {status}")
        console.print(f"Confidence: {result.confidence:.2f}")
        console.print(f"Grounding Ratio: {result.grounding_ratio:.2f}")

        if result.claim_analyses:
            console.print("\n[bold]Claim-by-Claim Analysis[/bold]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Claim", style="dim", width=50)
            table.add_column("Supported", justify="center")
            table.add_column("Confidence", justify="right")
            table.add_column("Supporting Fact")

            for analysis in result.claim_analyses:
                supported_icon = "✅" if analysis.is_supported else "❌"
                table.add_row(
                    analysis.claim,
                    supported_icon,
                    f"{analysis.confidence:.2f}",
                    analysis.supporting_fact or "N/A",
                )
            console.print(table)
        else:
            console.print("\n[yellow]No verifiable claims were extracted from the answer.[/yellow]")


    except ImportError as e:
        console.print(f"[bold red]Error[/bold red]: {e}")
        console.print("Please ensure all necessary backend dependencies are installed.")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred[/bold red]: {e}")
        sys.exit(1)
