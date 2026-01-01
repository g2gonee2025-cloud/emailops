"""Grounding check CLI command."""

import argparse
import logging
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)

try:
    from cortex.safety.grounding import GroundingCheckInput as GroundingCheckInput
    from cortex.safety.grounding import tool_check_grounding as tool_check_grounding
except Exception:
    GroundingCheckInput = None
    tool_check_grounding = None


def _find_backend_src() -> Path | None:
    current = Path(__file__).resolve()
    for parent in (current, *current.parents):
        candidate = parent / "backend" / "src" / "cortex" / "__init__.py"
        if candidate.is_file():
            return candidate.parent.parent
    return None


def _ensure_backend_on_path() -> None:
    backend_src = _find_backend_src()
    if backend_src is None:
        return
    backend_str = str(backend_src)
    if backend_str not in sys.path:
        sys.path.insert(0, backend_str)


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def setup_grounding_parser(
    parser: Any,
) -> None:
    """Add grounding command to parser."""
    grounding_parser = parser.add_parser(
        "grounding", help="Tools for checking answer grounding"
    )
    grounding_subparsers = grounding_parser.add_subparsers(dest="subcommand")

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
    grounding_parser.set_defaults(func=_default_grounding_handler)


def _default_grounding_handler(args: argparse.Namespace) -> None:
    if not getattr(args, "subcommand", None):
        print("Please specify a grounding subcommand. Use --help for details.")


def run_grounding_check(args: argparse.Namespace) -> None:
    """Run the grounding check tool and display results."""
    console = Console()

    try:
        _ensure_backend_on_path()
        global GroundingCheckInput, tool_check_grounding
        if GroundingCheckInput is None or tool_check_grounding is None:
            grounding_module = import_module("cortex.safety.grounding")
            GroundingCheckInput = grounding_module.GroundingCheckInput
            tool_check_grounding = grounding_module.tool_check_grounding

        input_data = GroundingCheckInput(
            answer_candidate=getattr(args, "answer", ""),
            facts=getattr(args, "facts", []),
            use_llm=bool(getattr(args, "use_llm", False)),
        )
        result = tool_check_grounding(input_data)

        console.print("\n[bold]Grounding Check Results[/bold]")
        console.print("---")

        status = (
            "[bold green]GROUNDED[/bold green]"
            if result.is_grounded
            else "[bold red]NOT GROUNDED[/bold red]"
        )
        console.print(f"Overall Status: {status}")
        console.print(f"Confidence: {_format_float(result.confidence)}")
        console.print(f"Grounding Ratio: {_format_float(result.grounding_ratio)}")

        analyses_value = getattr(result, "claim_analyses", None)
        analyses = analyses_value if isinstance(analyses_value, list) else []
        if analyses:
            console.print("\n[bold]Claim-by-Claim Analysis[/bold]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Claim", style="dim", width=50)
            table.add_column("Supported", justify="center")
            table.add_column("Confidence", justify="right")
            table.add_column("Supporting Fact")

            for analysis in analyses:
                supported = bool(getattr(analysis, "is_supported", False))
                supported_icon = "✅" if supported else "❌"
                claim_text = _safe_text(getattr(analysis, "claim", ""))
                confidence_text = _format_float(getattr(analysis, "confidence", None))
                supporting_fact = _safe_text(getattr(analysis, "supporting_fact", None))
                table.add_row(
                    claim_text,
                    supported_icon,
                    confidence_text,
                    supporting_fact or "N/A",
                )
            console.print(table)
        else:
            console.print(
                "\n[yellow]No verifiable claims were extracted from the answer.[/yellow]"
            )

    except ImportError as exc:
        if exc.name not in {
            "cortex",
            "cortex.safety",
            "cortex.safety.grounding",
        }:
            raise
        console.print(f"[bold red]Error[/bold red]: {exc}")
        console.print("Please ensure all necessary backend dependencies are installed.")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Grounding check failed.")
        console.print("[bold red]An unexpected error occurred[/bold red]: " f"{exc}")
        sys.exit(1)
