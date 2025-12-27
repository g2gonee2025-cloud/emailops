"""
Cortex CLI - Safety Commands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure backend package is importable
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_SRC = PROJECT_ROOT / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.append(str(BACKEND_SRC))

from cortex.safety.grounding import (
    GroundingCheckInput,
    tool_check_grounding,
)
from cortex_cli.style import colorize as _colorize


def _run_grounding_check(args: argparse.Namespace) -> None:
    """Run the grounding check tool."""
    if not args.json:
        print(f"{_colorize('▶ GROUNDING CHECK', 'bold')}\n")
        print(f"  Answer:  {_colorize(args.answer[:80] + '...', 'cyan')}")
        print(f"  Facts:   {_colorize(f'{len(args.facts)} provided', 'dim')}")
        print(f"  LLM Mode:{'Enabled' if args.use_llm else 'Disabled'}")
        print()
        print(f"  {_colorize('⏳', 'yellow')} Analyzing...")

    check_input = GroundingCheckInput(
        answer_candidate=args.answer,
        facts=args.facts,
        use_llm=args.use_llm,
    )

    try:
        result = tool_check_grounding(check_input)

        if args.json:
            print(result.model_dump_json(indent=2))
        else:
            is_grounded_text = (
                _colorize("✓ GROUNDED", "green")
                if result.is_grounded
                else _colorize("✗ NOT GROUNDED", "red")
            )
            print(f"\n{_colorize('RESULT:', 'bold')} {is_grounded_text}")
            print(f"  Confidence:       {result.confidence:.2%}")
            print(f"  Supported Claims: {result.grounding_ratio:.2%}")
            print(f"  Method:           {result.method}")

            if result.unsupported_claims:
                print(f"\n{_colorize('UNSUPPORTED CLAIMS:', 'yellow')}")
                for i, claim in enumerate(result.unsupported_claims, 1):
                    print(f"  {i}. {claim}")
            print()

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"\n  {_colorize('ERROR:', 'red')} {e}")
        sys.exit(1)


def setup_safety_parser(subparsers: argparse._SubParsersAction) -> None:
    """Setup the 'safety' command and its subcommands."""
    safety_parser = subparsers.add_parser(
        "safety",
        help="Run safety and verification tools",
        description="A suite of tools for ensuring the safety, factuality, and grounding of LLM outputs.",
    )
    safety_subparsers = safety_parser.add_subparsers(
        dest="safety_command",
        title="Safety Commands",
        metavar="<command>",
    )

    # Grounding check command
    grounding_parser = safety_subparsers.add_parser(
        "grounding",
        help="Check if an answer is grounded in provided facts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Verifies that an LLM-generated answer is factually supported by a list of provided context snippets (facts).

This tool helps mitigate hallucinations by:
  1. Extracting factual claims from the answer.
  2. Comparing each claim against the provided facts using embeddings or an LLM.
  3. Reporting which claims are unsupported.

Example:
  cortex safety grounding "The sky is blue." --facts "The color of the sky is blue due to Rayleigh scattering." "The ocean is also blue."
""",
    )
    grounding_parser.add_argument(
        "answer",
        metavar="ANSWER",
        help="The answer string to verify.",
    )
    grounding_parser.add_argument(
        "--facts",
        nargs="+",
        required=True,
        metavar="FACT",
        help="One or more fact strings to use as context.",
    )
    grounding_parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use a more accurate (but slower) LLM-based check.",
    )
    grounding_parser.add_argument(
        "--json",
        action="store_true",
        help="Output the full analysis as a JSON object.",
    )
    grounding_parser.set_defaults(func=_run_grounding_check)
