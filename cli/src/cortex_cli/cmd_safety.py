"""
Cortex CLI - Safety Commands.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

from cortex_cli.style import colorize as _colorize

logger = logging.getLogger(__name__)


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


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        safe_text = text.encode("ascii", errors="replace").decode("ascii")
        print(safe_text)


def _format_percent(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "n/a"


def _run_grounding_check(args: argparse.Namespace) -> None:
    """Run the grounding check tool."""
    try:
        _ensure_backend_on_path()
        grounding_module = import_module("cortex.safety.grounding")
        GroundingCheckInput = grounding_module.GroundingCheckInput
        tool_check_grounding = grounding_module.tool_check_grounding

        json_output = bool(getattr(args, "json", False))
        answer = getattr(args, "answer", "") or ""
        facts_value = getattr(args, "facts", []) or []
        facts = list(facts_value) if isinstance(facts_value, (list, tuple)) else []
        use_llm = bool(getattr(args, "use_llm", False))

        preview = answer[:80]
        if len(answer) > 80:
            preview = f"{preview}..."

        if not json_output:
            _safe_print(f"{_colorize('▶ GROUNDING CHECK', 'bold')}\n")
            _safe_print(f"  Answer:  {_colorize(preview, 'cyan')}")
            _safe_print(f"  Facts:   {_colorize(f'{len(facts)} provided', 'dim')}")
            _safe_print(f"  LLM Mode: {'Enabled' if use_llm else 'Disabled'}")
            _safe_print("")
            _safe_print(f"  {_colorize('⏳', 'yellow')} Analyzing...")

        check_input = GroundingCheckInput(
            answer_candidate=answer,
            facts=facts,
            use_llm=use_llm,
        )
        result = tool_check_grounding(check_input)

        if json_output:
            print(result.model_dump_json(indent=2))
        else:
            is_grounded_text = (
                _colorize("✓ GROUNDED", "green")
                if result.is_grounded
                else _colorize("✗ NOT GROUNDED", "red")
            )
            _safe_print(f"\n{_colorize('RESULT:', 'bold')} {is_grounded_text}")
            _safe_print(f"  Confidence:       {_format_percent(result.confidence)}")
            _safe_print(
                f"  Supported Claims: {_format_percent(result.grounding_ratio)}"
            )
            _safe_print(f"  Method:           {result.method}")

            if result.unsupported_claims:
                _safe_print(f"\n{_colorize('UNSUPPORTED CLAIMS:', 'yellow')}")
                for i, claim in enumerate(result.unsupported_claims, 1):
                    _safe_print(f"  {i}. {claim}")
            _safe_print("")

    except ImportError as exc:
        if exc.name not in {
            "cortex",
            "cortex.safety",
            "cortex.safety.grounding",
        }:
            raise
        message = "Unable to load the grounding tool. Check your environment."
        if getattr(args, "json", False):
            print(json.dumps({"error": message}, indent=2))
        else:
            _safe_print(f"\n  {_colorize('ERROR:', 'red')} {message}")
        sys.exit(1)
    except Exception:
        logger.exception("Grounding check failed.")
        if getattr(args, "json", False):
            print(json.dumps({"error": "Grounding check failed."}, indent=2))
        else:
            _safe_print(f"\n  {_colorize('ERROR:', 'red')} Grounding check failed.")
        sys.exit(1)


def setup_safety_parser(
    subparsers: Any,
) -> None:
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
