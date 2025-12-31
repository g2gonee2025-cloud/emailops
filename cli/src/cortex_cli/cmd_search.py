"""
Search command for Cortex CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import httpx

from .style import colorize

# Make this configurable
API_BASE_URL = os.getenv("CORTEX_API_URL", "http://127.0.0.1:8000/api/v1")


def setup_search_parser(
    subparsers: Any,
) -> None:
    """Setup search command parser."""
    search_parser = subparsers.add_parser(
        "search",
        help="Search indexed emails with natural language",
        description="""
Search your indexed emails using natural language queries.
Examples:
  cortex search "contract renewal terms"
  cortex search "emails from John about budget" --top-k 20
  cortex search "attachments mentioning quarterly report"
Uses an API call to the backend search service.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    search_parser.add_argument(
        "query",
        metavar="QUERY",
        help="Natural language search query",
    )
    search_parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=10,
        metavar="N",
        help="Number of results to return (default: 10)",
    )
    search_parser.add_argument(
        "--tenant",
        "-t",
        default="default",
        metavar="ID",
        help="Tenant ID (default: 'default')",
    )
    search_parser.add_argument(
        "--fusion",
        choices=["rrf", "weighted_sum"],
        default="rrf",
        help="Fusion method: rrf (default) or weighted_sum",
    )
    search_parser.add_argument(
        "--debug",
        action="store_true",
        help="Show detailed score breakdown (F/V/L)",
    )
    search_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    search_parser.set_defaults(func=_run_search_command)


def _display_search_results(
    data: dict[str, Any], query: str, top_k: int, debug: bool
) -> None:
    """Display search results in a human-readable format."""
    results = data.get("results", [])
    if not isinstance(results, list):
        results = []

    def _safe_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _safe_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    display_count = min(len(results), max(top_k, 0))
    if results:
        print(
            f"  {colorize('✓', 'green')} Found {len(results)} result(s), "
            f"showing {display_count}:\n"
        )
        for i, r in enumerate(results[:display_count], 1):
            score_value = _safe_float(r.get("score")) or 0.0
            content = _safe_text(r.get("content"))
            highlights_value = r.get("highlights", [])
            highlights = highlights_value if isinstance(highlights_value, list) else []
            highlight_text = _safe_text(highlights[0]) if highlights else ""
            text = content[:200] if content else highlight_text[:200]
            metadata = (
                r.get("metadata", {}) if isinstance(r.get("metadata"), dict) else {}
            )
            chunk_id = r.get("chunk_id") or r.get("message_id", "unknown")
            chunk_type = metadata.get("chunk_type", "unknown")

            # Score display
            score_str = f"{score_value:.4f}"
            if debug:
                fusion_score = _safe_float(r.get("fusion_score")) or 0.0
                vec_score = _safe_float(r.get("vector_score")) or 0.0
                lex_score = _safe_float(r.get("lexical_score")) or 0.0
                score_str += (
                    f" (F:{fusion_score:.3f} V:{vec_score:.3f} L:{lex_score:.3f})"
                )

            print(
                f"  {colorize(f'[{i}]', 'bold')} Score: {colorize(score_str, 'cyan')}"
            )
            print(f"      Source: {colorize(str(chunk_id), 'dim')} ({chunk_type})")
            preview = text.replace(chr(10), " ")
            print(f"      {preview}...")
            if debug and highlights:
                safe_highlights = [_safe_text(item) for item in highlights[:2]]
                print(f"      Highlights: {safe_highlights}")
            print()
    else:
        print(f"  {colorize('○', 'yellow')} No results found for: {query}")

    query_time = _safe_float(data.get("query_time_ms"))
    if query_time is None:
        print("  Query time: unknown")
    else:
        print(f"  Query time: {query_time:.2f} ms")
    print()


def _run_search_command(args: argparse.Namespace) -> None:
    """Run search command by calling the search API."""
    json_output = bool(getattr(args, "json", False))
    query = getattr(args, "query", "") or ""
    try:
        top_k = int(getattr(args, "top_k", 10))
    except (TypeError, ValueError):
        top_k = 10
    if top_k <= 0:
        top_k = 10

    if not json_output:
        from cortex_cli.main import _print_banner

        _print_banner()
        print(f"{colorize('▶ SEARCH', 'bold')}\n")
        print(f"  Query:   {colorize(query, 'cyan')}")
        print(f"  Top K:   {colorize(str(top_k), 'dim')}")
        if getattr(args, "debug", False):
            print(f"  {colorize('DEBUG MODE', 'yellow')}")
        print()

    headers = {
        "X-Tenant-ID": getattr(args, "tenant", "default"),
        "X-User-ID": "cli-user",
    }

    payload = {
        "query": query,
        "k": top_k,
        "filters": {},
        "fusion_method": getattr(args, "fusion", "rrf"),
    }

    if not json_output:
        print(f"  {colorize('⏳', 'yellow')} Searching via API...\n")

    try:
        with httpx.Client(base_url=API_BASE_URL, headers=headers) as client:
            response = client.post("search", json=payload, timeout=60)
            response.raise_for_status()
            try:
                results_data = response.json()
            except ValueError as exc:
                raise RuntimeError("API response was not valid JSON.") from exc
            if not isinstance(results_data, dict):
                raise RuntimeError("API response had unexpected structure.")

        if json_output:
            print(json.dumps(results_data, indent=2))
        else:
            _display_search_results(
                results_data, query, top_k, bool(getattr(args, "debug", False))
            )

    except httpx.RequestError as e:
        msg = f"API request failed: {e}"
        if json_output:
            print(json.dumps({"error": msg, "success": False}))
        else:
            print(f"\n  {colorize('ERROR:', 'red')} {msg}")
            print(f"  Is the Cortex API server running at {API_BASE_URL}?")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        msg = f"API returned an error: {e.response.status_code} {e.response.reason_phrase}"
        detail = None
        try:
            payload = e.response.json()
        except (ValueError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            detail = payload.get("detail")

        if json_output:
            print(json.dumps({"error": msg, "detail": detail, "success": False}))
        else:
            print(f"\n  {colorize('ERROR:', 'red')} {msg}")
            if detail:
                print(f"  Detail: {detail}")
        sys.exit(1)
    except Exception as e:
        if json_output:
            print(json.dumps({"error": str(e), "success": False}))
        else:
            print(f"\n  {colorize('ERROR:', 'red')} {e}")
        sys.exit(1)
