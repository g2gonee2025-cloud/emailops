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
    subparsers: argparse._SubParsersAction,
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
    if results:
        print(f"  {colorize('✓', 'green')} Found {len(results)} result(s):\n")
        for i, r in enumerate(results[:top_k], 1):
            score = r.get("score", 0)
            content = r.get("content", "") or ""
            highlights = r.get("highlights", [])
            text = (
                content[:200]
                if content
                else (highlights[0][:200] if highlights else "")
            )
            metadata = r.get("metadata", {})
            chunk_id = r.get("chunk_id") or r.get("message_id", "unknown")
            chunk_type = metadata.get("chunk_type", "unknown")

            # Score display
            score_str = f"{score:.4f}"
            if debug:
                fusion_score = r.get("fusion_score") or 0
                vec_score = r.get("vector_score") or 0
                lex_score = r.get("lexical_score") or 0
                score_str += (
                    f" (F:{fusion_score:.3f} V:{vec_score:.3f} L:{lex_score:.3f})"
                )

            print(
                f"  {colorize(f'[{i}]', 'bold')} Score: {colorize(score_str, 'cyan')}"
            )
            print(f"      Source: {colorize(str(chunk_id), 'dim')} ({chunk_type})")
            print(f"      {text.replace(chr(10), ' ')}...")
            if debug and highlights:
                print(f"      Highlights: {highlights[:2]}")
            print()
    else:
        print(f"  {colorize('○', 'yellow')} No results found for: {query}")

    query_time = data.get("query_time_ms", 0.0)
    print(f"  Query time: {query_time:.2f} ms")
    print()


def _run_search_command(args: argparse.Namespace) -> None:
    """Run search command by calling the search API."""
    if not args.json:
        from cortex_cli.main import _print_banner

        _print_banner()
        print(f"{colorize('▶ SEARCH', 'bold')}\n")
        print(f"  Query:   {colorize(args.query, 'cyan')}")
        print(f"  Top K:   {colorize(str(args.top_k), 'dim')}")
        if args.debug:
            print(f"  {colorize('DEBUG MODE', 'yellow')}")
        print()

    headers = {
        "X-Tenant-ID": args.tenant,
        "X-User-ID": "cli-user",
    }

    payload = {
        "query": args.query,
        "k": args.top_k,
        "filters": {},
        "fusion_method": args.fusion,
    }

    if not args.json:
        print(f"  {colorize('⏳', 'yellow')} Searching via API...\n")

    try:
        with httpx.Client(base_url=API_BASE_URL, headers=headers) as client:
            response = client.post("/search", json=payload, timeout=60)
            response.raise_for_status()
            results_data = response.json()

        if args.json:
            print(json.dumps(results_data, indent=2))
        else:
            _display_search_results(results_data, args.query, args.top_k, args.debug)

    except httpx.RequestError as e:
        msg = f"API request failed: {e}"
        if args.json:
            print(json.dumps({"error": msg, "success": False}))
        else:
            print(f"\n  {colorize('ERROR:', 'red')} {msg}")
            print(f"  Is the Cortex API server running at {API_BASE_URL}?")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        msg = f"API returned an error: {e.response.status_code} {e.response.reason_phrase}"
        try:
            detail = e.response.json().get("detail", "No details provided.")
        except json.JSONDecodeError:
            detail = e.response.text

        if args.json:
            print(json.dumps({"error": msg, "detail": detail, "success": False}))
        else:
            print(f"\n  {colorize('ERROR:', 'red')} {msg}")
            print(f"  Detail: {detail}")
        sys.exit(1)
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e), "success": False}))
        else:
            print(f"\n  {colorize('ERROR:', 'red')} {e}")
            sys.exit(1)
