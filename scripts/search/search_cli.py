#!/usr/bin/env python3
"""
Unified Search CLI for Cortex.
Uses the actual backend/src/cortex/retrieval/hybrid_search.py logic.
"""
import argparse
import sys
from pathlib import Path

# Add backend/src to path
# scripts/search/search_cli.py -> depth 2
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir / "backend" / "src"))

from cortex.retrieval.hybrid_search import (  # noqa: E402
    KBSearchInput,
    tool_kb_search_hybrid,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run hybrid search using Cortex backend."
    )
    parser.add_argument("query", help="Search query string")
    parser.add_argument("--limit", "-k", type=int, default=10, help="Number of results")
    parser.add_argument("--tenant", default="default", help="Tenant ID")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show full content"
    )

    args = parser.parse_args()

    try:
        print(f"Searching for: '{args.query}' (Tenant: {args.tenant})")
        print("-" * 60)

        # Construct input object
        input_args = KBSearchInput(
            query=args.query, tenant_id=args.tenant, user_id="cli-user", k=args.limit
        )

        # Run search
        results = tool_kb_search_hybrid(input_args)

        # Results is a SearchResults object with a 'results' list
        # (Based on hybrid_search.py SearchResults definition: results: List[SearchResultItem])
        items = results.results

        print(f"Found {len(items)} results.\n")

        for i, res in enumerate(items, 1):
            score = res.score
            chunk_id = res.chunk_id
            # SearchResultItem has 'content' (optional) and 'snippet' (str)
            text = res.content or res.snippet or ""
            meta = res.metadata or {}

            print(f"{i}. [Score={score:.4f}] {chunk_id}")
            if args.verbose:
                print(f"   Text: {text}")
                print(f"   Meta: {meta}")
            else:
                preview = (text or "")[:150].replace("\n", " ") + "..."
                print(f"   Text: {preview}")
            print()

    except Exception as e:
        print(f"Search failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
