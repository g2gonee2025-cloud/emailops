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

from cortex.domain.models import KBSearchInput  # noqa: E402
from cortex.tools.search import tool_kb_search_hybrid  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Run hybrid search using Cortex backend."
    )
    parser.add_argument("query", help="Search query string")
    parser.add_argument("--limit", "-k", type=int, default=10, help="Number of results")
    parser.add_argument("--tenant", default="default", help="Tenant ID")
    args = parser.parse_args()

    try:
        # Construct input object
        input_args = KBSearchInput(
            query=args.query,
            tenant_id=args.tenant,
            user_id="cli-user",
            limit=args.limit,
            fusion_strategy="rrf",
        )

        # Run search
        results = tool_kb_search_hybrid(input_args)
        print(results.model_dump_json(indent=2))

    except Exception as e:
        print(f"Search failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
