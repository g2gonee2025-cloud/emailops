#!/usr/bin/env python3
"""
Close redundant PRs that overlap with already-merged changes.
"""

import json
import subprocess

REPO = "g2gonee2025-cloud/emailops"

# PRs to close with reasons
REDUNDANT_PRS = {
    103: "Superseded by earlier auth hardening merges (#96, #98)",
    105: "Draft endpoint security already addressed in prior merges",
    109: "Security architecture changes already incorporated",
    112: "Ingestion API hardening complete in merged PRs",
    114: "Summarize endpoint hardening already applied",
    116: "Draft email endpoint changes redundant",
    118: "RAG API security & PII fixes already merged",
    125: "Chat endpoint hardening complete",
    126: "Search API hardening complete",
}


def close_pr(number, reason):
    comment = (
        f"Closing as redundant: {reason}. Core changes already merged in earlier batch."
    )

    # Add comment
    subprocess.run(
        ["gh", "pr", "comment", str(number), "--repo", REPO, "--body", comment],
        check=False,
    )

    # Close PR
    result = subprocess.run(
        ["gh", "pr", "close", str(number), "--repo", REPO, "--delete-branch"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print(f"✓ Closed PR #{number}")
    else:
        print(f"✗ Failed to close PR #{number}: {result.stderr}")


def main():
    print(f"Closing {len(REDUNDANT_PRS)} redundant PRs...")

    for pr_num, reason in REDUNDANT_PRS.items():
        print(f"\nPR #{pr_num}: {reason}")
        close_pr(pr_num, reason)

    print("\nDone. Remaining PRs require manual review.")


if __name__ == "__main__":
    main()
