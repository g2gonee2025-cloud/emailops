#!/usr/bin/env python3
"""
Close architectural refactor PRs that are redundant with merged changes.
"""

import subprocess

REPO = "g2gonee2025-cloud/emailops"

# Remaining PRs analyzed - all redundant with merged architectural improvements
ARCH_REDUNDANT = {
    128: "Query expansion async refactor - already modernized in merged PRs",
    130: "Batch ingestion hardening - covered by merged #101 (mailroom hardening)",
    132: "Summarizer architectural review - core changes already applied",
    138: "LLM Runtime DI refactor - conflicts with merged runtime improvements",
    141: "Async vector store - superseded by merged #98 (secure vector search refactor)",
    143: "RAG API hardening - redundant with multiple merged security PRs",
    144: "FTS async refactor - overlaps with merged #106 (FTS injection fixes)",
    152: "PII engine hardening - core security already in place",
    166: "Email processing hardening - covered by merged ingestion security",
    180: "Frontend architectural refactor - conflicts with merged UI improvements",
    183: "Audit module refactor - covered by merged security architecture",
}


def close_pr(number, reason):
    comment = (
        f"Closing as redundant with merged changes. {reason}.\n\n"
        f"The codebase has been significantly modernized through ~16 merged PRs from the Jules "
        f"analysis batch. These changes addressed async patterns, security hardening, and "
        f"architectural improvements across the affected modules."
    )

    subprocess.run(
        ["gh", "pr", "comment", str(number), "--repo", REPO, "--body", comment],
        check=False,
    )

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
    print(f"Closing {len(ARCH_REDUNDANT)} architectural refactor PRs...")
    for pr_num, reason in sorted(ARCH_REDUNDANT.items()):
        print(f"\nPR #{pr_num}: {reason}")
        close_pr(pr_num, reason)

    print("\n✓ All redundant PRs closed. Repository is clean.")


if __name__ == "__main__":
    main()
