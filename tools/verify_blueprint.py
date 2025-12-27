#!/usr/bin/env python3
"""Blueprint compliance checks for EmailOps.

This script enforces a small set of guardrails derived from docs/CANONICAL_BLUEPRINT.md.
Extend cautiously to avoid false positives; prefer high-signal checks only.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Final


def find_repo_root(start_path: Path | None = None) -> Path:
    """Dynamically find the repository root.

    The Cortex project root is defined as the first directory containing both a
    'backend' and 'frontend' subdirectory.
    """
    if start_path is None:
        start_path = Path(__file__).resolve()

    current_path = start_path.parent
    while current_path != current_path.parent:  # Stop at the filesystem root
        if (current_path / "backend").is_dir() and (current_path / "frontend").is_dir():
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError("Could not find repository root.")


REPO_ROOT: Final[Path] = find_repo_root()
BLUEPRINT_PATH: Final[Path] = REPO_ROOT / "docs" / "CANONICAL_BLUEPRINT.md"


def get_required_paths(root: Path) -> list[tuple[Path, str]]:
    """Generate the list of required paths relative to the project root."""
    return [
        (root / "docs" / "CANONICAL_BLUEPRINT.md", "Canonical blueprint must exist"),
        (root / "backend" / "src" / "cortex", "Backend cortex package must exist"),
        (root / "backend" / "migrations", "Alembic migrations directory must exist"),
        (root / "workers" / "src" / "cortex_workers", "Workers package must exist"),
        (root / "cli" / "src" / "cortex_cli", "CLI package must exist"),
        (
            root / ".github" / "copilot-instructions.md",
            "Copilot instructions must exist",
        ),
    ]


def collect_missing(paths: Iterable[tuple[Path, str]], root: Path) -> list[str]:
    """Check for the existence of required paths and return a list of missing ones."""
    missing: list[str] = []
    for path, description in paths:
        if not path.exists():
            try:
                rel = path.relative_to(root)
            except ValueError:
                rel = path
            missing.append(f"{description}: missing {rel}")
    return missing


def check_blueprint_content(path: Path) -> list[str]:
    """Validate the content of the blueprint file."""
    if not path.exists():
        # This check is duplicated in collect_missing, but provides a safeguard.
        return []

    try:
        content = path.read_text(encoding="utf-8")
    except IOError as exc:
        return [f"Blueprint file unreadable: {exc}"]

    issues: list[str] = []
    if not content.strip():
        issues.append("Blueprint is empty")
    if "Canonical Source of Truth" not in content:
        issues.append("Blueprint is missing 'Canonical Source of Truth' marker")
    return issues


def main() -> int:
    """Main function to run all blueprint verifications."""
    required_paths = get_required_paths(REPO_ROOT)

    failures: list[str] = []
    failures.extend(collect_missing(required_paths, REPO_ROOT))
    failures.extend(check_blueprint_content(BLUEPRINT_PATH))

    if failures:
        print("Blueprint verification failed:", file=sys.stderr)
        for item in failures:
            print(f" - {item}", file=sys.stderr)
        return 1

    print("Blueprint verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
