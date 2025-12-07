#!/usr/bin/env python3
"""Blueprint compliance checks for EmailOps.

This script enforces a small set of guardrails derived from docs/CANONICAL_BLUEPRINT.md.
Extend cautiously to avoid false positives; prefer high-signal checks only.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BLUEPRINT_PATH = REPO_ROOT / "docs" / "CANONICAL_BLUEPRINT.md"

# Required anchors to ensure the repo matches the canonical layout.
REQUIRED_PATHS: list[tuple[Path, str]] = [
    (BLUEPRINT_PATH, "Canonical blueprint must exist"),
    (REPO_ROOT / "backend" / "src" / "cortex", "Backend cortex package must exist"),
    (REPO_ROOT / "backend" / "migrations", "Alembic migrations directory must exist"),
    (REPO_ROOT / "workers" / "src" / "cortex_workers", "Workers package must exist"),
    (REPO_ROOT / "cli" / "src" / "cortex_cli", "CLI package must exist"),
    (
        REPO_ROOT / ".github" / "copilot-instructions.md",
        "Copilot instructions must exist",
    ),
]


def collect_missing(paths: Iterable[tuple[Path, str]]) -> list[str]:
    missing: list[str] = []
    for path, description in paths:
        if not path.exists():
            try:
                rel = path.relative_to(REPO_ROOT)
            except ValueError:
                rel = path
            missing.append(f"{description}: missing {rel}")
    return missing


def check_blueprint_content(path: Path) -> list[str]:
    # Existence is already validated via REQUIRED_PATHS; skip duplicate error.
    if not path.exists():
        return []

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive
        return [f"Blueprint unreadable: {exc}"]

    issues: list[str] = []
    if not content.strip():
        issues.append("Blueprint is empty")
    if "Canonical Source of Truth" not in content:
        issues.append("Blueprint is missing 'Canonical Source of Truth' marker")
    return issues


def main() -> int:
    failures: list[str] = []
    failures.extend(collect_missing(REQUIRED_PATHS))
    failures.extend(check_blueprint_content(BLUEPRINT_PATH))

    if failures:
        print("Blueprint verification failed:")
        for item in failures:
            print(f" - {item}")
        return 1

    print("Blueprint verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
