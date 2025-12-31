"""Configuration for Review CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ScanConfig(BaseModel):
    """Configuration for file scanning."""

    directories: list[Path] = Field(default_factory=list)
    extensions: set[str] = Field(
        default_factory=lambda: {".py", ".ts", ".tsx", ".js", ".jsx"}
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            "__pycache__",
            ".pytest_cache",
            "node_modules",
            ".git",
            "venv",
            ".venv",
            "dist",
            "build",
            ".next",
            "coverage",
            ".mypy_cache",
            ".ruff_cache",
            "egg-info",
        ]
    )
    exclude_files: set[str] = Field(
        default_factory=lambda: {
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            "poetry.lock",
        }
    )
    skip_tests: bool = False
    min_lines: int = 0
    max_file_size: int = 50000


class ReviewConfig(BaseModel):
    """Configuration for code review."""

    provider: Literal["openai", "jules"] = "openai"
    model: str = "openai-gpt-5"
    max_workers: int = 4
    dry_run: bool = False
    output_file: str = "review_report.json"
    incremental_save: bool = True


class Config(BaseModel):
    """Combined configuration."""

    scan: ScanConfig = Field(default_factory=ScanConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    project_root: Path = Field(default_factory=Path.cwd)


# Default directories to scan
DEFAULT_SCAN_DIRS = [
    "backend/src",
    "frontend/src",
    "cli/src",
    "scripts",
    "workers/src",
]

# Language extensions grouped
EXTENSION_GROUPS = {
    "Python": [".py"],
    "TypeScript": [".ts", ".tsx"],
    "JavaScript": [".js", ".jsx"],
    "Styles": [".css", ".scss"],
    "Config": [".yaml", ".yml", ".json", ".toml"],
    "Shell": [".sh"],
    "SQL": [".sql"],
}

# Available models by provider
PROVIDER_MODELS = {
    "openai": ["openai-gpt-5", "openai-gpt-oss-120b"],
    "jules": [
        "jules-default",
    ],
}
