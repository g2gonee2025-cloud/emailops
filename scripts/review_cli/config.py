"""Configuration for Review CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ScanConfig(BaseModel):
    """Configuration for file scanning."""

    directories: list[Path] = Field(default_factory=list)
    extensions: list[str] = Field(
        default_factory=lambda: sorted([".py", ".ts", ".tsx", ".js", ".jsx"])
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
    exclude_files: list[str] = Field(
        default_factory=lambda: sorted(
            [
                "package-lock.json",
                "yarn.lock",
                "pnpm-lock.yaml",
                "poetry.lock",
            ]
        )
    )
    skip_tests: bool = False
    min_lines: int = Field(default=0, ge=0)
    max_file_size: int = Field(default=50000, ge=0)


class ReviewConfig(BaseModel):
    """Configuration for code review."""

    provider: Literal["openai", "jules"] = "openai"
    model: str = "openai-gpt-5"
    max_workers: int = Field(default=4, gt=0)
    dry_run: bool = False
    output_file: str = Field(default="review_report.json")
    incremental_save: bool = True

    @model_validator(mode="after")
    def validate_model(self) -> ReviewConfig:
        """Validate that the model is valid for the selected provider."""
        provider_models = PROVIDER_MODELS.get(self.provider)
        if not provider_models:
            raise ValueError(f"Unknown provider: {self.provider}")
        if self.model not in provider_models:
            raise ValueError(
                f"Invalid model '{self.model}' for provider '{self.provider}'. "
                f"Available models: {', '.join(provider_models)}"
            )
        return self


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
