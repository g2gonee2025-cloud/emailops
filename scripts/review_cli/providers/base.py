"""Abstract base for LLM review providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ReviewResult:
    """Result from a code review."""

    file: str
    issues: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    model: str = ""
    language: str = ""
    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None

    @property
    def has_issues(self) -> bool:
        return bool(self.issues)

    @property
    def is_success(self) -> bool:
        return self.error is None and not self.skipped


class ReviewProvider(ABC):
    """Abstract base class for review providers."""

    name: str = "base"

    @abstractmethod
    async def review_file(
        self,
        file_path: Path,
        content: str,
        context: str,
        language: str,
    ) -> ReviewResult:
        """Review a single file and return results."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        ...
