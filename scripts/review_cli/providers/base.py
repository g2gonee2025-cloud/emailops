"""Abstract base for LLM review providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Issue:
    """A single issue found in a file."""

    line_number: Optional[int]
    message: str
    severity: str


@dataclass
class ReviewResult:
    """Result from a code review."""

    file: Path
    issues: List[Issue] = field(default_factory=list)
    summary: str = ""
    model: str = ""
    language: str = ""
    error: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None

    def __post_init__(self):
        if self.skipped and not self.skip_reason:
            raise ValueError("skip_reason must be provided if skipped is True")

    @property
    def has_issues(self) -> bool:
        return bool(self.issues)

    @property
    def is_success(self) -> bool:
        return self.error is None and not self.skipped and not self.has_issues


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
