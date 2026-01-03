"""File scanner for discovering code files."""

from __future__ import annotations

import ast
import fnmatch
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.review_cli.config import ScanConfig

logger = logging.getLogger(__name__)


# Language detection by extension
LANGUAGE_MAP = {
    ".py": "Python",
    ".ts": "TypeScript",
    ".tsx": "TypeScript/React",
    ".js": "JavaScript",
    ".jsx": "JavaScript/React",
    ".css": "CSS",
    ".scss": "SCSS",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".json": "JSON",
    ".toml": "TOML",
    ".sh": "Shell/Bash",
    ".sql": "SQL",
}


def _normalize_extension(ext: str) -> str:
    if not isinstance(ext, str):
        return ""
    cleaned = ext.strip().lower()
    if not cleaned:
        return ""
    if not cleaned.startswith("."):
        return f".{cleaned}"
    return cleaned


class FileScanner:
    """Discovers and filters code files for review."""

    def __init__(self, project_root: Path, config: ScanConfig):
        self.project_root = project_root
        self.config = config
        self._extensions: set[str] = set()
        for ext in config.extensions:
            normalized = _normalize_extension(ext)
            if normalized:
                self._extensions.add(normalized)
        self._gitignore_patterns: list[str] | None = None

    def _load_gitignore_patterns(self) -> list[str]:
        """Load patterns from .gitignore file."""
        if self._gitignore_patterns is not None:
            return self._gitignore_patterns

        gitignore_path = self.project_root / ".gitignore"
        patterns = []
        if gitignore_path.exists():
            try:
                with gitignore_path.open(encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            patterns.append(line)
            except OSError as e:
                logger.warning("Could not read .gitignore file: %s", e)
        self._gitignore_patterns = patterns
        return patterns

    def _is_excluded(self, file_path: Path) -> bool:
        """Check if a file should be excluded from scanning."""
        try:
            rel_path = file_path.relative_to(self.project_root).as_posix()
        except ValueError:
            rel_path = file_path.as_posix()

        name = file_path.name

        # Check directory exclusion patterns
        for pattern in self.config.exclude_patterns:
            # If pattern is a directory name (e.g. "node_modules"), check path parts
            if "/" not in pattern and pattern in file_path.parts:
                return True
            # Glob match on full relative path
            if fnmatch.fnmatch(rel_path, pattern):
                return True
            # Glob match on directory prefix
            if fnmatch.fnmatch(rel_path, f"{pattern}/*"):
                return True

        # Check excluded files
        if name in self.config.exclude_files:
            return True

        # Check test files if skip_tests is enabled
        if self.config.skip_tests:
            # More precise test file matching
            is_test = re.search(r"(^|[/_\\\])(tests?|__tests__)([/_\\\]]|$)", rel_path.lower()) or re.search(
                r"(^|[/_])(test_.*|.*_test)\.py$", name.lower()
            )
            if is_test:
                return True

        # Check gitignore patterns
        # Simplified gitignore matching
        for pattern in self._load_gitignore_patterns():
            # Directory match
            if pattern.endswith("/"):
                pat = pattern.rstrip("/")
                if pat in file_path.parts:
                    return True
                if fnmatch.fnmatch(rel_path, f"{pat}/*") or fnmatch.fnmatch(
                    rel_path, f"**/{pat}/*"
                ):
                    return True

            # File/Glob match
            if "/" in pattern:
                if fnmatch.fnmatch(rel_path, pattern.lstrip("/")):
                    return True
            else:
                if fnmatch.fnmatch(name, pattern):
                    return True

        return False

    def _count_lines(self, file_path: Path) -> int:
        """Count lines in a file."""
        try:
            with file_path.open(encoding="utf-8", errors="ignore") as f:
                return sum(1 for _ in f)
        except OSError:
            return 0

    def scan(self) -> list[Path]:
        """Find all code files matching the configuration."""
        found_files: set[Path] = set()

        for scan_dir_entry in self.config.directories:
            # Type guard for safety
            scan_dir = Path(scan_dir_entry) if not isinstance(scan_dir_entry, Path) else scan_dir_entry

            if not scan_dir.is_absolute():
                scan_dir = self.project_root / scan_dir

            if not scan_dir.exists():
                logger.warning("Scan directory does not exist: %s", scan_dir)
                continue

            # Custom walk to respect exclusions at directory level
            for root, dirs, files in os.walk(scan_dir, topdown=True):
                root_path = Path(root)

                # Filter directories in-place to prevent traversal
                dirs[:] = [d for d in dirs if not self._is_dir_excluded(root_path / d)]

                for filename in files:
                    file_path = root_path / filename

                    # Check extension
                    if file_path.suffix.lower() not in self._extensions:
                        continue

                    # Check exclusions
                    if self._is_excluded(file_path):
                        continue

                    # Check file size
                    if self.config.max_file_size > 0:
                        try:
                            if file_path.stat().st_size > self.config.max_file_size:
                                continue
                        except OSError:
                            continue

                    # Check minimum lines
                    if self.config.min_lines > 0:
                        if self._count_lines(file_path) < self.config.min_lines:
                            continue

                    found_files.add(file_path)

        return sorted(list(found_files))

    def _is_dir_excluded(self, dir_path: Path) -> bool:
        """Check if a directory should be fully excluded from traversal."""
        # Check against simple directory name patterns
        for pattern in self.config.exclude_patterns:
            if "/" not in pattern and pattern == dir_path.name:
                return True
        return False

    @staticmethod
    def get_language(file_path: Path) -> str:
        """Get the language name for a file."""
        ext = file_path.suffix.lower()
        return LANGUAGE_MAP.get(ext, ext.lstrip(".").upper())

    def get_context(self, file_path: Path, content: str | None = None) -> str:
        """Extract contextual information (imports) from a file."""
        ext = file_path.suffix.lower()

        if ext == ".py":
            return self._get_python_context(file_path, content)
        elif ext in {".ts", ".tsx", ".js", ".jsx"}:
            return self._get_ts_context(file_path, content)
        elif ext in {".css", ".scss"}:
            return self._get_css_context(file_path)
        return "(No context)"

    def _get_python_context(self, file_path: Path, content: str | None = None) -> str:
        """Extract Python imports."""
        try:
            if content is None:
                with file_path.open(encoding="utf-8") as f:
                    content = f.read()

            tree = ast.parse(content, filename=str(file_path))
        except (SyntaxError, OSError, ValueError):
            return "(Could not parse)"

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)

        if imports:
            return f"Imports: {', '.join(imports[:15])}"
        return "(No imports)"

    def _get_ts_context(self, file_path: Path, content: str | None = None) -> str:
        """Extract TypeScript/JavaScript imports."""
        try:
            if content is None:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            return "(Could not read)"

        pattern = r"(?:import|from)\s+['\"]([^'\"]+)['\"]"
        matches = re.findall(pattern, content)
        external = [m for m in matches if not m.startswith(".")][:10]

        if external:
            return f"Imports: {', '.join(external)}"
        return "(No external imports)"

    def _get_css_context(self, file_path: Path) -> str:
        """Check for associated component files."""
        parent = file_path.parent
        base = file_path.stem.replace(".module", "")

        for ext in [".tsx", ".jsx", ".ts", ".js"]:
            comp = parent / f"{base}{ext}"
            if comp.exists():
                return f"Associated component: {comp.name}"
        return "(Standalone stylesheet)"
