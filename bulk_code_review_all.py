#!/usr/bin/env python3
"""
Bulk Code Review Script (All Languages).

Concurrently sends all project code files to LLM for issue detection.
Includes: Python, TypeScript, JavaScript, React (TSX/JSX), CSS, YAML, JSON configs.
Excludes: .txt, .log, binary files, generated files, etc.
Features: Rate limit retry with exponential backoff, intelligent context.
"""

import argparse
import ast
import datetime
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv(".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("bulk_review")


class RateLimitRetry:
    """Handles rate limit retries with exponential backoff."""

    def __init__(self, max_retries: int = 5, base_delay: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._lock = threading.Lock()
        self._global_wait_until = 0

    def wait_if_needed(self):
        """Wait if global rate limit is active."""
        with self._lock:
            now = time.time()
            if self._global_wait_until > now:
                wait_time = self._global_wait_until - now
                logger.warning(f"Rate limit active, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)

    def set_global_wait(self, seconds: float):
        """Set a global wait period for all threads."""
        with self._lock:
            self._global_wait_until = max(
                self._global_wait_until, time.time() + seconds
            )

    def get_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = self.base_delay * (2**attempt)
        jitter = delay * 0.1 * (0.5 - time.time() % 1)
        return min(delay + jitter, 120)


class BulkCodeReviewer:
    """Encapsulates the logic for the bulk code review script."""

    PROJECT_ROOT: Path = Path(__file__).parent.resolve()

    # Directories to scan
    SCAN_DIRS: list[Path] = [
        PROJECT_ROOT / "backend" / "src",
        PROJECT_ROOT / "cli" / "src",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "frontend" / "src",
        PROJECT_ROOT / "workers" / "src",
    ]

    # File extensions to include
    CODE_EXTENSIONS: set[str] = {
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".css",
        ".scss",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
        ".sh",
        ".sql",
    }

    # Extensions to explicitly exclude
    EXCLUDE_EXTENSIONS: set[str] = {
        ".txt",
        ".log",
        ".md",
        ".rst",
        ".csv",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".ico",
        ".webp",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".pdf",
        ".doc",
        ".docx",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".pyc",
        ".pyo",
        ".so",
        ".dll",
        ".exe",
        ".lock",
        ".map",
        ".min.js",
        ".min.css",
        ".d.ts",  # TypeScript declaration files
    }

    # Directory patterns to exclude
    EXCLUDE_PATTERNS: list[str] = [
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        ".git",
        "venv",
        ".venv",
        "reference code",
        "dist",
        "build",
        ".next",
        ".nuxt",
        "coverage",
        "htmlcov",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        "egg-info",
        ".eggs",
    ]

    # Files to explicitly exclude
    EXCLUDE_FILES: set[str] = {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "Pipfile.lock",
    }

    DEFAULT_MODELS: list[str] = ["openai-gpt-5"]

    # Language-specific prompts
    REVIEW_PROMPT: str = """
    <task>
    You are a Senior Code Reviewer. Analyze the following {language} file for potential issues.
    DO NOT suggest fixes. Only identify and describe problems.
    </task>

    <categories>
    Report issues in these categories:
    1. LOGIC_ERRORS: Incorrect conditionals, wrong comparisons, off-by-one, infinite loops, unreachable code
    2. NULL_SAFETY: Missing null/undefined checks, unguarded optional access, NoneType/TypeError risks
    3. EXCEPTION_HANDLING: Unhandled exceptions, swallowed errors, missing cleanup, wrong error types
    4. SECURITY: Hardcoded secrets, injection vulnerabilities, unsafe operations, XSS/CSRF risks
    5. PERFORMANCE: Inefficient loops, N+1 queries, memory leaks, redundant computations, unnecessary re-renders
    6. STYLE: Major style violations, naming issues, dead code, magic numbers, accessibility issues
    7. TYPE_ERRORS: Incorrect types, missing types on public APIs, type mismatches, any-abuse
    </categories>

    <context>
    File: {file_path}
    Language: {language}
    Related Context:
    {imports_context}
    </context>

    <code>
    {file_content}
    </code>

    <output_format>
    Respond ONLY with a JSON object:
    {{
      "file": "<filename>",
      "issues": [
        {{"category": "<LOGIC_ERRORS|NULL_SAFETY|EXCEPTION_HANDLING|SECURITY|PERFORMANCE|STYLE|TYPE_ERRORS>", "line": <int or null>, "description": "<issue>"}}
      ],
      "summary": "<1-sentence overall assessment>"
    }}
    If no issues found, return {{"file": "<filename>", "issues": [], "summary": "No issues detected."}}
    </output_format>
    """

    def __init__(
        self, models: list[str], max_workers: int = 8, skip_tests: bool = False
    ):
        self.models = models
        self.max_workers = max_workers
        self.skip_tests = skip_tests
        self.api_key = os.getenv("LLM_API_KEY") or os.getenv("DO_API_KEY")
        self.base_url = (
            os.getenv("LLM_ENDPOINT") or "https://inference.do-ai.run/v1"
        ).rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if not self.api_key:
            logger.error("Missing API key. Set LLM_API_KEY or DO_API_KEY.")
            sys.exit(1)

        self.session = requests.Session()
        self.session.headers.update(self.headers)

        self.results_by_file: dict[str, list[dict[str, Any]]] = {}
        self.total_failed = 0
        self.skipped_files: set[str] = set()
        self.rate_limiter = RateLimitRetry()
        self._lock = threading.Lock()

    def _get_language(self, file_path: Path) -> str:
        """Determine the language/file type for context."""
        ext = file_path.suffix.lower()
        lang_map = {
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
        return lang_map.get(ext, ext.lstrip(".").upper())

    def _load_gitignore_patterns(self) -> list[str]:
        """Load patterns from .gitignore file."""
        gitignore_path = self.PROJECT_ROOT / ".gitignore"
        patterns = []
        if gitignore_path.exists():
            with gitignore_path.open() as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line)
        return patterns

    def _is_excluded(self, file_path: Path, gitignore_patterns: list[str]) -> bool:
        """Check if a file should be excluded."""
        rel_path = str(file_path.relative_to(self.PROJECT_ROOT))

        # Check exclude patterns (directories)
        for pattern in self.EXCLUDE_PATTERNS:
            if pattern in rel_path:
                return True

        # Check excluded files
        if file_path.name in self.EXCLUDE_FILES:
            return True

        # Check excluded extensions
        if file_path.suffix.lower() in self.EXCLUDE_EXTENSIONS:
            return True

        # Check test files if skip_tests is enabled
        if self.skip_tests:
            if "test" in file_path.name.lower() or "tests" in rel_path.lower():
                return True

        # Check gitignore patterns
        for pattern in gitignore_patterns:
            if pattern in rel_path:
                return True
            if pattern.endswith("/") and pattern.rstrip("/") in rel_path:
                return True

        return False

    def collect_files(self) -> list[Path]:
        """Find all code files in configured directories."""
        gitignore_patterns = self._load_gitignore_patterns()
        files = []

        for scan_dir in self.SCAN_DIRS:
            if not scan_dir.exists():
                continue
            for file_path in scan_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in self.CODE_EXTENSIONS:
                    continue
                if self._is_excluded(file_path, gitignore_patterns):
                    continue
                files.append(file_path)

        return files

    def _extract_python_imports(self, file_path: Path) -> list[str]:
        """Parse Python file and extract import statements."""
        try:
            with file_path.open(encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(file_path))
        except SyntaxError:
            return []

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        return imports

    def _extract_ts_imports(self, content: str) -> list[str]:
        """Extract import paths from TypeScript/JavaScript."""
        pattern = r"(?:import|from)\s+['\"]([^'\"]+)['\"]"
        matches = re.findall(pattern, content)
        return [m for m in matches if not m.startswith(".")][:10]

    def _get_context(self, file_path: Path) -> str:
        """Get contextual information based on file type."""
        ext = file_path.suffix.lower()

        if ext == ".py":
            imports = self._extract_python_imports(file_path)
            if imports:
                return f"Imports: {', '.join(imports[:15])}"
            return "(No imports)"

        if ext in {".ts", ".tsx", ".js", ".jsx"}:
            try:
                content = file_path.read_text(encoding="utf-8")
                imports = self._extract_ts_imports(content)
                if imports:
                    return f"Imports: {', '.join(imports[:15])}"
            except OSError:
                pass
            return "(No external imports)"

        if ext in {".css", ".scss"}:
            # Check for component association
            parent = file_path.parent
            base = file_path.stem.replace(".module", "")
            for ext in [".tsx", ".jsx", ".ts", ".js"]:
                comp = parent / f"{base}{ext}"
                if comp.exists():
                    return f"Associated component: {comp.name}"
            return "(Standalone stylesheet)"

        return "(No context)"

    def _call_llm_with_retry(self, model: str, prompt: str) -> dict[str, Any]:
        """Make LLM API call with rate limit retry."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
        }

        for attempt in range(self.rate_limiter.max_retries):
            self.rate_limiter.wait_if_needed()

            try:
                resp = self.session.post(url, json=payload, timeout=120)

                # Handle rate limits
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", 30))
                    self.rate_limiter.set_global_wait(retry_after)
                    delay = self.rate_limiter.get_delay(attempt)
                    logger.warning(
                        f"Rate limited (429). Retry {attempt + 1}/{self.rate_limiter.max_retries} in {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue

                resp.raise_for_status()
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                )
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
                return json.loads(content)

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt < self.rate_limiter.max_retries - 1:
                    time.sleep(self.rate_limiter.get_delay(attempt))
                    continue
            except requests.RequestException as e:
                if attempt < self.rate_limiter.max_retries - 1:
                    time.sleep(self.rate_limiter.get_delay(attempt))
                    continue
                return {"error": f"RequestException: {e}"}
            except json.JSONDecodeError as e:
                return {"error": f"JSONDecodeError: {e}"}

        return {"error": "Max retries exceeded"}

    def review_file_with_model(self, file_path: Path, model: str) -> dict[str, Any]:
        """Review a single file with a specific model."""
        try:
            content = file_path.read_text(encoding="utf-8")

            if len(content) > 50000:
                return {"skipped": True, "reason": "File too large"}
            if not content.strip():
                return {"skipped": True, "reason": "Empty file"}

            rel_path = str(file_path.relative_to(self.PROJECT_ROOT))
            language = self._get_language(file_path)
            imports_context = self._get_context(file_path)

            prompt = self.REVIEW_PROMPT.format(
                file_path=rel_path,
                language=language,
                imports_context=imports_context,
                file_content=content[:15000],
            )

            result = self._call_llm_with_retry(model, prompt)
            result["model"] = model
            result["file"] = rel_path
            result["language"] = language
            return result

        except UnicodeDecodeError:
            return {"skipped": True, "reason": "Binary file"}
        except Exception as e:
            return {"error": str(e), "model": model}

    def run_review(self):
        """Execute the bulk code review."""
        logger.info("Collecting code files...")
        files = self.collect_files()
        logger.info(f"Found {len(files)} files. Models: {self.models}")

        all_tasks = [(file, model) for file in files for model in self.models]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self.review_file_with_model, file, model): (file, model)
                for file, model in all_tasks
            }

            for future in as_completed(future_to_task):
                file_path, model = future_to_task[future]
                file_key = str(file_path.relative_to(self.PROJECT_ROOT))

                try:
                    result = future.result()

                    with self._lock:
                        if result.get("skipped"):
                            if (
                                file_key not in self.results_by_file
                                and file_key not in self.skipped_files
                            ):
                                self.skipped_files.add(file_key)
                        elif result.get("error"):
                            self.total_failed += 1
                            logger.error(
                                f"❌ {file_path.name} ({model}): {result.get('error')[:80]}"
                            )
                        else:
                            if file_key not in self.results_by_file:
                                self.results_by_file[file_key] = []
                            self.results_by_file[file_key].append(result)
                            issue_count = len(result.get("issues", []))

                            log_func = (
                                logger.warning if issue_count > 0 else logger.info
                            )
                            log_func(
                                f"{'⚠️' if issue_count > 0 else '✅'} {file_path.name} ({model}): {issue_count} issues"
                            )

                            for issue in result.get("issues", []):
                                logger.info(
                                    f"    - [{issue.get('category')}] {issue.get('description')}"
                                )

                            # Incremental Save
                            self.generate_report(incremental=True)

                except Exception as e:
                    with self._lock:
                        self.total_failed += 1
                    logger.error(f"❌ Exception for {file_path.name} ({model}): {e}")

    def generate_report(self, incremental: bool = False):
        """Generate and print the final report."""
        all_issues = []
        # Create a copy of results to avoid runtime modification issues
        current_results = dict(self.results_by_file)

        for file_key, model_results in current_results.items():
            seen_descriptions = set()
            for r in model_results:
                for issue in r.get("issues", []):
                    desc = issue.get("description", "")[:100]
                    if desc not in seen_descriptions:
                        seen_descriptions.add(desc)
                        issue["file"] = file_key
                        issue["found_by"] = r.get("model", "unknown")
                        issue["language"] = r.get("language", "unknown")
                        all_issues.append(issue)

        report_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "summary": {
                "total_files_processed": len(current_results),
                "skipped": len(self.skipped_files),
                "failed": self.total_failed,
                "models": self.models,
                "total_unique_issues": len(all_issues),
            },
            "issues": all_issues,
        }

        filename = (
            "bulk_review_incremental.json"
            if incremental
            else "bulk_review_all_report.json"
        )
        report_path = self.PROJECT_ROOT / filename

        try:
            with open(report_path, "w") as f:
                json.dump(report_data, f, indent=2)
            if not incremental:
                logger.info(f"Report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

        if incremental:
            return

        print("\n" + "=" * 80)
        print("BULK CODE REVIEW REPORT (ALL LANGUAGES)")
        print("=" * 80)
        print(f"Files Reviewed: {len(self.results_by_file)}")
        print(f"Skipped: {len(self.skipped_files)}, Failed: {self.total_failed}")
        print(f"Models: {', '.join(self.models)}")
        print("-" * 80)

        if all_issues:
            print(f"\nTotal Unique Issues Found: {len(all_issues)}\n")
            categories = sorted(set(i.get("category") for i in all_issues))
            for category in categories:
                cat_issues = [i for i in all_issues if i.get("category") == category]
                if cat_issues:
                    print(f"\n### {category} ({len(cat_issues)} issues)")
                    for i in cat_issues[:20]:
                        line = f":L{i['line']}" if i.get("line") else ""
                        lang = f"[{i.get('language', '?')[:6]}]"
                        print(f"  {lang} {i['file']}{line}: {i['description'][:90]}")
        else:
            print("\n✅ No issues found across all reviewed files!")

        print("\n" + "=" * 80)


def main():
    """Main entry point for the script."""
    print("=" * 80)
    print(" Cortex Bulk Code Review Utility (All Languages)")
    print("=" * 80)
    print("Scanning: Python, TypeScript, JavaScript, CSS, YAML, JSON, Shell, SQL")
    print("Excluding: .txt, .log, .md, lock files, binaries, node_modules, etc.")
    print("Features: Rate limit retry with exponential backoff")
    print("=" * 80)

    parser = argparse.ArgumentParser(
        description="Bulk Code Review Script (All Languages)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=BulkCodeReviewer.DEFAULT_MODELS,
        help="Space-separated list of models to use for review.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent API calls.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip test files for faster review.",
    )
    parser.add_argument(
        "--file",
        help="Specific file to review (relative to project root).",
    )
    args = parser.parse_args()

    reviewer = BulkCodeReviewer(
        models=args.models,
        max_workers=args.workers,
        skip_tests=args.skip_tests,
    )

    if args.file:
        target = Path(args.file).resolve()
        if not target.exists():
            print(f"Error: File {target} not found")
            return

        # Override collecting all files
        def specific_file():
            return [target]

        reviewer.collect_files = specific_file

    reviewer.run_review()
    reviewer.generate_report()


if __name__ == "__main__":
    main()
