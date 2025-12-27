#!/usr/bin/env python3
"""
Bulk Code Review Script (Dual-Model).

Concurrently sends all project Python files to TWO LLMs for issue detection.
Models: openai-gpt-oss-120b + r1-distill-llama-70b
Context: Single file + depth-1 imports (no changes made, read-only analysis).
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional

import requests
from dotenv import load_dotenv

load_dotenv(".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("bulk_review")


class BulkCodeReviewer:
    """Encapsulates the logic for the bulk code review script."""

    # --- Configuration ---
    PROJECT_ROOT: Path = Path(__file__).parent.resolve()
    BACKEND_SRC: Path = PROJECT_ROOT / "backend" / "src"
    CLI_SRC: Path = PROJECT_ROOT / "cli" / "src"
    SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"

    SCAN_DIRS: list[Path] = [BACKEND_SRC, CLI_SRC, SCRIPTS_DIR]

    EXCLUDE_PATTERNS: list[str] = [
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        ".git",
        "venv",
        ".venv",
        "reference code",
        "tests",  # Skip test files for faster run
    ]

    DEFAULT_MODELS: list[str] = [
        "openai-gpt-oss-120b",
        # "openai-gpt-5",
    ]

    REVIEW_PROMPT: str = """
    <task>
    You are a Senior Code Reviewer. Analyze the following Python file for potential issues.
    DO NOT suggest fixes. Only identify and describe problems.
    </task>

    <categories>
    Report issues in these categories:
    1. LOGIC_ERRORS: Incorrect conditionals, wrong comparisons, off-by-one, infinite loops, unreachable code
    2. NULL_SAFETY: Missing None checks, unguarded optional access, NoneType errors
    3. EXCEPTION_HANDLING: Bare except, swallowed exceptions, missing cleanup, wrong exception types
    4. SECURITY: Hardcoded secrets, injection vulnerabilities, unsafe operations, path traversal
    5. PERFORMANCE: Inefficient loops, N+1 queries, memory leaks, redundant computations
    6. STYLE: Major PEP8 violations, naming issues, dead code, magic numbers
    7. TYPE_ERRORS: Incorrect type hints, missing types on public APIs, type mismatches
    </categories>

    <context>
    File: {file_path}
    Related Imports (Depth 1):
    {imports_context}

    Previous Review Findings (from prior analysis pass):
    {previous_findings}
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

    def __init__(self, models: list[str], max_workers: int = 10):
        self.models = models
        self.max_workers = max_workers
        self.api_key = os.getenv("LLM_API_KEY") or os.getenv("DO_API_KEY")
        self.base_url = (os.getenv("LLM_ENDPOINT") or "https://inference.do-ai.run/v1").rstrip("/")
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        if not self.api_key:
            logger.error("Missing API key. Set LLM_API_KEY or DO_API_KEY.")
            sys.exit(1)

        self.session = requests.Session()
        self.session.headers.update(self.headers)

        self.results_by_file: dict[str, list[dict[str, Any]]] = {}
        self.total_failed = 0
        self.skipped_files: set[str] = set()

        # Thread-safe cache for previous findings
        self._previous_findings_cache: dict[str, list[dict[str, Any]]] | None = None
        self._cache_lock = threading.Lock()

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

    def _is_gitignored(self, file_path: Path, gitignore_patterns: list[str]) -> bool:
        """Check if a file matches any gitignore pattern."""
        rel_path = str(file_path.relative_to(self.PROJECT_ROOT))
        for pattern in gitignore_patterns:
            if pattern in rel_path:
                return True
            if pattern.endswith("/") and pattern.rstrip("/") in rel_path:
                return True
        return False

    def collect_python_files(self) -> list[Path]:
        """Find all Python files in configured directories, respecting .gitignore."""
        gitignore_patterns = self._load_gitignore_patterns()
        files = []
        for scan_dir in self.SCAN_DIRS:
            if not scan_dir.exists():
                continue
            for py_file in scan_dir.rglob("*.py"):
                if any(excl in str(py_file) for excl in self.EXCLUDE_PATTERNS):
                    continue
                if self._is_gitignored(py_file, gitignore_patterns):
                    continue
                files.append(py_file)
        return files

    def _extract_imports(self, file_path: Path) -> list[str]:
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

    def _resolve_import_to_file(self, import_name: str) -> Path | None:
        """Attempt to resolve an import name to a local project file."""
        parts = import_name.split(".")
        for base in [self.BACKEND_SRC, self.CLI_SRC]:
            candidate = base.joinpath(*parts).with_suffix(".py")
            if candidate.exists():
                return candidate
            init_candidate = base.joinpath(*parts, "__init__.py")
            if init_candidate.exists():
                return init_candidate
        return None

    def _get_depth1_context(self, file_path: Path) -> str:
        """Get summary of depth-1 imports."""
        imports = self._extract_imports(file_path)
        context_parts = []
        for imp in imports[:10]:
            resolved = self._resolve_import_to_file(imp)
            if resolved and resolved.exists():
                try:
                    with resolved.open(encoding="utf-8") as f:
                        lines = f.read().split("\n")[:30]
                    context_parts.append(f"--- {imp} ---\n" + "\n".join(lines))
                except OSError as e:
                    logger.warning(f"Could not read import {resolved}: {e}")
        return "\n\n".join(context_parts)[:4000] if context_parts else "(No local imports)"

    def _load_previous_report_data(self):
        """Loads report data into the cache. Must be called within a lock."""
        self._previous_findings_cache = {}
        report_path = self.PROJECT_ROOT / "bulk_review_report.json"
        if not report_path.exists():
            logger.info("No previous report found at %s", report_path)
            return

        try:
            with open(report_path) as f:
                data = json.load(f)
            for issue in data.get("issues", []):
                file_key = issue.get("file", "")
                if file_key not in self._previous_findings_cache:
                    self._previous_findings_cache[file_key] = []
                self._previous_findings_cache[file_key].append(issue)
            logger.info(f"Loaded {len(data.get('issues', []))} previous findings.")
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load previous report: {e}")

    def _get_previous_findings(self, file_path: str) -> str:
        """Get previous findings for a specific file, using a thread-safe, lazy-loading cache."""
        if self._previous_findings_cache is None:  # 1st check (no lock)
            with self._cache_lock:
                if self._previous_findings_cache is None:  # 2nd check (with lock)
                    self._load_previous_report_data()

        findings = self._previous_findings_cache.get(file_path, [])
        if not findings:
            return "(No previous findings for this file)"

        lines = [f"- [{f.get('category', '?')}] {f.get('description', '')[:150]}" for f in findings[:5]]
        return "\n".join(lines)

    def _call_llm(self, model: str, prompt: str) -> dict[str, Any]:
        """Make a single LLM API call."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
        }
        try:
            resp = self.session.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            return json.loads(content)
        except (requests.RequestException, json.JSONDecodeError) as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def review_file_with_model(self, file_path: Path, model: str) -> dict[str, Any]:
        """Review a single file with a specific model."""
        try:
            content = file_path.read_text(encoding="utf-8")

            if len(content) > 50000:
                return {"skipped": True, "reason": "File too large"}
            if not content.strip():
                return {"skipped": True, "reason": "Empty file"}

            rel_path = str(file_path.relative_to(self.PROJECT_ROOT))
            imports_context = self._get_depth1_context(file_path)
            previous_findings = self._get_previous_findings(rel_path)

            prompt = self.REVIEW_PROMPT.format(
                file_path=rel_path,
                imports_context=imports_context[:3000],
                file_content=content[:15000],
                previous_findings=previous_findings,
            )

            result = self._call_llm(model, prompt)
            result["model"] = model
            result["file"] = rel_path
            return result

        except Exception as e:
            return {"error": str(e), "model": model}

    def run_review(self):
        """Execute the bulk code review."""
        logger.info("Collecting Python files...")
        files = self.collect_python_files()
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

                    if result.get("skipped"):
                        if file_key not in self.results_by_file and file_key not in self.skipped_files:
                            self.skipped_files.add(file_key)
                    elif result.get("error"):
                        self.total_failed += 1
                        logger.error(f"❌ {file_path.name} ({model}): {result.get('error')[:80]}")
                    else:
                        if file_key not in self.results_by_file:
                            self.results_by_file[file_key] = []
                        self.results_by_file[file_key].append(result)
                        issue_count = len(result.get("issues", []))

                        log_func = logger.warning if issue_count > 0 else logger.info
                        log_func(f"{'⚠️' if issue_count > 0 else '✅'} {file_path.name} ({model}): {issue_count} issues")

                except Exception as e:
                    self.total_failed += 1
                    logger.error(f"❌ Exception for {file_path.name} ({model}): {e}")

    def generate_report(self):
        """Generate and print the final report."""
        all_issues = []
        for file_key, model_results in self.results_by_file.items():
            seen_descriptions = set()
            for r in model_results:
                for issue in r.get("issues", []):
                    desc = issue.get("description", "")[:100]
                    if desc not in seen_descriptions:
                        seen_descriptions.add(desc)
                        issue["file"] = file_key
                        issue["found_by"] = r.get("model", "unknown")
                        all_issues.append(issue)

        report_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "summary": {
                "total_files_processed": len(self.results_by_file),
                "skipped": len(self.skipped_files),
                "failed": self.total_failed,
                "models": self.models,
                "total_unique_issues": len(all_issues),
            },
            "issues": all_issues,
        }

        report_path = self.PROJECT_ROOT / "bulk_review_report_v2.json"
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"Report saved to {report_path}")

        print("\n" + "=" * 80)
        print("BULK CODE REVIEW REPORT")
        print("=" * 80)
        print(f"Files Reviewed: {len(self.results_by_file)}")
        print(f"Skipped: {len(self.skipped_files)}, Failed: {self.total_failed}")
        print(f"Models: {', '.join(self.models)}")
        print("-" * 80)

        if all_issues:
            print(f"\nTotal Unique Issues Found: {len(all_issues)}\n")
            categories = sorted(list(set(i.get("category") for i in all_issues)))
            for category in categories:
                cat_issues = [i for i in all_issues if i.get("category") == category]
                if cat_issues:
                    print(f"\n### {category} ({len(cat_issues)} issues)")
                    for i in cat_issues[:15]:
                        line = f":L{i['line']}" if i.get("line") else ""
                        model_tag = f"[{i.get('found_by', '?')[:10]}]"
                        print(f"  {model_tag} {i['file']}{line}: {i['description'][:100]}")
        else:
            print("\n✅ No issues found across all reviewed files!")

        print("\n" + "=" * 80)


def main():
    """Main entry point for the script."""
    print("="*80)
    print(" Cortex Bulk Code Review Utility")
    print("="*80)
    print("WARNING: This script sends source code to a third-party API for analysis.")
    print("Ensure you are authorized to do so and trust the service provider.")
    print("="*80)

    parser = argparse.ArgumentParser(description="Bulk Code Review Script")
    parser.add_argument(
        "--models",
        nargs="+",
        default=BulkCodeReviewer.DEFAULT_MODELS,
        help="Space-separated list of models to use for review.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of concurrent API calls.",
    )
    args = parser.parse_args()

    reviewer = BulkCodeReviewer(models=args.models, max_workers=args.workers)
    reviewer.run_review()
    reviewer.generate_report()


if __name__ == "__main__":
    main()
