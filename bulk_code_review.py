#!/usr/bin/env python3
"""
Bulk Code Review Script (Dual-Model).

Concurrently sends all project Python files to TWO LLMs for issue detection.
Models: openai-gpt-oss-120b + r1-distill-llama-70b
Context: Single file + depth-1 imports (no changes made, read-only analysis).
"""

import ast
import datetime
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("bulk_review")

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.resolve()
BACKEND_SRC = PROJECT_ROOT / "backend" / "src"
CLI_SRC = PROJECT_ROOT / "cli" / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Directories to scan for Python files
SCAN_DIRS = [BACKEND_SRC, CLI_SRC, SCRIPTS_DIR]

# Exclude patterns
EXCLUDE_PATTERNS = [
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".git",
    "venv",
    ".venv",
    "reference code",
    "tests",  # Skip test files for faster run
]

# Concurrency control (50 concurrent API calls total)
MAX_WORKERS = 10

# Models to use (OSS only for this pass)
MODELS = [
    "openai-gpt-oss-120b",
    # "openai-gpt-5",  # Optional, disable by default
]

# API Configuration
API_KEY = os.getenv("LLM_API_KEY") or os.getenv("DO_API_KEY")
BASE_URL = (os.getenv("LLM_ENDPOINT") or "https://inference.do-ai.run/v1").rstrip("/")

if not API_KEY:
    print("Missing API key. Set LLM_API_KEY or DO_API_KEY.")
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# LLM Review Prompt
REVIEW_PROMPT = """
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


def load_gitignore_patterns() -> list[str]:
    """Load patterns from .gitignore file."""
    gitignore_path = PROJECT_ROOT / ".gitignore"
    patterns = []
    if gitignore_path.exists():
        with gitignore_path.open() as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
    return patterns


def is_gitignored(file_path: Path, gitignore_patterns: list[str]) -> bool:
    """Check if a file matches any gitignore pattern."""
    rel_path = str(file_path.relative_to(PROJECT_ROOT))
    for pattern in gitignore_patterns:
        # Simple matching (not full glob, but covers most cases)
        if pattern in rel_path:
            return True
        if pattern.endswith("/") and pattern.rstrip("/") in rel_path:
            return True
    return False


def collect_python_files() -> list[Path]:
    """Find all Python files in configured directories, respecting .gitignore."""
    gitignore_patterns = load_gitignore_patterns()
    files = []
    for scan_dir in SCAN_DIRS:
        if not scan_dir.exists():
            continue
        for py_file in scan_dir.rglob("*.py"):
            # Check hardcoded exclusions
            if any(excl in str(py_file) for excl in EXCLUDE_PATTERNS):
                continue
            # Check gitignore patterns
            if is_gitignored(py_file, gitignore_patterns):
                continue
            files.append(py_file)
    return files


def extract_imports(file_path: Path) -> list[str]:
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


def resolve_import_to_file(import_name: str) -> Path | None:
    """Attempt to resolve an import name to a local project file."""
    parts = import_name.split(".")
    for base in [BACKEND_SRC, CLI_SRC]:
        candidate = base / "/".join(parts) / "__init__.py"
        if candidate.exists():
            return candidate
        candidate = (
            base / "/".join(parts[:-1]) / f"{parts[-1]}.py"
            if len(parts) > 1
            else base / f"{parts[0]}.py"
        )
        if candidate.exists():
            return candidate
    return None


def get_depth1_context(file_path: Path) -> str:
    """Get summary of depth-1 imports."""
    imports = extract_imports(file_path)
    context_parts = []
    for imp in imports[:10]:
        resolved = resolve_import_to_file(imp)
        if resolved and resolved.exists():
            try:
                with resolved.open(encoding="utf-8") as f:
                    lines = f.read().split("\n")[:30]
                context_parts.append(f"--- {imp} ---\n" + "\n".join(lines))
            except Exception:
                pass
    return (
        "\\n\\n".join(context_parts)[:4000] if context_parts else "(No local imports)"
    )


# Load previous report for context
PREVIOUS_REPORT_PATH = PROJECT_ROOT / "bulk_review_report.json"
_previous_findings_cache: dict[str, list[dict]] = {}


def load_previous_report():
    """Load the previous bulk review report into cache."""
    global _previous_findings_cache
    if _previous_findings_cache:
        return  # Already loaded

    if not PREVIOUS_REPORT_PATH.exists():
        logger.info("No previous report found at %s", PREVIOUS_REPORT_PATH)
        return

    try:
        with open(PREVIOUS_REPORT_PATH) as f:
            data = json.load(f)
        for issue in data.get("issues", []):
            file_key = issue.get("file", "")
            if file_key not in _previous_findings_cache:
                _previous_findings_cache[file_key] = []
            _previous_findings_cache[file_key].append(issue)
        logger.info(
            f"Loaded {len(data.get('issues', []))} previous findings for context"
        )
    except Exception as e:
        logger.warning(f"Failed to load previous report: {e}")


def get_previous_findings(file_path: str) -> str:
    """Get previous findings for a specific file as context string."""
    if not _previous_findings_cache:
        load_previous_report()

    findings = _previous_findings_cache.get(file_path, [])
    if not findings:
        return "(No previous findings for this file)"

    lines = []
    for f in findings[:5]:  # Limit to 5 previous findings
        cat = f.get("category", "?")
        desc = f.get("description", "")[:150]
        lines.append(f"- [{cat}] {desc}")
    return "\\n".join(lines)


def call_llm(model: str, prompt: str) -> dict:
    """Make a single LLM API call."""
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
    }
    try:
        resp = requests.post(url, json=payload, headers=HEADERS, timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            content = (
                data.get("choices", [{}])[0].get("message", {}).get("content") or "{}"
            )
            # Strip <think> tags if present
            import re

            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            import json

            return json.loads(content)
        else:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}


def review_file_with_model(file_path: Path, model: str) -> tuple[str, dict]:
    """Review a single file with a specific model."""
    try:
        with file_path.open(encoding="utf-8") as f:
            content = f.read()

        if len(content) > 50000:
            return model, {
                "file": str(file_path),
                "skipped": True,
                "reason": "File too large",
            }
        if not content.strip():
            return model, {
                "file": str(file_path),
                "skipped": True,
                "reason": "Empty file",
            }

        imports_context = get_depth1_context(file_path)

        # Load previous findings for this file
        previous_findings = get_previous_findings(
            str(file_path.relative_to(PROJECT_ROOT))
        )

        prompt = REVIEW_PROMPT.format(
            file_path=str(file_path.relative_to(PROJECT_ROOT)),
            imports_context=imports_context[:3000],
            file_content=content[:15000],
            previous_findings=previous_findings,
        )

        result = call_llm(model, prompt)
        result["model"] = model
        result["file"] = str(file_path.relative_to(PROJECT_ROOT))
        return model, result

    except Exception as e:
        return model, {"file": str(file_path), "error": str(e), "model": model}


def main():
    logger.info("Collecting Python files...")
    files = collect_python_files()
    logger.info(f"Found {len(files)} files. Models: {MODELS}")

    results_by_file: dict[str, list[dict]] = {}
    total_failed = 0
    total_skipped = 0

    # Run each model SEQUENTIALLY
    for model in MODELS:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"PASS: {model}")
        logger.info(f"{'=' * 60}")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {
                executor.submit(review_file_with_model, f, model): f for f in files
            }

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    _, result = future.result()
                    file_key = str(file_path.relative_to(PROJECT_ROOT))

                    if result.get("skipped"):
                        total_skipped += 1
                    elif result.get("error"):
                        total_failed += 1
                        logger.error(f"❌ {file_path.name}: {result.get('error')[:50]}")
                    else:
                        if file_key not in results_by_file:
                            results_by_file[file_key] = []
                        results_by_file[file_key].append(result)
                        issue_count = len(result.get("issues", []))
                        if issue_count > 0:
                            logger.warning(f"⚠️  {file_path.name}: {issue_count} issues")
                        else:
                            logger.info(f"✅ {file_path.name}: Clean")

                except Exception as e:
                    total_failed += 1
                    logger.error(f"❌ Exception: {e}")

        logger.info(f"Completed pass with {model}")

    # --- Merge and Dedupe Issues ---
    all_issues = []
    for file_key, model_results in results_by_file.items():
        seen_descriptions = set()
        for r in model_results:
            for issue in r.get("issues", []):
                desc = issue.get("description", "")[:100]
                if desc not in seen_descriptions:
                    seen_descriptions.add(desc)
                    issue["file"] = file_key
                    issue["found_by"] = r.get("model", "unknown")
                    all_issues.append(issue)

    # --- Generate Report ---
    report_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "summary": {
            "total_files": len(results_by_file),
            "skipped": total_skipped // 2,
            "failed": total_failed,
            "models": MODELS,
            "total_unique_issues": len(all_issues),
        },
        "issues": all_issues,
    }

    report_path = PROJECT_ROOT / "bulk_review_report_v2.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"Report saved to {report_path}")

    print("\n" + "=" * 80)
    print("DUAL-MODEL BULK CODE REVIEW REPORT (Sequential)")
    print("=" * 80)
    print(f"Files Reviewed: {len(results_by_file)}")
    print(f"Skipped: {total_skipped // 2}, Failed: {total_failed}")
    print(f"Models: {', '.join(MODELS)}")
    print("-" * 80)

    if all_issues:
        print(f"\nTotal Unique Issues Found: {len(all_issues)}\n")

        for category in [
            "LOGIC_ERRORS",
            "NULL_SAFETY",
            "EXCEPTION_HANDLING",
            "SECURITY",
            "PERFORMANCE",
            "STYLE",
            "TYPE_ERRORS",
        ]:
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


if __name__ == "__main__":
    main()
