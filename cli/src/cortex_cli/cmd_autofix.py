#!/usr/bin/env python3
"""
Automated Code Issue Fixer.

Fixes low-hanging fruit issues from bulk_review_report_v2.json:
- Magic numbers → named constants
- Duplicate imports removal
- Missing type hints on simple functions
- __import__("time") → import time
- Unnecessary hasattr checks
- Long lines (via ruff)
"""

import re
import subprocess
import sys
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = ROOT / "backend" / "src"

# ============================================================================
# MAGIC NUMBER FIXES
# ============================================================================

MAGIC_NUMBER_FIXES = {
    # File: (line_pattern, old_value, new_constant_name, new_constant_value)
    "backend/src/cortex/routes_auth.py": [
        (r"86400", "86400", "SECONDS_PER_DAY", "86400"),
    ],
    "backend/src/cortex/intelligence/summarizer.py": [
        (r"\b4\b", "4", "CHARS_PER_TOKEN_ESTIMATE", "4"),
    ],
    "backend/src/cortex/intelligence/graph.py": [
        (r"8000", "8000", "DEFAULT_CHUNK_SIZE", "8000"),
        (r"500", "500", "DEFAULT_OVERLAP", "500"),
    ],
    "backend/src/cortex/db/session.py": [
        (r"1\.0", "1.0", "SLOW_QUERY_THRESHOLD_SECONDS", "1.0"),
    ],
    "backend/src/cortex/retrieval/cache.py": [
        (r"\[:50\]", "[:50]", "LOG_QUERY_TRUNCATE_LEN", "50"),
    ],
    "backend/src/cortex/utils/atomic_io.py": [
        (r"0o600", "0o600", "FILE_PERMISSION_OWNER_RW", "0o600"),
    ],
}


def fix_magic_numbers():
    """Replace magic numbers with named constants."""
    print("\n🔢 Fixing magic numbers...")
    fixed_count = 0

    for rel_path, fixes in MAGIC_NUMBER_FIXES.items():
        file_path = ROOT / rel_path
        if not file_path.exists():
            print(f"  ⚠️ Skipping {rel_path} (not found)")
            continue

        content = file_path.read_text()
        original = content
        constants_to_add = []

        for _pattern, old_val, const_name, const_val in fixes:
            # Check if constant already exists
            if const_name in content:
                continue

            # Add to constants list
            constants_to_add.append(f"{const_name} = {const_val}")

            # Replace usage (be careful with context)
            # Only replace if it's a standalone value, not part of larger number
            if old_val == "[:50]":
                content = content.replace("query[:50]", f"query[:{const_name}]")
            elif old_val == "4" and "summarizer" in rel_path:
                # Special case: chars per token estimate
                content = re.sub(
                    r"len\(text\)\s*//\s*4",
                    f"len(text) // {const_name}",
                    content,
                )

        # Insert constants after imports
        if constants_to_add:
            # Find the end of imports
            lines = content.split("\n")
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("import ") or line.startswith("from "):
                    insert_idx = i + 1
                elif line.strip() and not line.startswith("#") and insert_idx > 0:
                    # Found first non-import, non-comment line
                    break

            # Insert constants
            const_block = "\n# Constants\n" + "\n".join(constants_to_add) + "\n"
            lines.insert(insert_idx, const_block)
            content = "\n".join(lines)

        if content != original:
            file_path.write_text(content)
            print(f"  ✅ {rel_path}")
            fixed_count += len(constants_to_add)

    return fixed_count


# ============================================================================
# DUPLICATE IMPORT FIXES
# ============================================================================

DUPLICATE_IMPORT_FILES = [
    "backend/src/cortex/rag_api/routes_search.py",
]


def fix_duplicate_imports():
    """Remove duplicate imports."""
    print("\n📦 Fixing duplicate imports...")
    fixed_count = 0

    for rel_path in DUPLICATE_IMPORT_FILES:
        file_path = ROOT / rel_path
        if not file_path.exists():
            continue

        content = file_path.read_text()
        lines = content.split("\n")

        # Track seen imports
        seen_imports = set()
        new_lines = []
        removed = 0

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                if stripped in seen_imports:
                    removed += 1
                    continue
                seen_imports.add(stripped)
            new_lines.append(line)

        if removed > 0:
            file_path.write_text("\n".join(new_lines))
            print(f"  ✅ {rel_path} (removed {removed} duplicate imports)")
            fixed_count += removed

    return fixed_count


# ============================================================================
# __import__ REPLACEMENT
# ============================================================================

DUNDER_IMPORT_FILES = [
    "backend/src/cortex/db/session.py",
]


def fix_dunder_imports():
    """Replace __import__('module') with top-level import."""
    print("\n🔄 Fixing __import__ usage...")
    fixed_count = 0

    for rel_path in DUNDER_IMPORT_FILES:
        file_path = ROOT / rel_path
        if not file_path.exists():
            continue

        content = file_path.read_text()
        original = content

        # Find __import__("module") pattern
        pattern = r'__import__\(["\'](\w+)["\']\)'
        match = re.search(pattern, content)
        if match:
            module_name = match.group(1)
            # Replace with module_name
            content = re.sub(pattern, module_name, content)

            # Ensure 'import module' is at top
            if f"import {module_name}" not in content:
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if line.startswith("import ") or line.startswith("from "):
                        lines.insert(i, f"import {module_name}")
                        break
                content = "\n".join(lines)

        if content != original:
            file_path.write_text(content)
            print(f"  ✅ {rel_path}")
            fixed_count += 1

    return fixed_count


# ============================================================================
# UNNECESSARY HASATTR REMOVAL
# ============================================================================

HASATTR_FILES = [
    "backend/src/cortex/utils/atomic_io.py",
]


def fix_unnecessary_hasattr():
    """Remove unnecessary hasattr checks for guaranteed attributes."""
    print("\n🧹 Removing unnecessary hasattr checks...")
    fixed_count = 0

    for rel_path in HASATTR_FILES:
        file_path = ROOT / rel_path
        if not file_path.exists():
            continue

        original_content = file_path.read_text()
        content = original_content

        # First, replace the specific hasattr checks with "True"
        patterns = [
            r'hasattr\(os, ["\']open["\']\)',
            r'hasattr\(os, ["\']O_RDONLY["\']\)',
        ]
        for pattern in patterns:
            content = re.sub(pattern, "True", content)

        # Simplify compound boolean conditions
        content = re.sub(r"if True and True:", "if True:", content)

        # Now, use AST to safely remove 'if True:' blocks
        try:
            tree = ast.parse(content)
            transformer = IfTrueTransformer()
            new_tree = transformer.visit(tree)
            if transformer.changed:
                # Use ast.unparse if available (Python 3.9+)
                if hasattr(ast, "unparse"):
                    content = ast.unparse(new_tree)
                else:
                    # Fallback for older versions (less perfect formatting)
                    import astor
                    content = astor.to_source(new_tree)

                file_path.write_text(content)
                print(f"  ✅ {rel_path}")
                fixed_count += 1
        except (SyntaxError, ImportError) as e:
            print(f"  ⚠️ Could not process {rel_path} with AST: {e}")
            # Revert content if AST processing fails
            content = original_content

    return fixed_count


class IfTrueTransformer(ast.NodeTransformer):
    """
    An AST transformer that replaces 'if True:' blocks with their body.
    """
    def __init__(self):
        self.changed = False

    def visit_If(self, node):
        # Check if the test is a constant True value
        if isinstance(node.test, ast.NameConstant) and node.test.value is True:
            self.changed = True
            # The body of the 'if' statement is a list of nodes.
            # We need to visit each of them to continue the transformation recursively.
            new_body = [self.visit(n) for n in node.body]
            # De-indent the body by one level. Since we are returning a list of nodes,
            # the parent node (e.g., a function body) will handle the placement.
            # No explicit de-indentation is needed at the AST level.
            return new_body
        # If the condition is not 'if True:', visit the node's children as usual.
        return self.generic_visit(node)


# ============================================================================
# UNUSED IMPORT REMOVAL (via autoflake)
# ============================================================================


def fix_unused_imports():
    """Remove unused imports using autoflake."""
    print("\n🗑️ Removing unused imports...")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "autoflake",
                "--in-place",
                "--remove-all-unused-imports",
                "--remove-unused-variables",
                "--recursive",
                str(BACKEND_SRC),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            print("  ✅ autoflake completed successfully")
            return 1
        else:
            print(f"  ⚠️ autoflake warning: {result.stderr}")
            return 0
    except FileNotFoundError:
        print("  ⚠️ autoflake not installed, skipping")
        return 0


# ============================================================================
# IMPORT SORTING (via isort)
# ============================================================================


def fix_import_sorting():
    """Sort imports using isort."""
    print("\n📋 Sorting imports...")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "isort",
                "--profile=black",
                str(BACKEND_SRC),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            print("  ✅ isort completed successfully")
            return 1
        else:
            print(f"  ⚠️ isort warning: {result.stderr}")
            return 0
    except FileNotFoundError:
        print("  ⚠️ isort not installed, skipping")
        return 0


# ============================================================================
# CODE FORMATTING (via ruff)
# ============================================================================


def fix_formatting():
    """Fix formatting issues using ruff."""
    print("\n🎨 Formatting code...")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "format",
                str(BACKEND_SRC),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            print("  ✅ ruff format completed successfully")
            return 1
        else:
            print(f"  ⚠️ ruff warning: {result.stderr}")
            return 0
    except FileNotFoundError:
        print("  ⚠️ ruff not installed, skipping")
        return 0


# ============================================================================
# LINTING FIX (via ruff)
# ============================================================================


def fix_lint_issues():
    """Fix auto-fixable lint issues using ruff."""
    print("\n🔧 Fixing lint issues...")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--fix",
                "--unsafe-fixes",
                str(BACKEND_SRC),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        # ruff returns non-zero if there are unfixable issues
        print("  ✅ ruff check --fix completed")
        if result.stdout:
            # Count fixed issues
            fixed = result.stdout.count("Fixed")
            if fixed:
                print(f"  📊 Fixed {fixed} issues")
        return 1
    except FileNotFoundError:
        print("  ⚠️ ruff not installed, skipping")
        return 0


# ============================================================================
# SPECIFIC FILE FIXES
# ============================================================================


def fix_specific_issues():
    """Fix specific issues identified in the report."""
    print("\n🎯 Fixing specific identified issues...")
    fixed_count = 0

    # Fix 1: backend/src/cortex/safety/__init__.py - unused import
    safety_init = ROOT / "backend/src/cortex/safety/__init__.py"
    if safety_init.exists():
        content = safety_init.read_text()
        if "from .action_checker import check_action" in content and content.count("check_action") == 1:
            content = content.replace(
                "from .action_checker import check_action\n", ""
            )
            safety_init.write_text(content)
            print("  ✅ Removed unused check_action import from safety/__init__.py")
            fixed_count += 1

    # Fix 2: Add type hints to strip_injection_patterns
    if safety_init.exists():
        content = safety_init.read_text()
        if "def strip_injection_patterns(text):" in content:
            content = content.replace(
                "def strip_injection_patterns(text):",
                "def strip_injection_patterns(text: str) -> str:",
            )
            safety_init.write_text(content)
            print("  ✅ Added type hints to strip_injection_patterns")
            fixed_count += 1

    # Fix 3: _is_cache_valid return type
    text_extraction = ROOT / "backend/src/cortex/text_extraction.py"
    if text_extraction.exists():
        content = text_extraction.read_text()
        if (
            "def _is_cache_valid(" in content
            and "-> bool" not in content.split("def _is_cache_valid(")[1].split(":")[0]
        ):
            content = re.sub(
                r"def _is_cache_valid\(([^)]+)\):",
                r"def _is_cache_valid(\1) -> bool:",
                content,
            )
            text_extraction.write_text(content)
            print("  ✅ Added return type to _is_cache_valid")
            fixed_count += 1

    # Fix 4: _normalize_relation return type in graph.py
    graph_py = ROOT / "backend/src/cortex/intelligence/graph.py"
    if graph_py.exists():
        content = graph_py.read_text()
        if "def _normalize_relation(" in content:
            # Check if return type is missing
            func_match = re.search(r"def _normalize_relation\([^)]+\)\s*:", content)
            if func_match and "->" not in func_match.group():
                content = re.sub(
                    r"def _normalize_relation\(([^)]+)\):",
                    r"def _normalize_relation(\1) -> str:",
                    content,
                )
                graph_py.write_text(content)
                print("  ✅ Added return type to _normalize_relation")
                fixed_count += 1

    # Fix 5: get_query_cache return type in cache.py
    cache_py = ROOT / "backend/src/cortex/retrieval/cache.py"
    if cache_py.exists():
        content = cache_py.read_text()
        if "def get_query_cache(" in content:
            func_match = re.search(r"def get_query_cache\([^)]*\)\s*:", content)
            if func_match and "->" not in func_match.group():
                content = re.sub(
                    r"def get_query_cache\(([^)]*)\):",
                    r'def get_query_cache(\1) -> "QueryEmbeddingCache":',
                    content,
                )
                cache_py.write_text(content)
                print("  ✅ Added return type to get_query_cache")
                fixed_count += 1

    return fixed_count


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Run all fixes."""
    print("=" * 60)
    print("🔧 AUTOMATED CODE ISSUE FIXER")
    print("=" * 60)

    total_fixed = 0

    # Run specific targeted fixes first
    total_fixed += fix_specific_issues()
    total_fixed += fix_magic_numbers()
    total_fixed += fix_duplicate_imports()
    total_fixed += fix_dunder_imports()
    total_fixed += fix_unnecessary_hasattr()

    # Run tool-based fixes
    total_fixed += fix_unused_imports()
    total_fixed += fix_import_sorting()
    total_fixed += fix_lint_issues()
    total_fixed += fix_formatting()

    print("\n" + "=" * 60)
    print(f"✅ COMPLETE: Applied {total_fixed} fix categories")
    print("=" * 60)
    print("\n📋 Next steps:")
    print("   1. Run: pre-commit run --all-files")
    print("   2. Run: pytest")
    print("   3. Review changes: git diff")


if __name__ == "__main__":
    main()
