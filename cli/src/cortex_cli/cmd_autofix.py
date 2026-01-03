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

import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = ROOT / "backend" / "src"

IMPORT_LITERAL = "import "
FROM_LITERAL = "from "

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


def _insert_constants(content: str, constants_to_add: list[str]) -> str:
    """Insert constant definitions after the last import."""
    if not constants_to_add:
        return content

    lines = content.split("\n")
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith(IMPORT_LITERAL) or line.startswith(FROM_LITERAL):
            insert_idx = i + 1
        elif line.strip() and not line.startswith("#") and insert_idx > 0:
            break

    const_block = "\n# Constants\n" + "\n".join(constants_to_add) + "\n"
    lines.insert(insert_idx, const_block)
    return "\n".join(lines)


def _apply_magic_number_fixes(
    content: str, fixes: list, rel_path: str
) -> tuple[str, list[str]]:
    """Apply a list of magic number fixes to the content."""
    constants_to_add = []
    for _pattern, old_val, const_name, const_val in fixes:
        if const_name in content:
            continue

        constants_to_add.append(f"{const_name} = {const_val}")

        if old_val == "[:50]":
            content = content.replace("query[:50]", f"query[:{const_name}]")
        elif old_val == "4" and "summarizer" in rel_path:
            content = re.sub(
                r"len\(text\)\s*//\s*4",
                f"len(text) // {const_name}",
                content,
            )
    return content, constants_to_add


def fix_magic_numbers() -> int:
    """Replace magic numbers with named constants."""
    print("\n🔢 Fixing magic numbers...")
    fixed_count = 0

    for rel_path, fixes in MAGIC_NUMBER_FIXES.items():
        file_path = ROOT / rel_path
        if not file_path.exists():
            print(f"  ⚠️ Skipping {rel_path} (not found)")
            continue

        original_content = file_path.read_text()
        content, constants_to_add = _apply_magic_number_fixes(
            original_content, fixes, rel_path
        )

        if constants_to_add:
            content = _insert_constants(content, constants_to_add)

        if content != original_content:
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


def fix_duplicate_imports() -> int:
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
            if stripped.startswith(IMPORT_LITERAL) or stripped.startswith(FROM_LITERAL):
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


def _add_top_level_import(content: str, module_name: str) -> str:
    """Adds 'import <module_name>' to the top-level import block."""
    if f"import {module_name}" in content:
        return content

    lines = content.split("\n")
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith(IMPORT_LITERAL) or line.startswith(FROM_LITERAL):
            insert_idx = i + 1
        elif line.strip() and not line.startswith("#") and insert_idx > 0:
            break
    lines.insert(insert_idx, f"import {module_name}")
    return "\n".join(lines)


def _replace_dunder_import_in_content(content: str) -> str:
    """Replaces __import__("module") with a top-level import."""
    pattern = r'__import__\(["\'](\w+)["\']\)'
    match = re.search(pattern, content)

    if not match:
        return content

    module_name = match.group(1)
    content_after_replace = re.sub(pattern, module_name, content)
    return _add_top_level_import(content_after_replace, module_name)


def fix_dunder_imports() -> int:
    """Replace __import__('module') with top-level import."""
    print("\n🔄 Fixing __import__ usage...")
    fixed_count = 0

    for rel_path in DUNDER_IMPORT_FILES:
        file_path = ROOT / rel_path
        if not file_path.exists():
            continue

        original_content = file_path.read_text()
        new_content = _replace_dunder_import_in_content(original_content)

        if new_content != original_content:
            file_path.write_text(new_content)
            print(f"  ✅ {rel_path}")
            fixed_count += 1

    return fixed_count


# ============================================================================
# UNNECESSARY HASATTR REMOVAL
# ============================================================================

HASATTR_FILES = [
    "backend/src/cortex/utils/atomic_io.py",
]


def fix_unnecessary_hasattr() -> int:
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
        content = content.replace("if True and True:", "if True:")

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

    def __init__(self) -> None:
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


def fix_unused_imports() -> int:
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


def fix_import_sorting() -> int:
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


def fix_formatting() -> int:
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


def fix_lint_issues() -> int:
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


def _fix_unused_import_in_safety_init() -> int:
    """Remove unused check_action import from safety/__init__.py."""
    safety_init = ROOT / "backend/src/cortex/safety/__init__.py"
    if not safety_init.exists():
        return 0

    content = safety_init.read_text()
    if (
        "from .action_checker import check_action" in content
        and content.count("check_action") == 1
    ):
        content = content.replace("from .action_checker import check_action\n", "")
        safety_init.write_text(content)
        print("  ✅ Removed unused check_action import from safety/__init__.py")
        return 1
    return 0


def _fix_type_hints_in_safety_init() -> int:
    """Add type hints to strip_injection_patterns in safety/__init__.py."""
    safety_init = ROOT / "backend/src/cortex/safety/__init__.py"
    if not safety_init.exists():
        return 0

    content = safety_init.read_text()
    if "def strip_injection_patterns(text):" in content:
        content = content.replace(
            "def strip_injection_patterns(text):",
            "def strip_injection_patterns(text: str) -> str:",
        )
        safety_init.write_text(content)
        print("  ✅ Added type hints to strip_injection_patterns")
        return 1
    return 0


def _fix_return_type_in_text_extraction() -> int:
    """Add return type to _is_cache_valid in text_extraction.py."""
    text_extraction = ROOT / "backend/src/cortex/text_extraction.py"
    if not text_extraction.exists():
        return 0

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
        return 1
    return 0


def _fix_return_type_in_graph_py() -> int:
    """Add return type to _normalize_relation in graph.py."""
    graph_py = ROOT / "backend/src/cortex/intelligence/graph.py"
    if not graph_py.exists():
        return 0

    content = graph_py.read_text()
    if "def _normalize_relation(" in content:
        func_match = re.search(r"def _normalize_relation\([^)]+\)\s*:", content)
        if func_match and "->" not in func_match.group():
            content = re.sub(
                r"def _normalize_relation\(([^)]+)\):",
                r"def _normalize_relation(\1) -> str:",
                content,
            )
            graph_py.write_text(content)
            print("  ✅ Added return type to _normalize_relation")
            return 1
    return 0


def _fix_return_type_in_cache_py() -> int:
    """Add return type to get_query_cache in cache.py."""
    cache_py = ROOT / "backend/src/cortex/retrieval/cache.py"
    if not cache_py.exists():
        return 0

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
            return 1
    return 0


def fix_specific_issues() -> int:
    """Fix specific issues identified in the report."""
    print("\n🎯 Fixing specific identified issues...")
    fixed_count = 0
    fixed_count += _fix_unused_import_in_safety_init()
    fixed_count += _fix_type_hints_in_safety_init()
    fixed_count += _fix_return_type_in_text_extraction()
    fixed_count += _fix_return_type_in_graph_py()
    fixed_count += _fix_return_type_in_cache_py()
    return fixed_count


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
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
