
import ast
import sys
from pathlib import Path
from typing import List

import pytest

# Directories to scan for Python files
DIRECTORIES_TO_SCAN = [
    Path("backend/src"),
    Path("cli/src"),
]

# Root-level scripts to check for bare excepts
ROOT_SCRIPTS = [
    "wait_for_pod.py",
    "wait_for_node.py",
    "fetch_schema_variations.py",
    "fetch_real_s3.py",
    "fetch_real_s3_simple.py",
    "inspect_pgvector.py",
]


def get_python_files(paths: List[Path]) -> List[Path]:
    """Find all Python files in the given paths."""
    python_files = []
    for path in paths:
        if path.is_dir():
            python_files.extend(sorted(path.rglob("*.py")))
        elif path.is_file() and path.exists():
            python_files.append(path)
    return python_files


def find_bare_excepts(file_path: Path):
    """
    Check a single Python file for bare except clauses using the AST module.
    Yields tuples of (line_number, line_content) for each bare except.
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content, filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    # Get the line content for better error reporting
                    line_content = content.splitlines()[node.lineno - 1].strip()
                    yield (node.lineno, line_content)

    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Could not check {file_path}: {e}", file=sys.stderr)


@pytest.mark.parametrize(
    "paths_to_check",
    [
        pytest.param(DIRECTORIES_TO_SCAN, id="backend_and_cli_src"),
        pytest.param([Path(p) for p in ROOT_SCRIPTS], id="root_scripts"),
    ],
)
def test_no_bare_except_in_codebase(paths_to_check: List[Path]):
    """
    Test that no Python files in the specified paths have bare except clauses.
    This test is parameterized to run on different sets of files.
    """
    all_python_files = get_python_files(paths_to_check)
    bare_except_violations = []

    for py_file in all_python_files:
        bare_excepts = list(find_bare_excepts(py_file))
        if bare_excepts:
            for lineno, line in bare_excepts:
                bare_except_violations.append(f"{py_file}:{lineno}: {line}")

    if bare_except_violations:
        error_msg = "Found bare except clauses in the following files:\n" + "\n".join(
            bare_except_violations
        )
        error_msg += "\n\nPlease use 'except Exception:' instead of a bare except."
        raise AssertionError(error_msg)

