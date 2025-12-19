import sys
from pathlib import Path


def test_no_bare_except_in_backend():
    """Test that no Python files in backend/src have bare except clauses."""
    backend_src = Path("backend/src")

    # Find all Python files
    python_files = list(backend_src.rglob("*.py"))

    bare_except_violations = []

    for py_file in python_files:
        try:
            with py_file.open("r", encoding="utf-8") as f:
                lines = f.readlines()
                for i, line in enumerate(lines, 1):
                    stripped = line.strip()
                    # Match bare except clauses (only "except:" or "except:" + whitespace)
                    if stripped == "except:" or (
                        stripped.startswith("except:") and len(stripped) <= 7
                    ):
                        bare_except_violations.append(f"{py_file}:{i}: {line.strip()}")
        except Exception:
            # Skip files we can't read
            continue

    if bare_except_violations:
        error_msg = "Found bare except clauses in the following files:\n" + "\n".join(
            bare_except_violations
        )
        error_msg += "\n\nPlease use specific exception types instead of bare except."
        raise AssertionError(error_msg)


def test_no_bare_except_in_root_scripts():
    """Test that no root Python utility files have bare except clauses."""
    root_files = [
        "wait_for_pod.py",
        "wait_for_node.py",
        "fetch_schema_variations.py",
        "fetch_real_s3.py",
        "fetch_real_s3_simple.py",
        "inspect_pgvector.py",
    ]

    bare_except_violations = []

    for filename in root_files:
        file_path = Path(filename)
        if not file_path.exists():
            continue

        try:
            with file_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
                for i, line in enumerate(lines, 1):
                    stripped = line.strip()
                    # Match bare except clauses (only "except:" or "except:" + whitespace)
                    if stripped == "except:" or (
                        stripped.startswith("except:") and len(stripped) <= 7
                    ):
                        bare_except_violations.append(f"{filename}:{i}: {line.strip()}")
        except Exception as e:
            print(f"Warning: Could not check {filename}: {e}", file=sys.stderr)

    if bare_except_violations:
        error_msg = "Found bare except clauses in root utility files:\n" + "\n".join(
            bare_except_violations
        )
        error_msg += "\n\nPlease use specific exception types instead of bare except."
        raise AssertionError(error_msg)


def test_no_bare_except_in_files():
    """Combined test to check for bare except clauses across the codebase."""
    test_no_bare_except_in_backend()
    test_no_bare_except_in_root_scripts()
