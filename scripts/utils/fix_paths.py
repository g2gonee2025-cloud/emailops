import ast
import re
from pathlib import Path


def has_path_import(content: str) -> bool:
    """
    Check if 'from pathlib import Path' is present using AST parsing.

    Falls back to regex for syntactically invalid files.
    """
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "pathlib":
                for alias in node.names:
                    if alias.name == "Path" or alias.name == "*":
                        return True
    except SyntaxError:
        # Fallback for files with syntax errors that ast can't parse
        import_pattern = re.compile(
            r"^\s*from\s+pathlib\s+import\s+.*(?:Path|\*).*", re.MULTILINE
        )
        return bool(import_pattern.search(content))
    return False


PROJECT_ROOT_DEPTH = 2
replacement = f'sys.path.append(str(Path(__file__).resolve().parents[{PROJECT_ROOT_DEPTH}] / "backend" / "src"))'


def fix_file_content(content: str) -> str | None:
    """
    Normalize hard-coded sys.path manipulations in a Python source file content.

    If changes are needed, returns the new content, otherwise returns None.
    """
    # Pattern 1: Common os.getcwd() pattern
    path_pattern = re.compile(
        r'sys\.path\.append\(os\.path\.join\(os\.getcwd\(\), "backend", "src"\)\)'
    )

    if not path_pattern.search(content):
        return None

    if not has_path_import(content):
        # Find the last import statement to add the new import after it
        last_import_match = list(
            re.finditer(r"^(?:from\s.+)?import\s.+(?:\n|$)", content, re.MULTILINE)
        )
        if last_import_match:
            insert_pos = last_import_match[-1].end()
            content = (
                content[:insert_pos]
                + "from pathlib import Path\n"
                + content[insert_pos:]
            )
        else:
            # No imports found, add at the top
            content = "from pathlib import Path\n" + content

    new_content = path_pattern.sub(replacement, content)
    return new_content


def main():
    """Iterate predefined subdirectories from this script's directory and fix paths."""
    root = Path(__file__).parent
    # Iterate over subdirectories
    for cat in ["verification", "ingestion", "ops", "search", "legacy"]:
        subdir = root / cat
        if not subdir.exists():
            continue
        for py_file in subdir.glob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8")
                new_content = fix_file_content(content)
                if new_content:
                    print(f"Fixing {py_file}")
                    py_file.write_text(new_content, encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                print(f"Error processing file {py_file}: {e}")


if __name__ == "__main__":
    main()
