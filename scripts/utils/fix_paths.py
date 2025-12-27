import re
from pathlib import Path


def fix_file(path: Path):
    """
    Normalize hard-coded sys.path manipulations in a Python source file.

    Parameters
    ----------
    path : Path
        Path to the file to process.

    Side effects
    ------------
    Reads the file, applies regex-based replacements to known sys.path-append
    patterns, and overwrites the file if any changes are made.
    """
    content = path.read_text(encoding="utf-8")

    # Pattern 1: Common os.getcwd() pattern
    pattern1 = (
        r'sys\.path\.append\(os\.path\.join\(os\.getcwd\(\), "backend", "src"\)\)'
    )

    # Replacement for depth 2 (scripts/category/script.py)
    # We need to ensure Path is imported if not already, but usually it's easier to just inject the whole block cleanly
    # if we match the pattern.

    replacement = (
        'sys.path.append(str(Path(__file__).resolve().parents[2] / "backend" / "src"))'
    )

    if re.search(pattern1, content):
        print(f"Fixing {path}")

        # Ensure pathlib is imported
        if not re.search(
            r"^\s*from\s+pathlib\s+import\s+(\*|(?:[^#\n]*\bPath\b(?!\s+as)))",
            content,
            re.M,
        ) and not re.search(r"^\s*import\s+pathlib\b(?!\s+as\b)", content, re.M):
            # Add import after first import or at top
            if "import os" in content:
                content = content.replace(
                    "import os", "import os\nfrom pathlib import Path"
                )
            elif "import sys" in content:
                content = content.replace(
                    "import sys", "import sys\nfrom pathlib import Path"
                )

        new_content = re.sub(pattern1, replacement, content)
        path.write_text(new_content, encoding="utf-8")


def main():
    """Iterate predefined subdirectories from this script's directory and fix paths."""
    root = Path(__file__).parent
    # Iterate over subdirectories
    for cat in ["verification", "ingestion", "ops", "search", "legacy"]:
        subdir = root / cat
        if not subdir.exists():
            continue
        for py_file in subdir.glob("*.py"):
            fix_file(py_file)


if __name__ == "__main__":
    main()
