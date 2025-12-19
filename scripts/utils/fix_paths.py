import re
from pathlib import Path


def fix_file(path: Path):
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
        if (
            "from pathlib import Path" not in content
            and "import pathlib" not in content
        ):
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
