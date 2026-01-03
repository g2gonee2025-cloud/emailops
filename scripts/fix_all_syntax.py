import argparse
import os
import re
import sys

# Regex to match a non-greedy __all__ tuple ending with a bracket instead of a parenthesis.
# Using a non-greedy match `*?` is critical to avoid spanning multiple `__all__` definitions
# or over-matching into unrelated code.
PATTERN = re.compile(r"(__all__\s*=\s*\([^\]]*?)]", re.DOTALL)


def fix_file(path, dry_run=True):
    """
    Reads a Python file, fixes the __all__ syntax, and writes it back.
    Returns True if the file was modified, False otherwise.
    """
    try:
        with open(path, encoding="utf-8", errors="surrogateescape") as f:
            content = f.read()

        if "__all__" not in content:
            return False

        new_content = PATTERN.sub(r"\1)", content)

        if new_content != content:
            print(f"Fixing {path}")
            if not dry_run:
                # Create a backup before writing
                os.rename(path, f"{path}.bak")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_content)
            return True
    except (OSError, UnicodeDecodeError) as e:
        print(f"Error processing {path}: {e}", file=sys.stderr)
    return False


def main():
    parser = argparse.ArgumentParser(
        description="""
        Recursively scans a directory for Python files and fixes a specific syntax error
        in `__all__` definitions where a tuple is incorrectly closed with `]` instead of `)`.
        """,
        epilog="""
        By default, the script runs in dry-run mode and will not modify any files.
        Use the --no-dry-run flag to apply changes. When changes are applied, backups of the
        original files are created with a .bak extension.
        """,
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="The target directory to scan for .py files. Defaults to the current directory.",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="Apply the fixes to the files. If not specified, runs in dry-run mode.",
    )

    args = parser.parse_args()
    target_directory = args.directory
    dry_run = args.dry_run

    if not os.path.isdir(target_directory):
        print(
            f"Error: The specified directory does not exist: {target_directory}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Scanning directory: {target_directory}")
    if dry_run:
        print("Running in dry-run mode. No files will be modified.")

    count = 0
    for dirpath, _, files in os.walk(target_directory, followlinks=False):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(dirpath, file)
                if os.path.islink(path):
                    continue
                if fix_file(path, dry_run=dry_run):
                    count += 1

    print(
        f"\nTotal files to be fixed: {count}"
        if dry_run
        else f"\nTotal fixed files: {count}"
    )

    if dry_run and count > 0:
        print(
            "To apply these changes, run the script again with the --no-dry-run flag."
        )


if __name__ == "__main__":
    main()
