#!/usr/bin/env python3
"""
Safe linting fix script that applies fixes incrementally and tests for syntax errors.
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path


def test_syntax(filepath):
    """Test if a Python file has valid syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        return True
    except SyntaxError:
        return False


def run_ruff_fix_safe(directory="emailops"):
    """Run ruff with only safe fixes."""
    print("Running ruff with safe fixes only...")
    result = subprocess.run(
        ["ruff", "check", directory, "--fix"],
        capture_output=True,
        text=True
    )
    print(f"Applied safe fixes. Exit code: {result.returncode}")
    return result.returncode == 0


def fix_blank_whitespace(filepath):
    """Fix W293: blank-line-with-whitespace."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    modified = False
    new_lines = []
    for line in lines:
        if line.strip() == '' and line != '\n':
            new_lines.append('\n')
            modified = True
        else:
            new_lines.append(line)

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"  Fixed blank whitespace in {filepath}")

    return modified


def fix_builtin_open(filepath):
    """Fix PTH123: builtin-open - convert open() to Path().open()."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if Path is already imported
    has_path_import = 'from pathlib import Path' in content

    # Find all open() calls
    pattern = r'\bopen\s*\('
    if re.search(pattern, content):
        # Add Path import if needed
        if not has_path_import:
            # Add import at the top after other imports
            lines = content.split('\n')
            import_added = False
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    continue
                elif line.strip() and not line.startswith('#'):
                    # Insert before first non-import line
                    lines.insert(i, 'from pathlib import Path')
                    lines.insert(i+1, '')
                    import_added = True
                    break

            if not import_added:
                lines.insert(0, 'from pathlib import Path')
                lines.insert(1, '')

            content = '\n'.join(lines)

        # Replace open() with Path().open()
        # Be careful not to replace if it's already Path().open()
        content = re.sub(
            r'\bopen\s*\((["\'])([^"\']+)\1',
            r'Path(\1\2\1).open',
            content
        )

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"  Fixed builtin-open in {filepath}")
        return True

    return False


def fix_unicode_characters(filepath):
    """Fix RUF001/RUF002/RUF003: ambiguous unicode characters."""
    replacements = {
        ''': "'",  # Right single quotation mark
        ''': "'",  # Left single quotation mark
        '"': '"',  # Left double quotation mark
        '"': '"',  # Right double quotation mark
        '–': '-',  # En dash
        '—': '-',  # Em dash
        '…': '...',  # Ellipsis
    }

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    modified = False
    for old, new in replacements.items():
        if old in content:
            content = content.replace(old, new)
            modified = True

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Fixed unicode characters in {filepath}")

    return modified


def fix_exception_handling(filepath):
    """Fix B904: raise-without-from-inside-except."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    modified = False
    new_lines = []
    in_except = False

    for i, line in enumerate(lines):
        if 'except' in line and ':' in line:
            in_except = True
        elif in_except and line.strip().startswith('raise '):
            # Check if it's a bare raise or has 'from'
            if ' from ' not in line and line.strip() != 'raise':
                # Look for the exception variable in the except clause
                if i > 0:
                    for j in range(i-1, max(0, i-10), -1):
                        if 'except' in lines[j]:
                            match = re.search(r'except .+ as (\w+)', lines[j])
                            if match:
                                exc_var = match.group(1)
                                line = line.rstrip() + f' from {exc_var}\n'
                                modified = True
                            break
        elif not line.strip().startswith(' ') and line.strip():
            in_except = False

        new_lines.append(line)

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"  Fixed exception handling in {filepath}")

    return modified


def get_ruff_errors(directory="emailops"):
    """Get current ruff errors."""
    result = subprocess.run(
        ["ruff", "check", directory, "--output-format", "json"],
        capture_output=True,
        text=True
    )

    import json
    try:
        errors = json.loads(result.stdout)
        return errors
    except:
        return []


def main():
    print("Starting safe linting fixes...")

    # Step 1: Apply safe ruff fixes
    print("\nStep 1: Applying safe ruff fixes...")
    run_ruff_fix_safe()

    # Step 2: Get list of Python files
    emailops_dir = Path("emailops")
    python_files = list(emailops_dir.glob("*.py"))

    print(f"\nStep 2: Processing {len(python_files)} Python files...")

    # Step 3: Apply manual fixes file by file
    for filepath in python_files:
        print(f"\nProcessing {filepath}...")

        # Test syntax before
        if not test_syntax(filepath):
            print(f"  WARNING: {filepath} has syntax errors, skipping...")
            continue

        # Apply fixes
        fix_blank_whitespace(filepath)
        fix_unicode_characters(filepath)
        fix_exception_handling(filepath)

        # Don't apply builtin-open fix for now as it might break things
        # fix_builtin_open(filepath)

        # Test syntax after
        if not test_syntax(filepath):
            print(f"  ERROR: {filepath} has syntax errors after fixes, reverting...")
            subprocess.run(["git", "restore", str(filepath)])

    # Step 4: Show remaining errors
    print("\n" + "="*60)
    print("Checking remaining errors...")
    result = subprocess.run(
        ["ruff", "check", "emailops", "--statistics"],
        capture_output=True,
        text=True
    )
    print(result.stdout)

    # Count errors
    lines = result.stdout.strip().split('\n')
    if lines[-1].startswith('Found'):
        print(f"\n{lines[-1]}")

    print("\nSafe fixes completed!")
    print("Note: Some errors like E402 (module-import-not-at-top) and PTH123 (builtin-open)")
    print("were not auto-fixed to avoid breaking the code.")


if __name__ == "__main__":
    main()
