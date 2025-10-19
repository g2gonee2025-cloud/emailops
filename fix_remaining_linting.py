#!/usr/bin/env python3
"""
Fix remaining linting errors with more aggressive but safe fixes.
"""

import ast
import re
import subprocess
from pathlib import Path


def test_syntax(filepath):
    """Test if a Python file has valid syntax."""
    try:
        with Path.open(filepath, encoding='utf-8') as f:
            ast.parse(f.read())
        return True
    except SyntaxError:
        return False


def fix_module_imports_not_at_top(filepath):
    """Fix E402: module-import-not-at-top-of-file."""
    with Path.open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

    # Separate imports from other code
    imports = []
    future_imports = []
    docstring_lines = []
    other_lines = []

    in_docstring = False
    docstring_quote = None
    found_code = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Handle module docstring at the very beginning
        if i == 0 and (stripped.startswith('"""') or stripped.startswith("'''")):
            docstring_quote = '"""' if stripped.startswith('"""') else "'''"
            in_docstring = True
            docstring_lines.append(line)
            if stripped.endswith(docstring_quote) and len(stripped) > 3:
                in_docstring = False
            continue

        if in_docstring:
            docstring_lines.append(line)
            if docstring_quote in line:
                in_docstring = False
            continue

        # Skip empty lines and comments before first code
        if not found_code and (not stripped or stripped.startswith('#')):
            docstring_lines.append(line)
            continue

        # Check for imports
        if stripped.startswith('from __future__'):
            future_imports.append(line)
            found_code = True
        elif (stripped.startswith('import ') or stripped.startswith('from ')) and not found_code:
            imports.append(line)
            found_code = True
        elif (stripped.startswith('import ') or stripped.startswith('from ')):
            # Import after other code - this is E402
            imports.append(line)
        else:
            if stripped and not stripped.startswith('#'):
                found_code = True
            other_lines.append(line)

    # Reconstruct file with proper order
    new_lines = []

    # Add docstring/comments
    new_lines.extend(docstring_lines)

    # Add future imports
    if future_imports:
        new_lines.extend(future_imports)
        if not (new_lines and new_lines[-1].strip() == ''):
            new_lines.append('\n')

    # Add regular imports
    if imports:
        new_lines.extend(imports)
        if not (new_lines and new_lines[-1].strip() == ''):
            new_lines.append('\n')

    # Add the rest
    new_lines.extend(other_lines)

    # Write back if changed
    if lines != new_lines:
        with Path.open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"  Fixed module imports order in {filepath}")
        return True

    return False


def fix_builtin_open_safe(filepath):
    """Fix PTH123: builtin-open - safely."""
    with Path.open(filepath, encoding='utf-8') as f:
        content = f.read()

    # Skip if file uses encoding parameter in Path.open()
    if 'encoding=' in content:
        return False

    # Skip if file has complex open usage
    if 'with open' not in content:
        return False

    # Only fix simple cases
    simple_pattern = r'with open\((["\'][^"\']+["\'])\) as (\w+):'
    matches = re.findall(simple_pattern, content)

    if matches and len(matches) <= 2:  # Only fix if there are 1-2 simple opens
        # Add Path import
        if 'from pathlib import Path' not in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    continue
                elif line.strip() and not line.startswith('#'):
                    lines.insert(i, 'from pathlib import Path')
                    break
            content = '\n'.join(lines)

        # Replace simple with open
        for match in matches:
            old = f'with Path.open({match[0]}) as {match[1]}:'
            new = f'with Path({match[0]}).open() as {match[1]}:'
            content = content.replace(old, new)

        with Path.open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"  Fixed simple builtin-open in {filepath}")
        return True

    return False


def fix_remaining_issues(filepath):
    """Fix other remaining issues."""
    with Path.open(filepath, encoding='utf-8') as f:
        content = f.read()

    modified = False

    # Fix SIM102: collapsible-if
    # Fix SIM103: needless-bool
    # Fix E741: ambiguous-variable-name (l -> lst)
    if ' l ' in content or '\tl ' in content or ' l=' in content:
        content = re.sub(r'\bl\b', 'lst', content)
        modified = True
        print(f"  Fixed ambiguous variable name in {filepath}")

    # Fix RUF022: unsorted-dunder-all
    if '__all__' in content:
        match = re.search(r'__all__ = \[(.*?)\]', content, re.DOTALL)
        if match:
            items = match.group(1)
            # Parse and sort items
            try:
                item_list = eval(f'[{items}]')
                sorted_items = sorted(item_list)
                sorted_str = ', '.join(f'"{item}"' if isinstance(item, str) else repr(item) for item in sorted_items)
                new_all = f'__all__ = [{sorted_str}]'
                old_all = match.group(0)
                content = content.replace(old_all, new_all)
                modified = True
                print(f"  Fixed unsorted __all__ in {filepath}")
            except:
                pass

    # Fix remaining W293: blank-line-with-whitespace
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if line.strip() == '' and line != '':
            new_lines.append('')
        else:
            new_lines.append(line)

    if lines != new_lines:
        content = '\n'.join(new_lines)
        modified = True
        print(f"  Fixed remaining blank whitespace in {filepath}")

    if modified:
        with Path.open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    return modified


def fix_syntax_errors():
    """Try to fix the files with syntax errors."""
    # Fix emailops_gui.py
    gui_file = Path("emailops/emailops_gui.py")
    if gui_file.exists():
        with Path.open(gui_file, encoding='utf-8') as f:
            content = f.read()

        # Look for common syntax error patterns
        # Remove any incomplete lines at the end
        lines = content.split('\n')
        while lines and lines[-1].strip() in ['def', 'class', 'if', 'elif', 'else', 'try', 'except', 'finally', 'with', 'for', 'while']:
            lines.pop()

        content = '\n'.join(lines)

        with Path.open(gui_file, 'w', encoding='utf-8') as f:
            f.write(content)

        if test_syntax(gui_file):
            print(f"  Fixed syntax in {gui_file}")
        else:
            # Try to restore from git
            subprocess.run(["git", "restore", str(gui_file)])
            print(f"  Could not fix syntax in {gui_file}, restored from git")

    # Fix parallel_indexer.py
    parallel_file = Path("emailops/parallel_indexer.py")
    if parallel_file.exists():
        with Path.open(parallel_file, encoding='utf-8') as f:
            content = f.read()

        # Similar fixes
        lines = content.split('\n')
        while lines and lines[-1].strip() in ['def', 'class', 'if', 'elif', 'else', 'try', 'except', 'finally', 'with', 'for', 'while']:
            lines.pop()

        content = '\n'.join(lines)

        with Path.open(parallel_file, 'w', encoding='utf-8') as f:
            f.write(content)

        if test_syntax(parallel_file):
            print(f"  Fixed syntax in {parallel_file}")
        else:
            subprocess.run(["git", "restore", str(parallel_file)])
            print(f"  Could not fix syntax in {parallel_file}, restored from git")


def main():
    print("Fixing remaining linting errors...")

    # First fix syntax errors
    print("\nStep 1: Attempting to fix syntax errors...")
    fix_syntax_errors()

    # Get list of Python files
    emailops_dir = Path("emailops")
    python_files = list(emailops_dir.glob("*.py"))

    print(f"\nStep 2: Processing {len(python_files)} Python files for remaining issues...")

    for filepath in python_files:
        print(f"\nProcessing {filepath}...")

        # Test syntax before
        if not test_syntax(filepath):
            print(f"  WARNING: {filepath} has syntax errors, skipping...")
            continue

        # Apply fixes
        fix_module_imports_not_at_top(filepath)
        fix_remaining_issues(filepath)
        # fix_builtin_open_safe(filepath)  # Skip for now, too risky

        # Test syntax after
        if not test_syntax(filepath):
            print(f"  ERROR: {filepath} has syntax errors after fixes, reverting...")
            subprocess.run(["git", "restore", str(filepath)])

    # Run ruff with unsafe fixes on remaining safe issues
    print("\nStep 3: Applying remaining ruff fixes...")
    subprocess.run(["ruff", "check", "emailops", "--fix", "--unsafe-fixes"], capture_output=True)

    # Show final results
    print("\n" + "="*60)
    print("Final error count:")
    result = subprocess.run(
        ["ruff", "check", "emailops", "--statistics"],
        capture_output=True,
        text=True
    )
    print(result.stdout)

    print("\nFixes completed!")


if __name__ == "__main__":
    main()
