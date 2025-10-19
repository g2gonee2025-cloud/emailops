#!/usr/bin/env python3
"""
Script to fix common linting errors in the emailops project.
"""

import re
from pathlib import Path


def fix_exception_handling(content: str) -> str:
    """Fix B904: Add 'from None' to exception raising."""
    # Pattern to find raise statements without 'from' clause
    pattern = r'(\s+)(raise\s+\w+\([^)]+\))(\s*)$'

    def replacement(match):
        indent, raise_stmt, trailing = match.groups()
        # Check if it's already has 'from' clause
        if 'from' in raise_stmt:
            return match.group(0)
        return f"{indent}{raise_stmt} from None{trailing}"

    # Apply to lines that look like they're in except blocks
    lines = content.split('\n')
    in_except = False
    result_lines = []

    for i, line in enumerate(lines):
        if 'except' in line and ':' in line:
            in_except = True
        elif line and not line[0].isspace() and not line.strip().startswith('#'):
            in_except = False

        if in_except and line.strip().startswith('raise '):
            if 'from' not in line:
                line = re.sub(r'^(\s*raise\s+.+)$', r'\1 from None', line)

        result_lines.append(line)

    return '\n'.join(result_lines)


def fix_path_open(content: str) -> str:
    """Fix PTH123: Replace Path.open() with Path.open()."""
    lines = content.split('\n')
    result_lines = []

    for line in lines:
        # Skip if it's already using Path
        if 'Path(' in line and '.open(' in line:
            result_lines.append(line)
            continue

        # Pattern to match Path.open() calls with path variable
        if re.match(r'^\s*with\s+open\(', line):
            # Extract the path variable
            match = re.match(r'^(\s*)with\s+open\(([^,\)]+)', line)
            if match:
                indent, path_var = match.groups()
                # Check if path_var looks like a simple variable (not a string literal)
                if not (path_var.startswith('"') or path_var.startswith("'")):
                    # Replace Path.open(path_var, ...) with Path(path_var).open(...)
                    line = re.sub(
                        r'open\(([^,\)]+)(.*)\)',
                        r'Path(\1).open(\2)',
                        line
                    )
                    # Add Path import if needed
                    if 'Path(' in line and 'from pathlib import Path' not in content:
                        # Will handle imports separately
                        pass

        result_lines.append(line)

    return '\n'.join(result_lines)


def fix_imports_order(content: str) -> str:
    """Fix E402: Move imports to top of file."""
    lines = content.split('\n')

    # Find the docstring end
    in_docstring = False
    docstring_end = 0
    for i, line in enumerate(lines):
        if '"""' in line or "'''" in line:
            if not in_docstring:
                in_docstring = True
            else:
                docstring_end = i + 1
                break

    # Collect all imports
    imports = []
    other_lines = []

    for i, line in enumerate(lines):
        if i <= docstring_end:
            other_lines.append(line)
        elif line.startswith('import ') or line.startswith('from '):
            imports.append(line)
        else:
            other_lines.append(line)

    # Sort imports (standard library first, then third party, then local)
    std_imports = []
    third_party_imports = []
    local_imports = []

    for imp in imports:
        if imp.startswith('from .'):
            local_imports.append(imp)
        elif any(imp.startswith(f'import {lib}') or imp.startswith(f'from {lib}')
                for lib in ['os', 'sys', 'json', 'logging', 're', 'asyncio',
                           'pathlib', 'dataclasses', 'typing', 'unicodedata']):
            std_imports.append(imp)
        else:
            third_party_imports.append(imp)

    # Reconstruct file
    result = []
    # Add everything up to docstring end
    result.extend(lines[:docstring_end + 1])

    # Add sorted imports
    if std_imports:
        result.extend(sorted(set(std_imports)))
    if std_imports and (third_party_imports or local_imports):
        result.append('')
    if third_party_imports:
        result.extend(sorted(set(third_party_imports)))
    if third_party_imports and local_imports:
        result.append('')
    if local_imports:
        result.extend(sorted(set(local_imports)))

    # Add rest of file (skipping original import lines)
    result.append('')
    for i, line in enumerate(lines[docstring_end + 1:], docstring_end + 1):
        if not (line.startswith('import ') or line.startswith('from ')):
            result.append(line)

    return '\n'.join(result)


def fix_file(filepath: Path) -> bool:
    """Fix a single Python file."""
    try:
        content = filepath.read_text(encoding='utf-8')
        original = content

        # Apply fixes
        content = fix_imports_order(content)
        content = fix_exception_handling(content)
        # Skip Path.open() fix for now as it's complex

        # Write back if changed
        if content != original:
            filepath.write_text(content, encoding='utf-8')
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    """Main function to fix linting errors."""
    emailops_dir = Path('emailops')
    helpers_dir = Path('helpers & diagnostics')

    fixed_files = []

    # Process emailops directory
    if emailops_dir.exists():
        for py_file in emailops_dir.glob('*.py'):
            if fix_file(py_file):
                fixed_files.append(py_file)

    # Process helpers directory
    if helpers_dir.exists():
        for py_file in helpers_dir.glob('*.py'):
            if fix_file(py_file):
                fixed_files.append(py_file)

    if fixed_files:
        print(f"Fixed {len(fixed_files)} files:")
        for f in fixed_files:
            print(f"  - {f}")
    else:
        print("No files were modified.")

    return len(fixed_files)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() > 0 else 1)
