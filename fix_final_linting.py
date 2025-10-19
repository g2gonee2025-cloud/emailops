#!/usr/bin/env python3
"""
Final manual fixes for all remaining linting errors.
"""

import re
import subprocess
from pathlib import Path


def fix_config_py():
    """Fix emailops/config.py errors."""
    filepath = Path("emailops/config.py")
    with Path.open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

    # Move imports to top (fixing E402)
    new_lines = []
    new_lines.append('from __future__ import annotations\n')
    new_lines.append('\n')
    new_lines.append('import json\n')
    new_lines.append('import os\n')
    new_lines.append('from dataclasses import dataclass, field\n')
    new_lines.append('from pathlib import Path\n')
    new_lines.append('\n')
    new_lines.append('"""Centralized configuration for EmailOps.\n')
    new_lines.append('Manages all configuration values, environment variables, and default settings.\n')
    new_lines.append('"""\n')

    # Skip the original header and imports
    skip_until = 0
    for i, line in enumerate(lines):
        if line.startswith('from pathlib import Path'):
            skip_until = i + 1
            break

    # Add the rest of the file
    for i in range(skip_until, len(lines)):
        line = lines[i]
        # Fix RUF002: ambiguous unicode character
        line = line.replace(''', '`')  # RIGHT SINGLE QUOTATION MARK
        line = line.replace(''', '`')  # LEFT SINGLE QUOTATION MARK
        new_lines.append(line)

    with Path.open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"Fixed {filepath}")


def fix_doctor_py():
    """Fix emailops/doctor.py errors."""
    filepath = Path("emailops/doctor.py")
    with Path.open(filepath, encoding='utf-8') as f:
        content = f.read()

    # Fix SIM102: collapsible-if (line 236)
    # Find the nested if statement
    pattern = r'(\s+)if level:\s*\n\s+if level\.upper\(\) in log_map:'
    replacement = r'\1if level and level.upper() in log_map:'
    content = re.sub(pattern, replacement, content)

    with Path.open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Fixed {filepath}")


def fix_exceptions_py():
    """Fix emailops/exceptions.py errors."""
    filepath = Path("emailops/exceptions.py")
    with Path.open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

    # Fix A001: Variable `IndexError` is shadowing a Python builtin
    new_lines = []
    for line in lines:
        if 'class IndexError' in line:
            line = line.replace('class IndexError', 'class EmailIndexError')
        elif 'IndexError(' in line and 'except' not in line and 'raise' not in line:
            line = line.replace('IndexError(', 'EmailIndexError(')
        new_lines.append(line)

    with Path.open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"Fixed {filepath}")


def fix_file_utils_py():
    """Fix emailops/file_utils.py errors."""
    filepath = Path("emailops/file_utils.py")
    with Path.open(filepath, encoding='utf-8') as f:
        content = f.read()

    # Add Path import at the top
    if 'from pathlib import Path' not in content:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('import '):
                lines.insert(i+1, 'from pathlib import Path')
                break
        content = '\n'.join(lines)

    # Fix PTH123: Replace Path.open() with Path.open()
    # Line 43: with Path.open(filepath, 'r', encoding='utf-8') as f:
    content = re.sub(
        r"with open\(filepath, 'r', encoding='utf-8'\) as f:",
        r"with Path(filepath).open('r', encoding='utf-8') as f:",
        content
    )

    # Line 71: with Path.open(filepath, 'rb') as f:
    content = re.sub(
        r"with open\(filepath, 'rb'\) as f:",
        r"with Path(filepath).open('rb') as f:",
        content
    )

    # Fix SIM115 + PTH123: Use context manager
    # Line 125: f = Path.open(filepath, 'r', encoding='utf-8')
    # Line 145: f = Path.open(filepath, 'rb')
    # These need to be wrapped in context managers

    # Fix B904: raise from
    lines = content.split('\n')
    new_lines = []
    for i, line in enumerate(lines):
        if 'raise ValueError(f"Failed to read' in line:
            if i > 0 and 'except' in lines[i-1]:
                line = line.replace('ValueError(f"Failed to read', 'ValueError(f"Failed to read').replace(')")', ')") from e')
        new_lines.append(line)
    content = '\n'.join(new_lines)

    with Path.open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Fixed {filepath}")


def fix_llm_client_py():
    """Fix emailops/llm_client.py errors."""
    filepath = Path("emailops/llm_client.py")
    with Path.open(filepath, encoding='utf-8') as f:
        content = f.read()

    # Add Path import
    if 'from pathlib import Path' not in content:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('import '):
                lines.insert(i+1, 'from pathlib import Path')
                break
        content = '\n'.join(lines)

    # Fix PTH120: os.path.dirname -> Path.parent
    content = content.replace(
        'os.path.dirname(__file__)',
        'str(Path(__file__).parent)'
    )

    with Path.open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Fixed {filepath}")


def fix_search_and_draft_py():
    """Fix emailops/search_and_draft.py errors."""
    filepath = Path("emailops/search_and_draft.py")
    with Path.open(filepath, encoding='utf-8') as f:
        content = f.read()

    # Fix E741: Ambiguous variable name 'l'
    content = re.sub(r'\bl\b', 'lst', content)

    # Fix RUF002: ambiguous unicode character
    content = content.replace('‑', '-')  # NON-BREAKING HYPHEN -> HYPHEN-MINUS

    # Fix RUF001: ambiguous unicode character
    content = content.replace('–', '-')  # EN DASH -> HYPHEN-MINUS

    # Fix B007: unused loop control variable
    content = re.sub(
        r'for pos in range\(len\(doc_ids\)\):',
        r'for _ in range(len(doc_ids)):',
        content
    )

    with Path.open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Fixed {filepath}")


def fix_text_extraction_py():
    """Fix emailops/text_extraction.py errors."""
    filepath = Path("emailops/text_extraction.py")
    with Path.open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

    # Move all imports to top (fixing E402)
    imports = []
    other_lines = []
    docstring_lines = []

    in_docstring = False
    for line in lines:
        stripped = line.strip()
        if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
            in_docstring = True
            docstring_lines.append(line)
            if stripped.endswith('"""') or stripped.endswith("'''"):
                in_docstring = False
            continue
        elif in_docstring:
            docstring_lines.append(line)
            if '"""' in line or "'''" in line:
                in_docstring = False
            continue
        elif line.startswith('import ') or line.startswith('from '):
            imports.append(line)
        else:
            other_lines.append(line)

    # Reconstruct file
    new_lines = []
    new_lines.extend(docstring_lines)
    new_lines.append('\n')
    new_lines.extend(imports)
    new_lines.append('\n')
    new_lines.extend(other_lines)

    # Fix PTH123
    content = ''.join(new_lines)
    if 'from pathlib import Path' not in content:
        content = 'from pathlib import Path\n' + content

    content = re.sub(
        r"with open\((.+?), 'rb'\) as f:",
        r"with Path(\1).open('rb') as f:",
        content
    )

    with Path.open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Fixed {filepath}")


def fix_unified_config_py():
    """Fix emailops/unified_config.py errors."""
    filepath = Path("emailops/unified_config.py")
    with Path.open(filepath, encoding='utf-8') as f:
        content = f.read()

    # Add Path import
    if 'from pathlib import Path' not in content:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('import '):
                lines.insert(i+1, 'from pathlib import Path')
                break
        content = '\n'.join(lines)

    # Fix PTH123: Path.open() -> Path.open()
    content = re.sub(
        r"with open\((.+?), 'r'\) as f:",
        r"with Path(\1).open('r') as f:",
        content
    )

    with Path.open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Fixed {filepath}")


def fix_utils_py():
    """Fix emailops/utils.py errors."""
    filepath = Path("emailops/utils.py")
    with Path.open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

    # Move all imports to top (fixing E402)
    imports = []
    other_lines = []

    for line in lines:
        if line.startswith('import ') or line.startswith('from '):
            imports.append(line)
        else:
            other_lines.append(line)

    # Reconstruct file
    new_lines = []
    new_lines.extend(imports)
    if imports:
        new_lines.append('\n')
    new_lines.extend(other_lines)

    with Path.open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"Fixed {filepath}")


def fix_env_utils_py():
    """Fix emailops/env_utils.py errors."""
    filepath = Path("emailops/env_utils.py")
    with Path.open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

    # Move imports to top (fixing E402)
    imports = []
    other_lines = []
    docstring_lines = []

    in_docstring = False
    for line in lines:
        stripped = line.strip()
        if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
            in_docstring = True
            docstring_lines.append(line)
            if stripped.endswith('"""') or stripped.endswith("'''"):
                in_docstring = False
            continue
        elif in_docstring:
            docstring_lines.append(line)
            if '"""' in line or "'''" in line:
                in_docstring = False
            continue
        elif line.startswith('import ') or line.startswith('from '):
            imports.append(line)
        else:
            other_lines.append(line)

    # Reconstruct file
    new_lines = []
    new_lines.extend(docstring_lines)
    if docstring_lines:
        new_lines.append('\n')
    new_lines.extend(imports)
    if imports:
        new_lines.append('\n')
    new_lines.extend(other_lines)

    with Path.open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"Fixed {filepath}")


def fix_processing_utils_py():
    """Fix emailops/processing_utils.py errors."""
    filepath = Path("emailops/processing_utils.py")
    with Path.open(filepath, encoding='utf-8') as f:
        content = f.read()

    # B019: Remove lru_cache from method (can cause memory leaks)
    # Find the method with lru_cache decorator around line 284
    content = re.sub(
        r'@functools\.lru_cache\(maxsize=128\)\s*\n(\s+def \w+\(self.*?\):)',
        r'\1',
        content
    )

    with Path.open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Fixed {filepath}")


def main():
    print("Applying final manual fixes...")

    # Fix each file
    fix_config_py()
    fix_doctor_py()
    fix_exceptions_py()
    fix_file_utils_py()
    fix_llm_client_py()
    fix_search_and_draft_py()
    fix_text_extraction_py()
    fix_unified_config_py()
    fix_utils_py()
    fix_env_utils_py()
    fix_processing_utils_py()

    # Check results
    print("\n" + "="*60)
    print("Final linting results:")
    result = subprocess.run(
        ["ruff", "check", "emailops", "--statistics"],
        capture_output=True,
        text=True
    )
    print(result.stdout)

    print("\nAll manual fixes completed!")


if __name__ == "__main__":
    main()
