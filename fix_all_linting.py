#!/usr/bin/env python3
"""
Comprehensive script to fix all linting errors in the emailops project.
"""

import re
import sys
from pathlib import Path


def fix_imports_at_top(content: str) -> str:
    """Fix E402: Move all imports to top of file (after docstring)."""
    lines = content.split('\n')

    # Find the end of module docstring
    in_docstring = False
    docstring_end = -1
    docstring_quote = None

    for i, line in enumerate(lines):
        if not in_docstring:
            if line.strip().startswith('"""') or line.strip().startswith("'''"):
                docstring_quote = '"""' if '"""' in line else "'''"
                if line.count(docstring_quote) == 2:  # Single line docstring
                    docstring_end = i
                else:
                    in_docstring = True
        else:
            if docstring_quote in line:
                docstring_end = i
                break

    # Collect all imports
    imports = []
    non_imports = []

    for i, line in enumerate(lines):
        if i <= docstring_end:
            non_imports.append(line)
        elif line.strip().startswith('import ') or line.strip().startswith('from '):
            imports.append(line)
        else:
            non_imports.append(line)

    # Sort imports
    std_imports = []
    third_party = []
    local_imports = []

    for imp in imports:
        if imp.strip().startswith('from .'):
            local_imports.append(imp)
        elif any(imp.strip().startswith(f'import {lib}') or imp.strip().startswith(f'from {lib}')
                for lib in ['os', 'sys', 'json', 'logging', 're', 'asyncio', 'time', 'threading',
                           'pathlib', 'dataclasses', 'typing', 'unicodedata', 'contextlib',
                           'functools', 'collections']):
            std_imports.append(imp)
        elif imp.strip():
            third_party.append(imp)

    # Rebuild content
    result = lines[:docstring_end + 1]
    result.append('')

    if std_imports:
        result.extend(sorted(set(std_imports)))
    if third_party:
        if std_imports:
            result.append('')
        result.extend(sorted(set(third_party)))
    if local_imports:
        if std_imports or third_party:
            result.append('')
        result.extend(sorted(set(local_imports)))

    # Add non-import lines
    result.append('')
    skip_blank = True
    for i in range(docstring_end + 1, len(lines)):
        line = lines[i]
        if not (line.strip().startswith('import ') or line.strip().startswith('from ')):
            if skip_blank and not line.strip():
                continue
            skip_blank = False
            result.append(line)

    return '\n'.join(result)


def fix_open_to_path(content: str) -> str:
    """Fix PTH123: Replace Path.open() with Path().open()."""
    # Add Path import if needed
    if 'from pathlib import Path' not in content:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                lines.insert(i, 'from pathlib import Path')
                content = '\n'.join(lines)
                break

    # Replace Path.open() calls
    patterns = [
        (r'with open\(([^,\)]+),', r'with Path(\1).open('),
        (r'= open\(([^,\)]+),', r'= Path(\1).open('),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    return content


def fix_unused_arguments(content: str) -> str:
    """Fix ARG001/ARG002: Prefix unused arguments with underscore."""
    # Common unused argument patterns
    patterns = [
        (r'def \w+\([^)]*\b(event|e|kwargs|args|request|sender)\b',
         lambda m: m.group(0).replace(m.group(1), f'_{m.group(1)}')),
        (r'lambda\s+(\w+):',
         lambda m: f'lambda _{m.group(1)}:' if 'e' in m.group(1) or 'event' in m.group(1) else m.group(0)),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    return content


def fix_semicolons(content: str) -> str:
    """Fix E702: Split statements with semicolons onto separate lines."""
    lines = content.split('\n')
    result = []

    for line in lines:
        if ';' in line and not line.strip().startswith('#'):
            # Check if it's a real semicolon separator
            if re.search(r'\s*;\s*\w+', line):
                parts = line.split(';')
                indent = len(line) - len(line.lstrip())
                result.append(parts[0])
                for part in parts[1:]:
                    result.append(' ' * indent + part.strip())
            else:
                result.append(line)
        else:
            result.append(line)

    return '\n'.join(result)


def fix_ambiguous_vars(content: str) -> str:
    """Fix E741: Rename ambiguous variable names."""
    # Replace single letter 'l' variable with '_l'
    content = re.sub(r'\b(l)\s*:', r'_\1:', content)
    content = re.sub(r'lambda\s+l\s*=', r'lambda _l=', content)
    content = re.sub(r'def\s+\w+\([^)]*\bl\b', lambda m: m.group(0).replace(' l', ' _l'), content)
    content = re.sub(r'for\s+(\w+,\s*)?l\b', r'for \1_l', content)

    return content


def fix_blank_whitespace(content: str) -> str:
    """Fix W293: Remove whitespace from blank lines."""
    lines = content.split('\n')
    result = []

    for line in lines:
        if line.strip() == '':
            result.append('')
        else:
            result.append(line)

    return '\n'.join(result)


def fix_unused_imports(content: str) -> str:
    """Fix F401: Remove unused imports."""
    # Specific unused imports from the errors
    unused = [
        'emailops.email_indexer',
        'emailops.text_chunker',
        'google.generativeai as genai',
    ]

    lines = content.split('\n')
    result = []

    for line in lines:
        skip = False
        for unused_import in unused:
            if unused_import in line and 'import' in line:
                skip = True
                break
        if not skip:
            result.append(line)

    return '\n'.join(result)


def fix_unicode_chars(content: str) -> str:
    """Fix RUF001/RUF002/RUF003: Replace ambiguous unicode characters."""
    replacements = {
        '×': 'x',  # Multiplication sign to x
        '–': '-',  # En dash to hyphen
        '—': '-',  # Em dash to hyphen
        ''': "'",  # Right single quote to apostrophe
        ''': "'",  # Left single quote to apostrophe
        '"': '"',  # Left double quote
        '"': '"',  # Right double quote
        'ℹ': 'i',  # Information source to i
        '‑': '-',  # Non-breaking hyphen to hyphen
    }

    for old, new in replacements.items():
        content = content.replace(old, new)

    return content


def fix_exception_handling(content: str) -> str:
    """Fix B904: Add 'from None' to exception raising in except blocks."""
    lines = content.split('\n')
    result = []
    in_except = False

    for line in lines:
        if 'except' in line and ':' in line:
            in_except = True
        elif line and len(line) > 0 and not line[0].isspace():
            in_except = False

        if in_except and line.strip().startswith('raise '):
            if 'from' not in line and not line.strip().endswith('from None'):
                line = line.rstrip() + ' from None'

        result.append(line)

    return '\n'.join(result)


def fix_unused_loop_vars(content: str) -> str:
    """Fix B007: Prefix unused loop variables with underscore."""
    content = re.sub(r'for\s+(\w+)\s*,', lambda m: f'for _{m.group(1)},' if m.group(1) in ['pos', 'dirs', 'value'] else m.group(0), content)
    return content


def fix_shadowing_builtins(content: str) -> str:
    """Fix A001: Rename variables that shadow Python builtins."""
    # Specific case: IndexError class
    content = re.sub(r'class IndexError\(', r'class EmailOpsIndexError(', content)
    content = content.replace('"IndexError"', '"EmailOpsIndexError"')
    return content


def fix_sort_all(content: str) -> str:
    """Fix RUF022: Sort __all__ lists."""
    pattern = r'__all__\s*=\s*\[(.*?)\]'

    def sort_all(match):
        items = match.group(1)
        # Extract items
        item_list = re.findall(r'"([^"]+)"', items) + re.findall(r"'([^']+)'", items)
        if item_list:
            sorted_items = sorted(item_list)
            formatted = '[\n    ' + ',\n    '.join(f'"{item}"' for item in sorted_items) + '\n]'
            return f'__all__ = {formatted}'
        return match.group(0)

    content = re.sub(pattern, sort_all, content, flags=re.DOTALL)
    return content


def fix_file(filepath: Path) -> bool:
    """Apply all fixes to a single file."""
    try:
        content = filepath.read_text(encoding='utf-8')
        original = content

        # Apply all fixes
        content = fix_imports_at_top(content)
        content = fix_blank_whitespace(content)
        content = fix_semicolons(content)
        content = fix_ambiguous_vars(content)
        content = fix_unused_imports(content)
        content = fix_unicode_chars(content)
        content = fix_exception_handling(content)
        content = fix_unused_loop_vars(content)

        # Special fixes for specific files
        if 'exceptions.py' in str(filepath):
            content = fix_shadowing_builtins(content)
            content = fix_sort_all(content)

        if 'emailops_gui.py' in str(filepath):
            # Fix specific lambda arguments
            content = content.replace('lambda e:', 'lambda _e:')
            content = content.replace('def _view_conversation_txt(self, event:',
                                    'def _view_conversation_txt(self, _event:')
            content = content.replace('def _save_settings(self, event:',
                                    'def _save_settings(self, _event:')
            content = content.replace('def _load_settings(self, event:',
                                    'def _load_settings(self, _event:')
            content = content.replace('def _show_snippet(self, event)',
                                    'def _show_snippet(self, _event)')
            content = content.replace('def _change_log_level(self, event:',
                                    'def _change_log_level(self, _event:')
            content = content.replace('lambda l=line:', 'lambda _l=line:')
            content = content.replace('def process_summarize(conv_id, **kwargs):',
                                    'def process_summarize(conv_id, **_kwargs):')
            content = content.replace('def process_reply(conv_id, output_dir, **kwargs):',
                                    'def process_reply(conv_id, output_dir, **_kwargs):')

        # Write back if changed
        if content != original:
            filepath.write_text(content, encoding='utf-8')
            return True
        return False

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    """Main function to fix all linting errors."""
    emailops_dir = Path('emailops')
    helpers_dir = Path('helpers & diagnostics')

    fixed_files = []

    # Process emailops directory
    if emailops_dir.exists():
        for py_file in emailops_dir.glob('*.py'):
            if fix_file(py_file):
                fixed_files.append(py_file)
                print(f"Fixed: {py_file}")

    # Process helpers directory
    if helpers_dir.exists():
        for py_file in helpers_dir.glob('*.py'):
            if fix_file(py_file):
                fixed_files.append(py_file)
                print(f"Fixed: {py_file}")

    print(f"\n{'='*50}")
    print(f"Total files fixed: {len(fixed_files)}")

    if fixed_files:
        print("\nRunning ruff check to verify fixes...")
        import subprocess
        result = subprocess.run(['ruff', 'check', 'emailops', '--statistics'],
                              capture_output=True, text=True)
        print(result.stdout)

    return len(fixed_files)


if __name__ == "__main__":
    sys.exit(0 if main() > 0 else 1)
