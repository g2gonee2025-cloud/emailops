#!/usr/bin/env python3
"""Check all Python files in the project for syntax errors"""

import os
import py_compile
import ast
from pathlib import Path
from typing import List, Tuple

def check_python_file(filepath: Path) -> Tuple[bool, str]:
    """Check a Python file for syntax errors"""
    try:
        # First try to compile
        py_compile.compile(str(filepath), doraise=True)
        
        # Then try to parse AST
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)

def main():
    """Check all Python files in the project"""
    root = Path('.')
    
    # Files to check
    python_files = []
    
    # Root directory Python files
    for file in root.glob('*.py'):
        if not file.name.startswith('_'):
            python_files.append(file)
    
    # Emailops module files
    emailops_dir = root / 'emailops'
    if emailops_dir.exists():
        for file in emailops_dir.glob('*.py'):
            python_files.append(file)
    
    print("=" * 80)
    print("PYTHON CODE QUALITY CHECK")
    print("=" * 80)
    
    errors = []
    warnings = []
    
    for filepath in sorted(python_files):
        rel_path = filepath.relative_to(root)
        success, message = check_python_file(filepath)
        
        if not success:
            errors.append((rel_path, message))
            print(f"❌ {rel_path}: {message}")
        else:
            print(f"✅ {rel_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Files checked: {len(python_files)}")
    print(f"Errors found: {len(errors)}")
    
    if errors:
        print("\n⚠️ ERRORS FOUND:")
        for filepath, error in errors:
            print(f"  - {filepath}: {error}")
        return 1
    else:
        print("\n✅ All files passed syntax check!")
        return 0

if __name__ == "__main__":
    exit(main())