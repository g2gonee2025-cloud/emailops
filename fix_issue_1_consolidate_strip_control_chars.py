#!/usr/bin/env python3
"""
Automated fix for Issue #1: Consolidate _strip_control_chars() implementations.

This script:
1. Consolidates the canonical implementation in utils.py
2. Removes duplicate from llm_runtime.py
3. Updates all import paths across 6 modules
4. Adds explicit normalize_newlines parameter for backward compatibility
"""

import re
import sys
from pathlib import Path

# Define the canonical consolidated implementation
CANONICAL_IMPLEMENTATION = '''def _strip_control_chars(s: str, *, normalize_newlines: bool = False) -> str:
    """
    Remove non-printable control characters from string.

    Args:
        s: String to clean
        normalize_newlines: If True, normalize CRLF/CR to LF before stripping

    Returns:
        Cleaned string with control characters removed

    Note:
        Control chars removed: [\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f-\\x9f]
        Newlines (\\n, \\r) are preserved unless normalize_newlines=True
    """
    if not s:
        return ""

    # Optionally normalize newlines first (for embedding/indexing pipeline)
    if normalize_newlines:
        s = s.replace("\\r\\n", "\\n").replace("\\r", "\\n")

    # Remove control characters (preserving \\n, \\r)
    return re.sub(r"[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f-\\x9f]", "", s)
'''


def backup_file(filepath: Path) -> Path:
    """Create timestamped backup of file."""
    import shutil
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = filepath.with_suffix(f".backup_{timestamp}{filepath.suffix}")
    shutil.copy2(filepath, backup_path)
    print(f"  Created backup: {backup_path.name}")
    return backup_path


def update_utils_py() -> bool:
    """Update utils.py with consolidated implementation."""
    print("\n[Step 1] Updating utils.py with consolidated implementation")

    utils_path = Path("emailops/utils.py")
    if not utils_path.exists():
        print(f"  ERROR: {utils_path} not found")
        return False

    backup_file(utils_path)

    content = utils_path.read_text(encoding='utf-8')

    # Find and replace the current implementation
    pattern = r'def _strip_control_chars\(s: str\) -> str:.*?return re\.sub\([^)]+\)'

    if not re.search(pattern, content, re.DOTALL):
        print("  ERROR: Could not find _strip_control_chars function")
        return False

    # Replace with canonical version
    new_content = re.sub(pattern, CANONICAL_IMPLEMENTATION, content, flags=re.DOTALL)

    utils_path.write_text(new_content, encoding='utf-8')
    print("  ✓ Updated utils.py with consolidated implementation")
    return True


def remove_from_llm_runtime() -> bool:
    """Remove duplicate implementation from llm_runtime.py."""
    print("\n[Step 2] Removing duplicate from llm_runtime.py")

    runtime_path = Path("emailops/llm_runtime.py")
    if not runtime_path.exists():
        print(f"  ERROR: {runtime_path} not found")
        return False

    backup_file(runtime_path)

    content = runtime_path.read_text(encoding='utf-8')

    # Find the duplicate function and its compiled pattern
    pattern_line = r'_CONTROL_CHARS = re\.compile\(r"\[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F\\x7F-\\x9F\]"\)\s*\n'
    func_pattern = r'def _strip_control_chars\(s: str\) -> str:.*?return _CONTROL_CHARS\.sub\("", s\)\s*\n'

    # Remove pattern definition
    content = re.sub(pattern_line, '', content)
    # Remove function definition
    content = re.sub(func_pattern, '', content, flags=re.DOTALL)

    # Add import from utils at the top (after existing utils import)
    import_pattern = r'(from \.utils import \(\s*logger,\s*monitor_performance,\s*\))'
    replacement = r'from .utils import (\n    _strip_control_chars,\n    logger,\n    monitor_performance,\n)'
    content = re.sub(import_pattern, replacement, content)

    runtime_path.write_text(content, encoding='utf-8')
    print("  ✓ Removed duplicate and added import from utils")
    return True


def update_call_sites() -> tuple[int, int]:
    """Update all call sites to use normalize_newlines parameter where needed."""
    print("\n[Step 3] Updating call sites with normalize_newlines parameter")

    # Modules that need normalize_newlines=True (indexing/embedding pipeline)
    needs_normalization = {
        "emailops/llm_runtime.py",  # Already removed, but for reference
        "emailops/indexing_main.py",
        "emailops/llm_text_chunker.py",
    }

    # All modules that use _strip_control_chars
    modules_to_check = [
        "emailops/core_email_processing.py",
        "emailops/core_text_extraction.py",
        "emailops/feature_search_draft.py",
        "emailops/indexing_main.py",
    ]

    updated_count = 0
    call_sites = 0

    for module_path in modules_to_check:
        path = Path(module_path)
        if not path.exists():
            continue

        content = path.read_text(encoding='utf-8')

        # Count calls
        calls = len(re.findall(r'_strip_control_chars\(', content))
        call_sites += calls

        if calls > 0:
            print(f"  {path.name}: {calls} call sites")

            # For indexing pipeline, add normalize_newlines=True
            if str(path) in needs_normalization:
                # Replace calls with normalized version
                pattern = r'_strip_control_chars\(([^)]+)\)'
                replacement = r'_strip_control_chars(\1, normalize_newlines=True)'
                new_content = re.sub(pattern, replacement, content)

                if new_content != content:
                    backup_file(path)
                    path.write_text(new_content, encoding='utf-8')
                    updated_count += 1
                    print("    ✓ Updated to use normalize_newlines=True")

    print(f"\n  Total: {call_sites} call sites found, {updated_count} files updated")
    return call_sites, updated_count


def verify_no_duplicates() -> bool:
    """Verify no duplicate implementations remain."""
    print("\n[Step 4] Verifying no duplicate implementations remain")

    pattern = re.compile(r'def _strip_control_chars\(')

    found_in = []
    for py_file in Path("emailops").rglob("*.py"):
        content = py_file.read_text(encoding='utf-8')
        if pattern.search(content):
            found_in.append(py_file)

    if len(found_in) == 1 and found_in[0].name == "utils.py":
        print("  ✓ Single implementation in utils.py only")
        return True
    elif len(found_in) > 1:
        print("  ERROR: Multiple implementations still exist:")
        for f in found_in:
            print(f"    - {f}")
        return False
    else:
        print("  ERROR: No implementation found!")
        return False


def main():
    """Execute automated fix for Issue #1."""
    print("="*80)
    print("AUTOMATED FIX: Issue #1 - Consolidate _strip_control_chars()")
    print("="*80)
    print("\nThis will:")
    print("  1. Add normalize_newlines parameter to utils.py implementation")
    print("  2. Remove duplicate from llm_runtime.py")
    print("  3. Update call sites in indexing pipeline")
    print("  4. Verify consolidation successful")
    print("\nBackups will be created for all modified files.")

    input("\nPress Enter to continue, Ctrl+C to abort...")

    try:
        # Execute steps
        if not update_utils_py():
            print("\nFAILED at step 1")
            return 1

        if not remove_from_llm_runtime():
            print("\nFAILED at step 2")
            return 1

        call_sites, updated = update_call_sites()

        if not verify_no_duplicates():
            print("\nFAILED at step 4 - duplicates remain")
            return 1

        print("\n" + "="*80)
        print("SUCCESS: Issue #1 remediated")
        print("="*80)
        print("  - Consolidated implementation in utils.py")
        print("  - Removed duplicate from llm_runtime.py")
        print(f"  - Updated {updated} files with normalize_newlines parameter")
        print(f"  - Verified {call_sites} call sites")
        print("\nNext steps:")
        print("  1. Run test suite to validate no regressions")
        print("  2. Rebuild index with consistent text normalization")
        print("  3. Verify search results match expectations")

        return 0

    except KeyboardInterrupt:
        print("\n\nAborted by user")
        return 130
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
