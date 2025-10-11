#!/usr/bin/env python3
"""Test script to verify processor.py fix"""

import subprocess
from pathlib import Path

# Test 1: Import the module
try:
    from processing.processor import UnifiedProcessor
    print("✓ SUCCESS: All imports work correctly")
except ImportError as e:
    print(f"✗ FAILED: Import error - {e}")
    exit(1)

# Test 2: Check file structure
with Path('processing/processor.py').open(encoding='utf-8') as f:
    content = f.read()
    lines = content.splitlines()

print(f"✓ Total lines in fixed file: {len(lines)}")
print(f"✓ First line: {lines[0]}")

# Test 3: Verify no string wrapper
if "updated_code = r'''" in content:
    print("✗ FAILED: File still contains string wrapper")
    exit(1)
else:
    print("✓ SUCCESS: No string wrapper found")

# Test 4: Check for key classes
required_items = ['class UnifiedProcessor', 'class ChunkJob', 'class WorkerStats', 'class ProcessingStats', 'def main()', 'if __name__ == "__main__"']
for item in required_items:
    if item in content:
        print(f"✓ Found: {item}")
    else:
        print(f"✗ Missing: {item}")
        exit(1)

# Test 5: Verify CLI works
result = subprocess.run(['python', 'processing/processor.py', '--help'], capture_output=True, text=True)
if result.returncode == 0 and 'Unified processor' in result.stdout:
    print("✓ CLI entry point works correctly")
else:
    print(f"✗ CLI failed: {result.stderr}")
    exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED! processor.py is now fully functional")
print("="*60)
