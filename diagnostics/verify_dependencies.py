#!/usr/bin/env python3
"""Verify key Python dependencies for emailops_vertex_ai project."""

import json
import sys
from datetime import datetime
from pathlib import Path


def check_import(module_name):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        return True, None
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {e}"

def get_version(module_name):
    """Get the version of an installed module."""
    try:
        module = __import__(module_name)
        if hasattr(module, '__version__'):
            return module.__version__
        elif hasattr(module, 'version'):
            return module.version
        elif hasattr(module, 'VERSION'):
            return module.VERSION
        else:
            return "Version unknown"
    except Exception:
        return "N/A"

# List of critical dependencies to check
dependencies = [
    # Google Cloud dependencies
    'google.cloud.aiplatform',
    'google.auth',
    'google.api_core',

    # Core dependencies
    'streamlit',
    'qdrant_client',
    'pandas',
    'numpy',
    'pydantic',
    'langchain',
    'openai',

    # Other important packages
    'requests',
    'pytest',
    'python-dotenv',
    'yaml',
    'tiktoken',
    'tqdm',
]

# Additional module-specific checks
module_checks = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'streamlit': 'streamlit',
    'qdrant_client': 'qdrant_client',
    'pydantic': 'pydantic',
    'dotenv': 'dotenv',
    'yaml': 'yaml',
    'tiktoken': 'tiktoken',
    'tqdm': 'tqdm',
    'pytest': 'pytest',
    'requests': 'requests',
    'openai': 'openai',
}

print("=" * 60)
print("Python Dependencies Verification Report")
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Python Version: {sys.version}")
print("=" * 60)
print()

# Check each dependency
results = {}
failed_imports = []

print("Checking critical dependencies...")
print("-" * 60)

for dep in dependencies:
    success, error = check_import(dep)
    if success:
        print(f"✓ {dep:<30} - OK")
        results[dep] = {"status": "OK", "error": None}
    else:
        print(f"✗ {dep:<30} - FAILED: {error}")
        results[dep] = {"status": "FAILED", "error": error}
        failed_imports.append(dep)

print()
print("Checking module versions...")
print("-" * 60)

for module_name, import_name in module_checks.items():
    success, _ = check_import(import_name)
    if success:
        version = get_version(import_name)
        print(f"{module_name:<20} - Version: {version}")

# Summary
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total dependencies checked: {len(dependencies)}")
print(f"Successful imports: {len(dependencies) - len(failed_imports)}")
print(f"Failed imports: {len(failed_imports)}")

if failed_imports:
    print("\nFailed imports:")
    for dep in failed_imports:
        print(f"  - {dep}: {results[dep]['error']}")
else:
    print("\n✓ All critical dependencies are properly installed!")

# Save results to JSON
with Path('dependency_verification_results.json').open('w') as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "results": results,
        "summary": {
            "total_checked": len(dependencies),
            "successful": len(dependencies) - len(failed_imports),
            "failed": len(failed_imports)
        }
    }, f, indent=2)

print("\nResults saved to: dependency_verification_results.json")
