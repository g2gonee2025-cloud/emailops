#!/usr/bin/env python3
"""
Check for unused code and functions in the emailops project.
Uses multiple tools to find dead code, unused imports, and unused functions.
"""

import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


def install_tools():
    """Install required tools for dead code detection."""
    tools = ['vulture', 'pyflakes']

    print("Installing required tools...")
    for tool in tools:
        subprocess.run([sys.executable, '-m', 'pip', 'install', tool],
                      capture_output=True)
    print("Tools installed.\n")


def run_vulture(directory: str = "emailops") -> Dict[str, List[str]]:
    """
    Run vulture to find unused code.
    Returns dict with categories of unused items.
    """
    print("=" * 60)
    print("Running Vulture - Dead Code Detector")
    print("=" * 60)

    result = subprocess.run(
        ['vulture', directory, '--min-confidence', '80'],
        capture_output=True,
        text=True
    )

    output = result.stdout

    # Parse vulture output
    unused = {
        'unused_functions': [],
        'unused_variables': [],
        'unused_classes': [],
        'unused_imports': [],
        'unused_attributes': [],
        'unreachable_code': []
    }

    for line in output.split('\n'):
        if not line.strip():
            continue

        if 'unused function' in line:
            unused['unused_functions'].append(line)
        elif 'unused variable' in line:
            unused['unused_variables'].append(line)
        elif 'unused class' in line:
            unused['unused_classes'].append(line)
        elif 'unused import' in line:
            unused['unused_imports'].append(line)
        elif 'unused attribute' in line:
            unused['unused_attributes'].append(line)
        elif 'unreachable code' in line:
            unused['unreachable_code'].append(line)

    # Print summary
    print(f"Unused functions: {len(unused['unused_functions'])}")
    print(f"Unused variables: {len(unused['unused_variables'])}")
    print(f"Unused classes: {len(unused['unused_classes'])}")
    print(f"Unused imports: {len(unused['unused_imports'])}")
    print(f"Unused attributes: {len(unused['unused_attributes'])}")
    print(f"Unreachable code: {len(unused['unreachable_code'])}")

    # Print details
    if unused['unused_functions']:
        print("\n--- Top 10 Unused Functions ---")
        for item in unused['unused_functions'][:10]:
            print(f"  {item}")

    if unused['unused_classes']:
        print("\n--- Unused Classes ---")
        for item in unused['unused_classes']:
            print(f"  {item}")

    return unused


def run_pyflakes(directory: str = "emailops") -> List[str]:
    """
    Run pyflakes to find unused imports and undefined names.
    """
    print("\n" + "=" * 60)
    print("Running Pyflakes - Unused Imports & Undefined Names")
    print("=" * 60)

    result = subprocess.run(
        ['pyflakes', directory],
        capture_output=True,
        text=True
    )

    issues = result.stdout.split('\n')
    issues = [i for i in issues if i.strip()]

    unused_imports = []
    undefined_names = []
    redefined = []

    for issue in issues:
        if 'imported but unused' in issue:
            unused_imports.append(issue)
        elif 'undefined name' in issue:
            undefined_names.append(issue)
        elif 'redefinition' in issue:
            redefined.append(issue)

    print(f"Unused imports: {len(unused_imports)}")
    print(f"Undefined names: {len(undefined_names)}")
    print(f"Redefinitions: {len(redefined)}")

    if unused_imports:
        print("\n--- Sample Unused Imports (first 10) ---")
        for item in unused_imports[:10]:
            print(f"  {item}")

    return issues


def run_ruff_unused(directory: str = "emailops") -> Dict[str, int]:
    """
    Run ruff to check for unused code patterns.
    """
    print("\n" + "=" * 60)
    print("Running Ruff - Unused Code Analysis")
    print("=" * 60)

    # Check for specific unused code patterns
    patterns = [
        'F401',  # unused imports
        'F841',  # unused variables
        'F821',  # undefined names
        'F811',  # redefinition of unused
    ]

    result = subprocess.run(
        ['ruff', 'check', directory, '--select', ','.join(patterns), '--output-format', 'json'],
        capture_output=True,
        text=True
    )

    try:
        issues = json.loads(result.stdout) if result.stdout else []
    except json.JSONDecodeError:
        issues = []

    # Count by error code
    counts = {}
    for issue in issues:
        code = issue.get('code', 'unknown')
        counts[code] = counts.get(code, 0) + 1

    print(f"F401 - Unused imports: {counts.get('F401', 0)}")
    print(f"F841 - Unused variables: {counts.get('F841', 0)}")
    print(f"F821 - Undefined names: {counts.get('F821', 0)}")
    print(f"F811 - Redefinition of unused: {counts.get('F811', 0)}")

    # Show examples
    if issues:
        print("\n--- Sample Issues ---")
        for issue in issues[:5]:
            filename = issue.get('filename', '')
            line = issue.get('location', {}).get('row', 0)
            message = issue.get('message', '')
            print(f"  {filename}:{line} - {message}")

    return counts


def analyze_function_usage(directory: str = "emailops") -> Dict[str, List[Dict]]:
    """
    Analyze which functions are defined but never called.
    Returns dict with function names as keys and list of file info as values.
    """
    print("\n" + "=" * 60)
    print("Analyzing Function Usage")
    print("=" * 60)

    import ast

    defined_functions = {}
    called_functions = set()

    # Collect all Python files
    py_files = list(Path(directory).rglob("*.py"))

    for filepath in py_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)

            # Find function definitions with line numbers
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    if func_name not in defined_functions:
                        defined_functions[func_name] = []

                    # Store file path and line number
                    defined_functions[func_name].append({
                        'path': str(filepath),
                        'line': node.lineno
                    })

                # Find function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        called_functions.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        called_functions.add(node.func.attr)
        except Exception:
            continue

    # Find unused functions (not called anywhere)
    unused_functions = {}
    for func_name, locations in defined_functions.items():
        if func_name not in called_functions:
            # Skip special methods and test functions
            if not (func_name.startswith('__') or
                   func_name.startswith('test_') or
                   func_name in ['main', 'setUp', 'tearDown']):
                unused_functions[func_name] = locations

    print(f"Total functions defined: {len(defined_functions)}")
    print(f"Functions that appear to be called: {len(called_functions)}")
    print(f"Potentially unused functions: {len(unused_functions)}")

    if unused_functions:
        print("\n--- Potentially Unused Functions ---")
        for func_name, locations in list(unused_functions.items())[:20]:
            for loc in locations:
                file_name = Path(loc['path']).name
                print(f"  {func_name} in {file_name}:{loc['line']}")

    return unused_functions


def check_import_usage(directory: str = "emailops") -> Dict[str, List[str]]:
    """
    Check which modules are imported but never used.
    """
    print("\n" + "=" * 60)
    print("Checking Import Usage")
    print("=" * 60)

    import ast

    unused_per_file = {}

    py_files = list(Path(directory).rglob("*.py"))

    for filepath in py_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)

            imports = set()
            used_names = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        imports.add(name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        imports.add(name)
                elif isinstance(node, ast.Name):
                    used_names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name):
                        used_names.add(node.value.id)

            unused = imports - used_names
            if unused:
                unused_per_file[str(filepath)] = list(unused)

        except Exception:
            continue

    total_unused = sum(len(v) for v in unused_per_file.values())
    print(f"Files with unused imports: {len(unused_per_file)}")
    print(f"Total unused imports: {total_unused}")

    if unused_per_file:
        print("\n--- Files with Unused Imports ---")
        for filepath, unused in list(unused_per_file.items())[:10]:
            filename = Path(filepath).name
            print(f"  {filename}: {', '.join(unused)}")

    return unused_per_file


def generate_report(directory: str = "emailops", format: str = "markdown"):
    """
    Generate a comprehensive unused code report.

    Args:
        directory: Directory to analyze
        format: Output format ('markdown' or 'html')
    """
    print("\n" + "=" * 60)
    print("UNUSED CODE ANALYSIS REPORT")
    print("=" * 60)

    import os

    # Save results to file
    report = []

    # Run all analyses
    vulture_results = run_vulture(directory)
    pyflakes_results = run_pyflakes(directory)
    ruff_results = run_ruff_unused(directory)
    unused_functions = analyze_function_usage(directory)
    unused_imports = check_import_usage(directory)

    # Get absolute path for VSCode links
    abs_dir = Path(directory).resolve()

    if format == "html":
        # Generate HTML report with clickable links
        report.append("<!DOCTYPE html>\n<html>\n<head>\n")
        report.append("<title>Unused Code Analysis Report</title>\n")
        report.append("<style>\n")
        report.append("body { font-family: Arial, sans-serif; margin: 20px; }\n")
        report.append("h1 { color: #333; }\n")
        report.append("h2 { color: #555; }\n")
        report.append("a { color: #0066cc; text-decoration: none; }\n")
        report.append("a:hover { text-decoration: underline; }\n")
        report.append(".function-link { margin: 5px 0; padding: 5px; background: #f5f5f5; }\n")
        report.append(".file-path { color: #666; font-size: 0.9em; }\n")
        report.append("</style>\n</head>\n<body>\n")
        report.append("<h1>Unused Code Analysis Report</h1>\n")
        report.append(f"<p>Directory analyzed: <code>{directory}</code></p>\n")

        # Summary
        report.append("<h2>Summary</h2>\n")
        report.append("<ul>\n")
        report.append(f"<li>Unused functions: {len(unused_functions)}</li>\n")
        report.append(f"<li>Files with unused imports: {len(unused_imports)}</li>\n")
        report.append(f"<li>Unused variables (vulture): {len(vulture_results['unused_variables'])}</li>\n")
        report.append(f"<li>Unused classes: {len(vulture_results['unused_classes'])}</li>\n")
        report.append(f"<li>Unreachable code blocks: {len(vulture_results['unreachable_code'])}</li>\n")
        report.append("</ul>\n")

        # Unused functions with VSCode links
        if unused_functions:
            report.append("<h2>Potentially Unused Functions</h2>\n")
            report.append("<p>Click on any function to open it in VSCode:</p>\n")
            report.append("<div>\n")
            for func_name, locations in sorted(unused_functions.items()):
                for loc in locations:
                    abs_path = Path(loc['path']).resolve()
                    vscode_link = f"vscode://file/{abs_path}:{loc['line']}:1"
                    file_name = Path(loc['path']).name
                    report.append(f'<div class="function-link">\n')
                    report.append(f'  <a href="{vscode_link}"><strong>{func_name}()</strong></a>\n')
                    report.append(f'  <span class="file-path">in {file_name}:{loc["line"]}</span>\n')
                    report.append(f'</div>\n')
            report.append("</div>\n")

        report.append("</body>\n</html>\n")

        # Save HTML report
        report_path = Path("UNUSED_CODE_REPORT.html")

    else:  # Markdown format
        report.append("# Unused Code Analysis Report\n")
        report.append(f"Directory analyzed: `{directory}`\n\n")

        # Summary
        report.append("## Summary\n")
        report.append(f"- Unused functions: {len(unused_functions)}\n")
        report.append(f"- Files with unused imports: {len(unused_imports)}\n")
        report.append(f"- Unused variables (vulture): {len(vulture_results['unused_variables'])}\n")
        report.append(f"- Unused classes: {len(vulture_results['unused_classes'])}\n")
        report.append(f"- Unreachable code blocks: {len(vulture_results['unreachable_code'])}\n")

        # Unused functions with VSCode links
        if unused_functions:
            report.append("\n## Potentially Unused Functions\n")
            report.append("*Click on any link to open in VSCode:*\n\n")
            for func_name, locations in sorted(unused_functions.items()):
                for loc in locations:
                    abs_path = Path(loc['path']).resolve()
                    file_name = Path(loc['path']).name
                    # Create VSCode link for markdown
                    vscode_link = f"vscode://file/{abs_path}:{loc['line']}:1"
                    report.append(f"- **[`{func_name}()`]({vscode_link})** in {file_name}:{loc['line']}\n")

        if unused_imports:
            report.append("\n## Unused Imports by File\n")
            for filepath, imports in unused_imports.items():
                abs_path = Path(filepath).resolve()
                vscode_link = f"vscode://file/{abs_path}:1:1"
                report.append(f"- **[{Path(filepath).name}]({vscode_link})**: {', '.join(imports)}\n")

        # Save Markdown report
        report_path = Path("UNUSED_CODE_REPORT.md")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report)

    print(f"\n✅ Report saved to {report_path}")

    # Also generate HTML report if markdown was requested
    if format == "markdown":
        generate_report(directory, "html")
        print(f"✅ HTML report with clickable links saved to UNUSED_CODE_REPORT.html")

    # Suggest cleanup commands
    print("\n" + "=" * 60)
    print("SUGGESTED CLEANUP COMMANDS")
    print("=" * 60)
    print("1. Remove unused imports automatically:")
    print("   ruff check --fix --select F401 emailops")
    print("\n2. Remove unused variables automatically:")
    print("   ruff check --fix --select F841 emailops")
    print("\n3. Review vulture findings with lower confidence:")
    print("   vulture emailops --min-confidence 60")
    print("\n4. Generate detailed dead code report:")
    print("   vulture emailops --min-confidence 70 > dead_code.txt")


def main():
    """Main function to run all unused code checks."""
    import argparse

    parser = argparse.ArgumentParser(description='Check for unused code in Python project')
    parser.add_argument('directory', nargs='?', default='emailops',
                       help='Directory to analyze (default: emailops)')
    parser.add_argument('--install', action='store_true',
                       help='Install required tools first')

    args = parser.parse_args()

    if args.install:
        install_tools()

    try:
        generate_report(args.directory)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Some tools may not be installed. Run with --install flag:")
        print("  python check_unused_code.py --install")
        sys.exit(1)


if __name__ == "__main__":
    main()
