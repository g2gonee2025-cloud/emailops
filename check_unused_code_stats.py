#!/usr/bin/env python3
"""
Enhanced code analysis tool that provides detailed statistics on function usage.
Shows call counts, caller information, and dependency relationships.
"""

import ast
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


def install_tools():
    """Install required tools for dead code detection."""
    tools = ['vulture', 'pyflakes']

    print("Installing required tools...")
    for tool in tools:
        subprocess.run([sys.executable, '-m', 'pip', 'install', tool],
                      capture_output=True)
    print("Tools installed.\n")


def analyze_function_usage_detailed(directory: str = "emailops") -> dict:
    """
    Analyze function definitions and calls with detailed statistics.
    Returns comprehensive usage data including call counts and caller information.
    """
    print("\n" + "=" * 60)
    print("Analyzing Detailed Function Usage Statistics")
    print("=" * 60)

    # Data structures to track function usage
    function_definitions = {}  # func_name -> [{file, line, class}]
    function_calls = defaultdict(list)  # func_name -> [{caller_file, caller_func, line}]
    call_counts = Counter()  # func_name -> count
    file_dependencies = defaultdict(set)  # file -> set of files it depends on

    # Collect all Python files
    py_files = list(Path(directory).rglob("*.py"))

    for filepath in py_files:
        try:
            with Path.open(filepath, encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)

            file_name = str(filepath)
            current_class = None
            current_function = None

            # Use a visitor to track context
            class FunctionVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.current_class = None
                    self.current_function = None
                    self.function_stack = []

                def visit_ClassDef(self, node):
                    old_class = self.current_class
                    self.current_class = node.name
                    self.generic_visit(node)
                    self.current_class = old_class

                def visit_FunctionDef(self, node):
                    # Record function definition
                    func_name = node.name
                    if func_name not in function_definitions:
                        function_definitions[func_name] = []

                    function_definitions[func_name].append({
                        'file': file_name,
                        'line': node.lineno,
                        'class': self.current_class,
                        'full_name': f"{self.current_class}.{func_name}" if self.current_class else func_name
                    })

                    # Track current function context for calls
                    old_function = self.current_function
                    self.current_function = func_name
                    self.function_stack.append(func_name)
                    self.generic_visit(node)
                    self.function_stack.pop()
                    self.current_function = old_function

                def visit_Call(self, node):
                    # Track function calls
                    func_name = None

                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr

                    if func_name:
                        call_counts[func_name] += 1
                        function_calls[func_name].append({
                            'caller_file': file_name,
                            'caller_function': self.current_function or '<module>',
                            'caller_class': self.current_class,
                            'line': node.lineno
                        })

                    self.generic_visit(node)

            visitor = FunctionVisitor()
            visitor.visit(tree)

        except Exception as e:
            print(f"  Error processing {filepath}: {e}")
            continue

    # Analyze the collected data
    total_functions = len(function_definitions)
    called_functions = set(call_counts.keys())
    unused_functions = set(function_definitions.keys()) - called_functions

    # Filter out special methods and test functions from unused
    unused_functions = {
        func for func in unused_functions
        if not (func.startswith('__') or
               func.startswith('test_') or
               func in ['main', 'setUp', 'tearDown'])
    }

    print("\n📊 Overall Statistics:")
    print(f"  Total functions defined: {total_functions}")
    print(f"  Functions with calls: {len(called_functions)}")
    print(f"  Potentially unused functions: {len(unused_functions)}")
    print(f"  Total function calls tracked: {sum(call_counts.values())}")

    # Most called functions
    most_called = call_counts.most_common(10)
    if most_called:
        print("\n🔥 Top 10 Most Called Functions:")
        for func_name, count in most_called:
            print(f"  {func_name:30} - {count:5} calls")

    # Functions with most diverse callers
    caller_diversity = {}
    for func_name, calls in function_calls.items():
        unique_files = set(call['caller_file'] for call in calls)
        unique_functions = set(call['caller_function'] for call in calls)
        caller_diversity[func_name] = {
            'file_count': len(unique_files),
            'function_count': len(unique_functions),
            'total_calls': len(calls)
        }

    most_diverse = sorted(
        caller_diversity.items(),
        key=lambda x: (x[1]['file_count'], x[1]['function_count']),
        reverse=True
    )[:10]

    if most_diverse:
        print("\n🌐 Functions Called from Most Places:")
        for func_name, stats in most_diverse:
            print(f"  {func_name:30} - {stats['file_count']} files, {stats['function_count']} functions")

    return {
        'definitions': function_definitions,
        'calls': dict(function_calls),
        'call_counts': dict(call_counts),
        'unused': list(unused_functions),
        'caller_diversity': caller_diversity
    }


def generate_detailed_report(directory: str = "emailops"):
    """
    Generate a comprehensive report with detailed function usage statistics.
    """
    print("\n" + "=" * 60)
    print("DETAILED FUNCTION USAGE ANALYSIS")
    print("=" * 60)

    # Analyze function usage
    usage_data = analyze_function_usage_detailed(directory)

    # Generate HTML report with detailed statistics
    report = []
    report.append("<!DOCTYPE html>\n<html>\n<head>\n")
    report.append("<title>Function Usage Statistics Report</title>\n")
    report.append("<style>\n")
    report.append("""
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; }
        h3 { color: #7f8c8d; }
        a { color: #3498db; text-decoration: none; }
        a:hover { text-decoration: underline; color: #2980b9; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .stat-card { background: #ecf0f1; padding: 15px; border-radius: 5px; text-align: center; }
        .stat-number { font-size: 2em; font-weight: bold; color: #2c3e50; }
        .stat-label { color: #7f8c8d; margin-top: 5px; }
        .function-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .function-table th { background: #3498db; color: white; padding: 10px; text-align: left; }
        .function-table td { padding: 8px; border-bottom: 1px solid #ecf0f1; }
        .function-table tr:hover { background: #f8f9fa; }
        .call-count { background: #3498db; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.9em; display: inline-block; }
        .unused { background: #e74c3c; }
        .rarely-used { background: #f39c12; }
        .frequently-used { background: #27ae60; }
        .caller-list { font-size: 0.9em; color: #7f8c8d; }
        .file-link { color: #3498db; }
        .tabs { display: flex; border-bottom: 2px solid #ecf0f1; margin: 20px 0; }
        .tab { padding: 10px 20px; cursor: pointer; background: #ecf0f1; margin-right: 5px; border-radius: 5px 5px 0 0; }
        .tab.active { background: #3498db; color: white; }
        .tab-content { display: none; padding: 20px 0; }
        .tab-content.active { display: block; }
        .dependency-graph { background: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .search-box { padding: 10px; width: 100%; border: 1px solid #ddd; border-radius: 5px; margin: 10px 0; }
    </style>
    <script>
        function showTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById(tabName + '-tab').classList.add('active');
            document.getElementById(tabName + '-content').classList.add('active');
        }
        function filterTable(inputId, tableId) {
            const input = document.getElementById(inputId).value.toLowerCase();
            const table = document.getElementById(tableId);
            const rows = table.getElementsByTagName('tr');
            for (let i = 1; i < rows.length; i++) {
                const text = rows[i].textContent.toLowerCase();
                rows[i].style.display = text.includes(input) ? '' : 'none';
            }
        }
    </script>
    </head>\n<body>\n""")

    report.append("<div class='container'>\n")
    report.append("<h1>📊 Function Usage Statistics Report</h1>\n")
    report.append(f"<p>Analysis of: <code>{Path(directory).resolve()}</code></p>\n")

    # Summary statistics cards
    total_functions = len(usage_data['definitions'])
    total_calls = sum(usage_data['call_counts'].values())
    unused_count = len(usage_data['unused'])
    used_count = len(usage_data['call_counts'])

    report.append("<div class='stats-grid'>\n")
    report.append(f"""
        <div class='stat-card'>
            <div class='stat-number'>{total_functions}</div>
            <div class='stat-label'>Total Functions</div>
        </div>
        <div class='stat-card'>
            <div class='stat-number'>{used_count}</div>
            <div class='stat-label'>Used Functions</div>
        </div>
        <div class='stat-card'>
            <div class='stat-number'>{unused_count}</div>
            <div class='stat-label'>Unused Functions</div>
        </div>
        <div class='stat-card'>
            <div class='stat-number'>{total_calls}</div>
            <div class='stat-label'>Total Function Calls</div>
        </div>
    """)
    report.append("</div>\n")

    # Tabs for different views
    report.append("""
        <div class='tabs'>
            <div id='all-tab' class='tab active' onclick='showTab("all")'>All Functions</div>
            <div id='unused-tab' class='tab' onclick='showTab("unused")'>Unused Functions</div>
            <div id='popular-tab' class='tab' onclick='showTab("popular")'>Most Used</div>
            <div id='callers-tab' class='tab' onclick='showTab("callers")'>Call Details</div>
        </div>
    """)

    # All Functions Tab
    report.append("<div id='all-content' class='tab-content active'>\n")
    report.append("<h2>All Functions with Usage Statistics</h2>\n")
    report.append("<input type='text' id='all-search' class='search-box' placeholder='Search functions...' onkeyup='filterTable(\"all-search\", \"all-table\")'>\n")
    report.append("<table id='all-table' class='function-table'>\n")
    report.append("<tr><th>Function</th><th>File</th><th>Line</th><th>Call Count</th><th>Called By</th></tr>\n")

    for func_name, definitions in sorted(usage_data['definitions'].items()):
        for defn in definitions:
            file_name = Path(defn['file']).name
            line = defn['line']
            abs_path = Path(defn['file']).resolve()
            vscode_link = f"vscode://file/{abs_path}:{line}:1"

            call_count = usage_data['call_counts'].get(func_name, 0)

            # Determine usage level
            if call_count == 0:
                count_class = 'unused'
            elif call_count < 3:
                count_class = 'rarely-used'
            else:
                count_class = 'frequently-used'

            # Get unique callers
            calls = usage_data['calls'].get(func_name, [])
            unique_callers = set()
            for call in calls:
                caller = f"{Path(call['caller_file']).name}:{call['caller_function']}"
                unique_callers.add(caller)

            callers_summary = ', '.join(list(unique_callers)[:3])
            if len(unique_callers) > 3:
                callers_summary += f" (+{len(unique_callers)-3} more)"

            report.append(f"""
                <tr>
                    <td><a href='{vscode_link}'><strong>{func_name}</strong></a></td>
                    <td class='file-link'>{file_name}</td>
                    <td>{line}</td>
                    <td><span class='call-count {count_class}'>{call_count}</span></td>
                    <td class='caller-list'>{callers_summary if callers_summary else 'None'}</td>
                </tr>
            """)

    report.append("</table>\n")
    report.append("</div>\n")

    # Unused Functions Tab
    report.append("<div id='unused-content' class='tab-content'>\n")
    report.append("<h2>🚫 Potentially Unused Functions</h2>\n")
    report.append("<p>These functions appear to have no calls within the analyzed codebase:</p>\n")
    report.append("<table class='function-table'>\n")
    report.append("<tr><th>Function</th><th>File</th><th>Line</th><th>Action</th></tr>\n")

    for func_name in sorted(usage_data['unused']):
        if func_name in usage_data['definitions']:
            for defn in usage_data['definitions'][func_name]:
                file_name = Path(defn['file']).name
                line = defn['line']
                abs_path = Path(defn['file']).resolve()
                vscode_link = f"vscode://file/{abs_path}:{line}:1"

                report.append(f"""
                    <tr>
                        <td><a href='{vscode_link}'><strong>{func_name}</strong></a></td>
                        <td>{file_name}</td>
                        <td>{line}</td>
                        <td><a href='{vscode_link}'>Review →</a></td>
                    </tr>
                """)

    report.append("</table>\n")
    report.append("</div>\n")

    # Most Used Functions Tab
    report.append("<div id='popular-content' class='tab-content'>\n")
    report.append("<h2>🔥 Most Frequently Called Functions</h2>\n")
    report.append("<table class='function-table'>\n")
    report.append("<tr><th>Function</th><th>Call Count</th><th>Called From Files</th><th>Called From Functions</th></tr>\n")

    # Get top 50 most called functions
    top_functions = sorted(usage_data['call_counts'].items(), key=lambda x: x[1], reverse=True)[:50]

    for func_name, call_count in top_functions:
        calls = usage_data['calls'].get(func_name, [])
        unique_files = set(Path(call['caller_file']).name for call in calls)
        unique_functions = set(call['caller_function'] for call in calls)

        report.append(f"""
            <tr>
                <td><strong>{func_name}</strong></td>
                <td><span class='call-count frequently-used'>{call_count}</span></td>
                <td>{len(unique_files)} files</td>
                <td>{len(unique_functions)} functions</td>
            </tr>
        """)

    report.append("</table>\n")
    report.append("</div>\n")

    # Detailed Callers Tab
    report.append("<div id='callers-content' class='tab-content'>\n")
    report.append("<h2>📞 Detailed Call Information</h2>\n")
    report.append("<p>Functions with their complete caller information:</p>\n")
    report.append("<input type='text' id='caller-search' class='search-box' placeholder='Search function calls...' onkeyup='filterTable(\"caller-search\", \"caller-table\")'>\n")
    report.append("<table id='caller-table' class='function-table'>\n")
    report.append("<tr><th>Function</th><th>Total Calls</th><th>Callers (File:Function:Line)</th></tr>\n")

    # Show functions with calls, sorted by call count
    for func_name, call_count in sorted(usage_data['call_counts'].items(), key=lambda x: x[1], reverse=True):
        calls = usage_data['calls'].get(func_name, [])

        # Group calls by caller
        caller_groups = defaultdict(list)
        for call in calls:
            caller_key = f"{Path(call['caller_file']).name}:{call['caller_function']}"
            caller_groups[caller_key].append(call['line'])

        # Format caller information
        caller_info = []
        for caller, lines in sorted(caller_groups.items()):
            line_nums = ', '.join(str(l) for l in sorted(lines)[:5])
            if len(lines) > 5:
                line_nums += f" (+{len(lines)-5})"
            caller_info.append(f"{caller} (lines: {line_nums})")

        caller_html = '<br>'.join(caller_info[:10])
        if len(caller_info) > 10:
            caller_html += f"<br><em>... and {len(caller_info)-10} more callers</em>"

        report.append(f"""
            <tr>
                <td><strong>{func_name}</strong></td>
                <td><span class='call-count frequently-used'>{call_count}</span></td>
                <td style='font-size: 0.9em;'>{caller_html}</td>
            </tr>
        """)

    report.append("</table>\n")
    report.append("</div>\n")

    report.append("</div>\n")  # Close container
    report.append("</body>\n</html>\n")

    # Save HTML report
    report_path = Path("FUNCTION_USAGE_STATS.html")
    with Path.open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report)

    print(f"\n✅ Detailed statistics report saved to {report_path}")

    # Also generate a CSV file for further analysis
    generate_csv_export(usage_data)

    return usage_data


def generate_csv_export(usage_data):
    """Generate CSV file with function usage data for further analysis."""
    import csv

    csv_path = Path("function_usage_stats.csv")

    with Path.open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Function', 'File', 'Line', 'Class', 'Call_Count', 'Unique_Callers', 'Unique_Files', 'Status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for func_name, definitions in usage_data['definitions'].items():
            call_count = usage_data['call_counts'].get(func_name, 0)
            calls = usage_data['calls'].get(func_name, [])

            unique_callers = set(call['caller_function'] for call in calls)
            unique_files = set(Path(call['caller_file']).name for call in calls)

            status = 'Unused' if call_count == 0 else 'Used'

            for defn in definitions:
                writer.writerow({
                    'Function': func_name,
                    'File': Path(defn['file']).name,
                    'Line': defn['line'],
                    'Class': defn.get('class', ''),
                    'Call_Count': call_count,
                    'Unique_Callers': len(unique_callers),
                    'Unique_Files': len(unique_files),
                    'Status': status
                })

    print(f"✅ CSV export saved to {csv_path}")


def main():
    """Main function to run detailed function usage analysis."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze function usage statistics in Python project')
    parser.add_argument('directory', nargs='?', default='emailops',
                       help='Directory to analyze (default: emailops)')
    parser.add_argument('--install', action='store_true',
                       help='Install required tools first')

    args = parser.parse_args()

    if args.install:
        install_tools()

    try:
        generate_detailed_report(args.directory)

        print("\n" + "=" * 60)
        print("📈 ANALYSIS COMPLETE")
        print("=" * 60)
        print("\nGenerated files:")
        print("  1. FUNCTION_USAGE_STATS.html - Interactive web report")
        print("  2. function_usage_stats.csv - CSV data for further analysis")
        print("\nOpen FUNCTION_USAGE_STATS.html in your browser to explore:")
        print("  • Detailed function call statistics")
        print("  • Which files call each function")
        print("  • Call frequency and distribution")
        print("  • Unused function identification")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Some tools may not be installed. Run with --install flag:")
        print("  python check_unused_code_stats.py --install")
        sys.exit(1)


if __name__ == "__main__":
    main()
