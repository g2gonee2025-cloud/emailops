
#!/usr/bin/env python3
"""
Comprehensive Code Analyzer - Combines unused code detection, function statistics,
and dependency validation. Checks for:
- Unused functions, variables, classes, and imports
- Function call statistics and relationships
- Incorrectly referenced files and dependencies
- Missing imports and broken references
"""

import ast
import csv
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


def install_tools():
    """Install required tools for analysis."""
    tools = ['vulture', 'pyflakes', 'pipreqs', 'pipdeptree']

    print("Installing required tools...")
    for tool in tools:
        subprocess.run([sys.executable, '-m', 'pip', 'install', tool],
                      capture_output=True)
    print("Tools installed.\n")


class DependencyChecker:
    """Check for dependency and file reference issues."""

    def __init__(self, directory: str = "emailops"):
        self.directory = Path(directory)
        self.issues = {
            'missing_files': [],
            'broken_imports': [],
            'circular_dependencies': [],
            'missing_requirements': [],
            'unused_requirements': [],
            'hardcoded_paths': []
        }

    def check_import_references(self) -> dict[str, list[str]]:
        """Check for imports that reference non-existent modules or files."""
        print("\n" + "=" * 60)
        print("Checking Import References")
        print("=" * 60)

        py_files = list(self.directory.rglob("*.py"))

        for filepath in py_files:
            try:
                with Path.open(filepath, encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            self._check_module_exists(alias.name, filepath, node.lineno)

                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            # Check relative imports
                            if node.level > 0:
                                self._check_relative_import(node.module, node.level, filepath, node.lineno)
                            else:
                                self._check_module_exists(node.module, filepath, node.lineno)

                        # Check imported names
                        for alias in node.names:
                            if node.module and alias.name != '*':
                                self._check_name_in_module(node.module, alias.name, filepath, node.lineno)

            except Exception as e:
                print(f"  Error processing {filepath}: {e}")

        return self.issues

    def _check_module_exists(self, module_name: str, filepath: Path, line: int):
        """Check if a module exists either as installed package or local file."""
        # Check if it's a local module
        parts = module_name.split('.')

        # Try to find as a Python file in the project
        current_dir = filepath.parent
        module_path = current_dir

        for part in parts[:-1]:
            module_path = module_path / part

        # Check for .py file or package directory
        py_file = module_path / f"{parts[-1]}.py"
        pkg_dir = module_path / parts[-1] / "__init__.py"

        # Also check from project root
        root_py = self.directory / f"{module_name.replace('.', '/')}.py"
        root_pkg = self.directory / module_name.replace('.', '/') / "__init__.py"

        if not any([py_file.exists(), pkg_dir.exists(), root_py.exists(), root_pkg.exists()]):
            # Check if it's a standard library or installed package
            try:
                __import__(module_name)
            except ImportError:
                self.issues['broken_imports'].append({
                    'file': str(filepath),
                    'line': line,
                    'module': module_name,
                    'type': 'missing_module'
                })

    def _check_relative_import(self, module: str, level: int, filepath: Path, line: int):
        """Check if relative import is valid."""
        current_dir = filepath.parent

        # Go up directories based on level
        for _ in range(level - 1):
            current_dir = current_dir.parent
            if current_dir == self.directory.parent:
                self.issues['broken_imports'].append({
                    'file': str(filepath),
                    'line': line,
                    'module': module,
                    'type': 'relative_import_outside_package'
                })
                return

        # Check if target module exists
        if module:
            target = current_dir / module.replace('.', '/')
            if not (target.with_suffix('.py').exists() or (target / "__init__.py").exists()):
                self.issues['broken_imports'].append({
                    'file': str(filepath),
                    'line': line,
                    'module': module,
                    'type': 'relative_import_not_found'
                })

    def _check_name_in_module(self, module: str, name: str, filepath: Path, line: int):
        """Check if a specific name exists in a module."""
        # This is a simplified check - full implementation would need to parse the target module
        pass

    def check_hardcoded_paths(self) -> list[dict]:
        """Find hardcoded file paths that don't exist."""
        print("\n" + "=" * 60)
        print("Checking Hardcoded File Paths")
        print("=" * 60)

        # Patterns for file paths
        path_patterns = [
            r'["\']([a-zA-Z0-9_/\\.-]+\.(py|txt|json|yml|yaml|csv|log|md|html|css|js))["\']',
            r'Path\(["\']([^"\']+)["\']\)',
            r'open\(["\']([^"\']+)["\']\)',
            r'\.read\(["\']([^"\']+)["\']\)',
            r'\.write\(["\']([^"\']+)["\']\)'
        ]

        py_files = list(self.directory.rglob("*.py"))

        for filepath in py_files:
            try:
                with Path.open(filepath, encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')

                for i, line in enumerate(lines, 1):
                    for pattern in path_patterns:
                        matches = re.findall(pattern, line)
                        for match in matches:
                            path_str = match[0] if isinstance(match, tuple) else match

                            # Skip obvious non-paths
                            if path_str in ['r', 'w', 'a', 'rb', 'wb', 'ab', 'x']:
                                continue

                            # Check if path exists
                            path = Path(path_str)
                            if not path.is_absolute():
                                # Check relative to file location and project root
                                file_relative = filepath.parent / path
                                project_relative = self.directory / path

                                if not file_relative.exists() and not project_relative.exists():
                                    # Skip common false positives
                                    if not any(skip in path_str for skip in [
                                        'http://', 'https://', 'ftp://', '://',
                                        'example.', 'test.', 'demo.', 'sample.',
                                        '<', '>', '{', '}', '$', '%'
                                    ]):
                                        self.issues['hardcoded_paths'].append({
                                            'file': str(filepath),
                                            'line': i,
                                            'path': path_str
                                        })

            except Exception as e:
                print(f"  Error processing {filepath}: {e}")

        return self.issues['hardcoded_paths']

    def check_circular_dependencies(self) -> list[list[str]]:
        """Detect circular import dependencies."""
        print("\n" + "=" * 60)
        print("Checking for Circular Dependencies")
        print("=" * 60)

        # Build import graph
        import_graph = defaultdict(set)
        py_files = list(self.directory.rglob("*.py"))

        for filepath in py_files:
            try:
                with Path.open(filepath, encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)

                module_name = self._get_module_name(filepath)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            import_graph[module_name].add(alias.name)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        import_graph[module_name].add(node.module)

            except Exception:
                continue

        # Find cycles using DFS
        cycles = []
        visited = set()
        rec_stack = []

        def find_cycles(module, path):
            if module in rec_stack:
                # Found a cycle
                cycle_start = rec_stack.index(module)
                cycle = rec_stack[cycle_start:] + [module]
                if len(cycle) > 1 and cycle not in cycles:
                    cycles.append(cycle)
                return

            if module in visited:
                return

            visited.add(module)
            rec_stack.append(module)

            for imported in import_graph.get(module, []):
                find_cycles(imported, path + [imported])

            rec_stack.pop()

        for module in import_graph:
            if module not in visited:
                find_cycles(module, [module])

        self.issues['circular_dependencies'] = cycles
        return cycles

    def _get_module_name(self, filepath: Path) -> str:
        """Convert file path to module name."""
        relative = filepath.relative_to(self.directory)
        parts = list(relative.parts[:-1]) + [relative.stem]
        return '.'.join(parts)

    def check_requirements(self) -> dict[str, list[str]]:
        """Check for missing or unused requirements."""
        print("\n" + "=" * 60)
        print("Checking Requirements")
        print("=" * 60)

        # Get imports from code
        imports = set()
        py_files = list(self.directory.rglob("*.py"))

        for filepath in py_files:
            try:
                with Path.open(filepath, encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imports.add(node.module.split('.')[0])

            except Exception:
                continue

        # Check requirements.txt
        req_file = self.directory.parent / 'requirements.txt'
        if req_file.exists():
            with Path.open(req_file) as f:
                requirements = set()
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract package name
                        pkg = re.split('[<>=!]', line)[0].strip()
                        requirements.add(pkg.lower())

            # Find missing requirements (imported but not in requirements)
            stdlib_modules = set(sys.stdlib_module_names) if hasattr(sys, 'stdlib_module_names') else set()
            for imp in imports:
                if imp not in stdlib_modules and imp.lower() not in requirements:
                    # Check if it's a local module
                    if not (self.directory / imp).exists() and not (self.directory / f"{imp}.py").exists():
                        self.issues['missing_requirements'].append(imp)

            # Find unused requirements (in requirements but not imported)
            imported_lower = {imp.lower() for imp in imports}
            for req in requirements:
                if req not in imported_lower:
                    self.issues['unused_requirements'].append(req)

        return {
            'missing': self.issues['missing_requirements'],
            'unused': self.issues['unused_requirements']
        }


class ComprehensiveAnalyzer:
    """Main analyzer combining all analysis functions."""

    def __init__(self, directory: str = "emailops"):
        self.directory = directory
        self.dep_checker = DependencyChecker(directory)
        self.results = {}

    def run_vulture(self) -> dict[str, list[str]]:
        """Run vulture to find unused code."""
        print("=" * 60)
        print("Running Vulture - Dead Code Detector")
        print("=" * 60)

        result = subprocess.run(
            ['vulture', self.directory, '--min-confidence', '80'],
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

        return unused

    def run_pyflakes(self) -> list[str]:
        """Run pyflakes to find unused imports and undefined names."""
        print("\n" + "=" * 60)
        print("Running Pyflakes - Unused Imports & Undefined Names")
        print("=" * 60)

        result = subprocess.run(
            ['pyflakes', self.directory],
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

        return issues

    def run_ruff_unused(self) -> dict[str, int]:
        """Run ruff to check for unused code patterns."""
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
            ['ruff', 'check', self.directory, '--select', ','.join(patterns), '--output-format', 'json'],
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

        return counts

    def analyze_function_usage_detailed(self) -> dict:
        """
        Analyze USER-DEFINED function definitions and calls with detailed statistics.
        Filters out built-in/native functions and only tracks project functions.
        Returns comprehensive usage data including call counts and caller information.
        """
        print("\n" + "=" * 60)
        print("Analyzing User-Defined Function Usage Statistics")
        print("=" * 60)

        # Data structures to track function usage
        function_definitions = {}  # func_name -> [{file, line, class}]
        function_calls = defaultdict(list)  # func_name -> [{caller_file, caller_func, line}]
        call_counts = Counter()  # func_name -> count

        # First pass: collect all user-defined functions
        py_files = list(Path(self.directory).rglob("*.py"))
        user_defined_functions = set()  # Track all functions defined in the project

        print("  Scanning for user-defined functions...")
        for filepath in py_files:
            try:
                with Path.open(filepath, encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        user_defined_functions.add(node.name)
                        # Also track method names with common prefixes
                        if '.' not in node.name:
                            # Add common method call patterns
                            user_defined_functions.add(f"self.{node.name}")
                            user_defined_functions.add(f"cls.{node.name}")
            except Exception:
                continue

        print(f"  Found {len(user_defined_functions)} user-defined function names")

        # Second pass: analyze definitions and calls
        for filepath in py_files:
            try:
                with Path.open(filepath, encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)

                file_name = str(filepath)

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

                    def _process_function_def(self, node):
                        """Common processing for both sync and async function definitions."""
                        func_name = node.name
                        if func_name not in function_definitions:
                            function_definitions[func_name] = []

                        function_definitions[func_name].append({
                            'file': file_name,
                            'line': node.lineno,
                            'class': self.current_class,
                            'full_name': f"{self.current_class}.{func_name}" if self.current_class else func_name,
                            'is_method': self.current_class is not None
                        })

                        # Track current function context for calls
                        old_function = self.current_function
                        self.current_function = func_name
                        self.function_stack.append(func_name)
                        self.generic_visit(node)
                        self.function_stack.pop()
                        self.current_function = old_function

                    def visit_FunctionDef(self, node):
                        # Record regular function definition
                        self._process_function_def(node)

                    def visit_AsyncFunctionDef(self, node):
                        # Record async function definition
                        self._process_function_def(node)

                    def visit_Call(self, node):
                        # Track function calls - ONLY USER-DEFINED FUNCTIONS
                        func_name = None

                        if isinstance(node.func, ast.Name):
                            # Direct function call: func()
                            func_name = node.func.id
                        elif isinstance(node.func, ast.Attribute):
                            # Method call: obj.method()
                            func_name = node.func.attr
                            # Check if it's a self/cls method call
                            if isinstance(node.func.value, ast.Name):
                                if node.func.value.id in ('self', 'cls'):
                                    # Track as method name for better matching
                                    func_name = node.func.attr

                        # Only track if it's a user-defined function
                        if func_name and func_name in user_defined_functions:
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

        # Analyze the collected data - only for user-defined functions
        total_functions = len(function_definitions)

        # Only consider functions that we've defined in our project
        called_functions = set(call_counts.keys()) & set(function_definitions.keys())
        unused_functions = set(function_definitions.keys()) - called_functions

        # Filter out special methods and test functions from unused
        unused_functions = {
            func for func in unused_functions
            if not (func.startswith('__') or
                   func.startswith('test_') or
                   func in ['main', 'setUp', 'tearDown'])
        }

        print("\n📊 User-Defined Function Statistics:")
        print(f"  Total user functions defined: {total_functions}")
        print(f"  User functions with calls: {len(called_functions)}")
        print(f"  Potentially unused user functions: {len(unused_functions)}")
        print(f"  Total user function calls tracked: {sum([count for func, count in call_counts.items() if func in function_definitions])}")

        # Show top unused functions
        if unused_functions:
            print("\n🚫 Top Unused User Functions:")
            for func_name in sorted(unused_functions)[:10]:
                if func_name in function_definitions:
                    for defn in function_definitions[func_name]:
                        file_name = Path(defn['file']).name
                        print(f"    - {func_name} in {file_name}:{defn['line']}")

        # Calculate caller diversity - only for user-defined functions
        caller_diversity = {}
        for func_name, calls in function_calls.items():
            if func_name in function_definitions:  # Only user-defined functions
                unique_files = set(call['caller_file'] for call in calls)
                unique_functions = set(call['caller_function'] for call in calls)
                caller_diversity[func_name] = {
                    'file_count': len(unique_files),
                    'function_count': len(unique_functions),
                    'total_calls': len(calls)
                }

        # Show most called user functions
        user_call_counts = {func: count for func, count in call_counts.items() if func in function_definitions}
        if user_call_counts:
            print("\n🔥 Most Called User Functions:")
            for func_name, count in sorted(user_call_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"    - {func_name}: {count} calls")

        return {
            'definitions': function_definitions,
            'calls': dict(function_calls),
            'call_counts': dict(call_counts),
            'unused': list(unused_functions),
            'caller_diversity': caller_diversity
        }

    def generate_comprehensive_report(self):
        """Generate a comprehensive HTML report combining all analyses."""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE CODE ANALYSIS REPORT")
        print("=" * 60)

        # Run all analyses
        vulture_results = self.run_vulture()
        pyflakes_results = self.run_pyflakes()
        ruff_results = self.run_ruff_unused()
        function_usage = self.analyze_function_usage_detailed()

        # Run dependency checks
        self.dep_checker.check_import_references()
        self.dep_checker.check_hardcoded_paths()
        self.dep_checker.check_circular_dependencies()
        self.dep_checker.check_requirements()

        # Generate HTML report
        report = []
        report.append("<!DOCTYPE html>\n<html>\n<head>\n")
        report.append("<title>Comprehensive Code Analysis Report</title>\n")
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
            .issue-card { background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 10px 0; }
            .error-card { background: #f8d7da; border-left: 4px solid #dc3545; padding: 10px; margin: 10px 0; }
            .success-card { background: #d4edda; border-left: 4px solid #28a745; padding: 10px; margin: 10px 0; }
            .function-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            .function-table th { background: #3498db; color: white; padding: 10px; text-align: left; }
            .function-table td { padding: 8px; border-bottom: 1px solid #ecf0f1; }
            .function-table tr:hover { background: #f8f9fa; }
            .call-count { background: #3498db; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.9em; display: inline-block; }
            .unused { background: #e74c3c; }
            .rarely-used { background: #f39c12; }
            .frequently-used { background: #27ae60; }
            .tabs { display: flex; border-bottom: 2px solid #ecf0f1; margin: 20px 0; }
            .tab { padding: 10px 20px; cursor: pointer; background: #ecf0f1; margin-right: 5px; border-radius: 5px 5px 0 0; }
            .tab.active { background: #3498db; color: white; }
            .tab-content { display: none; padding: 20px 0; }
            .tab-content.active { display: block; }
            .code-block { background: #2c3e50; color: #ecf0f1; padding: 10px; border-radius: 5px; font-family: 'Courier New', monospace; overflow-x: auto; }
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
        report.append("<h1>📊 Comprehensive Code Analysis Report</h1>\n")
        report.append("<h2 style='color: #e74c3c; margin-top: 10px;'>Focus: User-Defined Functions Only</h2>\n")
        report.append(f"<p>Analysis of: <code>{Path(self.directory).resolve()}</code></p>\n")
        report.append("<p><em>Native/built-in functions are excluded from this analysis</em></p>\n")

        # Summary statistics cards
        total_functions = len(function_usage['definitions'])
        total_calls = sum(function_usage['call_counts'].values())
        unused_count = len(function_usage['unused'])
        broken_imports = len(self.dep_checker.issues['broken_imports'])
        circular_deps = len(self.dep_checker.issues['circular_dependencies'])
        hardcoded_paths = len(self.dep_checker.issues['hardcoded_paths'])

        report.append("<div class='stats-grid'>\n")
        report.append(f"""
            <div class='stat-card'>
                <div class='stat-number'>{total_functions}</div>
                <div class='stat-label'>Total Functions</div>
            </div>
            <div class='stat-card'>
                <div class='stat-number'>{unused_count}</div>
                <div class='stat-label'>Unused Functions</div>
            </div>
            <div class='stat-card'>
                <div class='stat-number'>{broken_imports}</div>
                <div class='stat-label'>Broken Imports</div>
            </div>
            <div class='stat-card'>
                <div class='stat-number'>{circular_deps}</div>
                <div class='stat-label'>Circular Dependencies</div>
            </div>
            <div class='stat-card'>
                <div class='stat-number'>{hardcoded_paths}</div>
                <div class='stat-label'>Invalid File Paths</div>
            </div>
            <div class='stat-card'>
                <div class='stat-number'>{len(vulture_results['unreachable_code'])}</div>
                <div class='stat-label'>Unreachable Code</div>
            </div>
        """)
        report.append("</div>\n")

        # Tabs for different views
        report.append("""
            <div class='tabs'>
                <div id='overview-tab' class='tab active' onclick='showTab("overview")'>Overview</div>
                <div id='functions-tab' class='tab' onclick='showTab("functions")'>Functions</div>
                <div id='dependencies-tab' class='tab' onclick='showTab("dependencies")'>Dependencies</div>
                <div id='imports-tab' class='tab' onclick='showTab("imports")'>Import Issues</div>
                <div id='paths-tab' class='tab' onclick='showTab("paths")'>File Paths</div>
            </div>
        """)

        # Overview Tab
        report.append("<div id='overview-content' class='tab-content active'>\n")
        report.append("<h2>Analysis Overview</h2>\n")

        # Critical Issues
        if broken_imports or circular_deps or hardcoded_paths:
            report.append("<h3>⚠️ Critical Issues</h3>\n")
            if broken_imports:
                report.append(f"<div class='error-card'>Found {broken_imports} broken imports that need fixing</div>\n")
            if circular_deps:
                report.append(f"<div class='error-card'>Found {circular_deps} circular dependencies</div>\n")
            if hardcoded_paths:
                report.append(f"<div class='issue-card'>Found {hardcoded_paths} hardcoded paths that don't exist</div>\n")

        report.append("</div>\n")

        # Functions Tab
        report.append("<div id='functions-content' class='tab-content'>\n")
        report.append("<h2>User-Defined Function Analysis</h2>\n")
        report.append("<p><strong>Note:</strong> This analysis only includes functions defined in your project, excluding all built-in/native Python functions.</p>\n")

        # Unused functions
        if function_usage['unused']:
            report.append("<h3>🚫 Unused User Functions</h3>\n")
            report.append("<table class='function-table'>\n")
            report.append("<tr><th>Function</th><th>File</th><th>Line</th></tr>\n")

            for func_name in sorted(function_usage['unused'])[:50]:
                if func_name in function_usage['definitions']:
                    for defn in function_usage['definitions'][func_name]:
                        file_name = Path(defn['file']).name
                        abs_path = Path(defn['file']).resolve()
                        vscode_link = f"vscode://file/{abs_path}:{defn['line']}:1"
                        report.append(f"""
                            <tr>
                                <td><a href='{vscode_link}'>{func_name}</a></td>
                                <td>{file_name}</td>
                                <td>{defn['line']}</td>
                            </tr>
                        """)

            report.append("</table>\n")

        # Most called user functions - filter to only user-defined
        user_functions_only = {
            func: count for func, count in function_usage['call_counts'].items()
            if func in function_usage['definitions']
        }
        most_called = sorted(user_functions_only.items(), key=lambda x: x[1], reverse=True)[:20]
        if most_called:
            report.append("<h3>🔥 Most Called User Functions</h3>\n")
            report.append("<table class='function-table'>\n")
            report.append("<tr><th>Function</th><th>Call Count</th><th>Unique Callers</th></tr>\n")

            for func_name, count in most_called:
                diversity = function_usage['caller_diversity'].get(func_name, {})
                report.append(f"""
                    <tr>
                        <td>{func_name}</td>
                        <td><span class='call-count frequently-used'>{count}</span></td>
                        <td>{diversity.get('function_count', 0)}</td>
                    </tr>
                """)

            report.append("</table>\n")

        report.append("</div>\n")

        # Dependencies Tab
        report.append("<div id='dependencies-content' class='tab-content'>\n")
        report.append("<h2>Dependency Analysis</h2>\n")

        # Circular dependencies
        if self.dep_checker.issues['circular_dependencies']:
            report.append("<h3>🔄 Circular Dependencies</h3>\n")
            for cycle in self.dep_checker.issues['circular_dependencies']:
                cycle_str = ' → '.join(cycle)
                report.append(f"<div class='error-card'>{cycle_str}</div>\n")

        # Missing requirements
        if self.dep_checker.issues['missing_requirements']:
            report.append("<h3>📦 Missing Requirements</h3>\n")
            report.append("<p>These packages are imported but not in requirements.txt:</p>\n")
            report.append("<ul>\n")
            for req in self.dep_checker.issues['missing_requirements']:
                report.append(f"<li><code>{req}</code></li>\n")
            report.append("</ul>\n")

        # Unused requirements
        if self.dep_checker.issues['unused_requirements']:
            report.append("<h3>📦 Unused Requirements</h3>\n")
            report.append("<p>These packages are in requirements.txt but never imported:</p>\n")
            report.append("<ul>\n")
            for req in self.dep_checker.issues['unused_requirements']:
                report.append(f"<li><code>{req}</code></li>\n")
            report.append("</ul>\n")

        report.append("</div>\n")

        # Import Issues Tab
        report.append("<div id='imports-content' class='tab-content'>\n")
        report.append("<h2>Import Issues</h2>\n")

        if self.dep_checker.issues['broken_imports']:
            report.append("<h3>❌ Broken Imports</h3>\n")
            report.append("<table class='function-table'>\n")
            report.append("<tr><th>File</th><th>Line</th><th>Module</th><th>Issue Type</th></tr>\n")

            for imp in self.dep_checker.issues['broken_imports'][:50]:
                file_name = Path(imp['file']).name
                abs_path = Path(imp['file']).resolve()
                vscode_link = f"vscode://file/{abs_path}:{imp['line']}:1"
                report.append(f"""
                    <tr>
                        <td><a href='{vscode_link}'>{file_name}</a></td>
                        <td>{imp['line']}</td>
                        <td><code>{imp['module']}</code></td>
                        <td>{imp['type'].replace('_', ' ').title()}</td>
                    </tr>
                """)

            report.append("</table>\n")

        report.append("</div>\n")

        # File Paths Tab
        report.append("<div id='paths-content' class='tab-content'>\n")
        report.append("<h2>File Path Issues</h2>\n")

        if self.dep_checker.issues['hardcoded_paths']:
            report.append("<h3>📁 Invalid Hardcoded Paths</h3>\n")
            report.append("<table class='function-table'>\n")
            report.append("<tr><th>File</th><th>Line</th><th>Path</th></tr>\n")

            for path_issue in self.dep_checker.issues['hardcoded_paths'][:50]:
                file_name = Path(path_issue['file']).name
                abs_path = Path(path_issue['file']).resolve()
                vscode_link = f"vscode://file/{abs_path}:{path_issue['line']}:1"
                report.append(f"""
                    <tr>
                        <td><a href='{vscode_link}'>{file_name}</a></td>
                        <td>{path_issue['line']}</td>
                        <td><code>{path_issue['path']}</code></td>
                    </tr>
                """)

            report.append("</table>\n")

        report.append("</div>\n")

        # Close HTML
        report.append("</div>\n")  # Close container
        report.append("</body>\n</html>\n")

        # Save HTML report
        report_path = Path("COMPREHENSIVE_CODE_ANALYSIS.html")
        with Path.open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(report)

        print(f"\n✅ Comprehensive report saved to {report_path}")

        # Generate CSV export
        self._generate_csv_export(function_usage)

        # Print cleanup suggestions
        self._print_cleanup_suggestions()

    def _generate_csv_export(self, usage_data):
        """Generate CSV file with analysis data."""
        csv_path = Path("code_analysis_export.csv")

        with Path.open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Type', 'Item', 'File', 'Line', 'Details']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            # Write unused functions
            for func_name in usage_data['unused']:
                if func_name in usage_data['definitions']:
                    for defn in usage_data['definitions'][func_name]:
                        writer.writerow({
                            'Type': 'Unused Function',
                            'Item': func_name,
                            'File': Path(defn['file']).name,
                            'Line': defn['line'],
                            'Details': f"Class: {defn.get('class', 'None')}"
                        })

            # Write broken imports
            for imp in self.dep_checker.issues['broken_imports']:
                writer.writerow({
                    'Type': 'Broken Import',
                    'Item': imp['module'],
                    'File': Path(imp['file']).name,
                    'Line': imp['line'],
                    'Details': imp['type']
                })

            # Write invalid paths
            for path_issue in self.dep_checker.issues['hardcoded_paths']:
                writer.writerow({
                    'Type': 'Invalid Path',
                    'Item': path_issue['path'],
                    'File': Path(path_issue['file']).name,
                    'Line': path_issue['line'],
                    'Details': 'File not found'
                })

        print(f"✅ CSV export saved to {csv_path}")

    def _print_cleanup_suggestions(self):
        """Print suggested cleanup commands."""
        print("\n" + "=" * 60)
        print("SUGGESTED CLEANUP COMMANDS")
        print("=" * 60)
        print("1. Remove unused imports automatically:")
        print(f"   ruff check --fix --select F401 {self.directory}")
        print("\n2. Remove unused variables automatically:")
        print(f"   ruff check --fix --select F841 {self.directory}")
        print("\n3. Review all issues with vulture:")
        print(f"   vulture {self.directory} --min-confidence 60")
        print("\n4. Check for security issues:")
        print(f"   bandit -r {self.directory}")
        print("\n5. Format code:")
        print(f"   black {self.directory}")
        print("\n6. Sort imports:")
        print(f"   isort {self.directory}")


def main():
    """Main function to run comprehensive code analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Comprehensive code analysis tool for Python projects'
    )
    parser.add_argument('directory', nargs='?', default='emailops',
                       help='Directory to analyze (default: emailops)')
    parser.add_argument('--install', action='store_true',
                       help='Install required tools first')

    args = parser.parse_args()

    if args.install:
        install_tools()

    try:
        analyzer = ComprehensiveAnalyzer(args.directory)
        analyzer.generate_comprehensive_report()

        print("\n" + "=" * 60)
        print("📊 ANALYSIS COMPLETE")
        print("=" * 60)
        print("\nGenerated files:")
        print("  1. COMPREHENSIVE_CODE_ANALYSIS.html - Interactive web report")
        print("  2. code_analysis_export.csv - CSV data for further analysis")
        print("\nOpen COMPREHENSIVE_CODE_ANALYSIS.html in your browser to explore:")
        print("  • Unused code detection")
        print("  • Function usage statistics")
        print("  • Broken imports and references")
        print("  • Circular dependencies")
        print("  • Invalid file paths")
        print("  • Missing and unused requirements")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Some tools may not be installed. Run with --install flag:")
        print(f"  python {Path(__file__).name} --install")
        sys.exit(1)


if __name__ == "__main__":
    main()
