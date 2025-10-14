#!/usr/bin/env python3
"""
Local Code Quality Analysis for EmailOps Files
Performs comprehensive static analysis without requiring SonarQube server.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Files to analyze (from user's request)
TARGET_FILES = [
    "emailops/config.py",
    "emailops/doctor.py",
    "emailops/email_indexer.py",
    "emailops/env_utils.py",
    "emailops/index_metadata.py",
    "emailops/llm_client.py",
    "emailops/llm_runtime.py",
    "emailops/processor.py",
    "emailops/search_and_draft.py",
    "emailops/summarize_email_thread.py",
    "emailops/text_chunker.py",
    "emailops/utils.py",
    "emailops/validators.py",
    "emailops_gui.py",
]

def check_dependencies() -> list[str]:
    """Check and install required analysis tools."""
    tools = {
        "pylint": "pylint",
        "flake8": "flake8",
        "bandit": "bandit",
        "radon": "radon",
        "mypy": "mypy",
    }
    
    missing = []
    for tool_name, package in tools.items():
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", package],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                missing.append(package)
        except Exception:
            missing.append(package)
    
    if missing:
        print(f"Installing missing tools: {', '.join(missing)}")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install"] + missing,
                check=True,
                timeout=300
            )
            print("✓ Tools installed successfully\n")
        except Exception as e:
            print(f"⚠ Warning: Failed to install some tools: {e}\n")
            return missing
    
    return []

def run_pylint(files: list[Path]) -> dict[str, Any]:
    """Run Pylint analysis."""
    print("Running Pylint analysis...")
    result = {
        "tool": "pylint",
        "status": "success",
        "issues": [],
        "score": None,
    }
    
    try:
        cmd = [
            sys.executable, "-m", "pylint",
            "--output-format=json",
            "--disable=C0103,C0114,C0115,C0116",  # Disable some style checks
            "--max-line-length=120",
        ] + [str(f) for f in files]
        
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if proc.stdout:
            try:
                issues = json.loads(proc.stdout)
                result["issues"] = issues
                result["issue_count"] = len(issues)
            except Exception:
                result["raw_output"] = proc.stdout[:1000]
        
        # Extract score from stderr
        for line in proc.stderr.split("\n"):
            if "Your code has been rated at" in line:
                try:
                    score = float(line.split("rated at")[1].split("/")[0].strip())
                    result["score"] = score
                except Exception:
                    pass
        
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result

def run_flake8(files: list[Path]) -> dict[str, Any]:
    """Run Flake8 analysis."""
    print("Running Flake8 analysis...")
    result = {
        "tool": "flake8",
        "status": "success",
        "issues": [],
    }
    
    try:
        cmd = [
            sys.executable, "-m", "flake8",
            "--max-line-length=120",
            "--extend-ignore=E203,W503,E501",
            "--format=%(path)s:%(row)d:%(col)d: %(code)s %(text)s",
        ] + [str(f) for f in files]
        
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if proc.stdout:
            issues = []
            for line in proc.stdout.strip().split("\n"):
                if line:
                    issues.append(line)
            result["issues"] = issues
            result["issue_count"] = len(issues)
        
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result

def run_bandit(files: list[Path]) -> dict[str, Any]:
    """Run Bandit security analysis."""
    print("Running Bandit security analysis...")
    result = {
        "tool": "bandit",
        "status": "success",
        "issues": [],
    }
    
    try:
        cmd = [
            sys.executable, "-m", "bandit",
            "-r",
            "-f", "json",
        ] + [str(f) for f in files]
        
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if proc.stdout:
            try:
                data = json.loads(proc.stdout)
                result["issues"] = data.get("results", [])
                result["issue_count"] = len(data.get("results", []))
                result["metrics"] = data.get("metrics", {})
            except Exception:
                result["raw_output"] = proc.stdout[:1000]
        
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result

def run_radon_complexity(files: list[Path]) -> dict[str, Any]:
    """Run Radon complexity analysis."""
    print("Running Radon complexity analysis...")
    result = {
        "tool": "radon",
        "status": "success",
        "complexity": {},
    }
    
    try:
        cmd = [
            sys.executable, "-m", "radon",
            "cc",
            "-j",
        ] + [str(f) for f in files]
        
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if proc.stdout:
            try:
                complexity = json.loads(proc.stdout)
                result["complexity"] = complexity
                
                # Calculate stats
                high_complexity = []
                for file, funcs in complexity.items():
                    for func in funcs:
                        if func.get("complexity", 0) > 10:
                            high_complexity.append({
                                "file": file,
                                "function": func.get("name"),
                                "complexity": func.get("complexity"),
                            })
                
                result["high_complexity_functions"] = high_complexity
            except Exception:
                result["raw_output"] = proc.stdout[:1000]
        
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result

def run_mypy(files: list[Path]) -> dict[str, Any]:
    """Run MyPy type checking."""
    print("Running MyPy type analysis...")
    result = {
        "tool": "mypy",
        "status": "success",
        "issues": [],
    }
    
    try:
        cmd = [
            sys.executable, "-m", "mypy",
            "--ignore-missing-imports",
            "--no-error-summary",
        ] + [str(f) for f in files]
        
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if proc.stdout:
            issues = [line for line in proc.stdout.strip().split("\n") if line]
            result["issues"] = issues
            result["issue_count"] = len(issues)
        
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result

def generate_html_report(results: dict[str, Any], output_path: Path) -> None:
    """Generate HTML report from analysis results."""
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EmailOps Code Analysis Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 32px;
        }}
        .header p {{
            margin: 5px 0;
            opacity: 0.9;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .summary-card .value {{
            font-size: 36px;
            font-weight: bold;
            color: #333;
        }}
        .summary-card.score {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
        }}
        .summary-card.score h3,
        .summary-card.score .value {{
            color: white;
        }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            margin: 0 0 20px 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .issue {{
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #ffa500;
            background: #fffaf0;
            border-radius: 4px;
        }}
        .issue.error {{
            border-left-color: #dc3545;
            background: #fff5f5;
        }}
        .issue.warning {{
            border-left-color: #ffc107;
            background: #fffbf0;
        }}
        .issue.info {{
            border-left-color: #17a2b8;
            background: #f0f9ff;
        }}
        .issue-header {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .issue-location {{
            color: #666;
            font-size: 14px;
            font-family: 'Courier New', monospace;
        }}
        .complexity-high {{
            color: #dc3545;
        }}
        .complexity-medium {{
            color: #ffc107;
        }}
        .complexity-low {{
            color: #28a745;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #333;
        }}
        .status-success {{
            color: #28a745;
            font-weight: bold;
        }}
        .status-error {{
            color: #dc3545;
            font-weight: bold;
        }}
        .status-timeout {{
            color: #ffc107;
            font-weight: bold;
        }}
        pre {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
        }}
        .file-list {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }}
        .file-item {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 EmailOps Code Analysis Report</h1>
        <p><strong>Generated:</strong> {results.get('timestamp', 'N/A')}</p>
        <p><strong>Files Analyzed:</strong> {results.get('files_analyzed', 0)}</p>
        <p><strong>Total Lines:</strong> {results.get('total_lines', 0):,}</p>
    </div>
    
    <div class="summary">
        <div class="summary-card score">
            <h3>Pylint Score</h3>
            <div class="value">{results.get('pylint', {}).get('score', 'N/A')}</div>
        </div>
        <div class="summary-card">
            <h3>Issues Found</h3>
            <div class="value">{results.get('total_issues', 0)}</div>
        </div>
        <div class="summary-card">
            <h3>Security Issues</h3>
            <div class="value">{results.get('security_issues', 0)}</div>
        </div>
        <div class="summary-card">
            <h3>High Complexity</h3>
            <div class="value">{results.get('high_complexity_count', 0)}</div>
        </div>
    </div>
    
    <div class="section">
        <h2>📁 Files Analyzed</h2>
        <div class="file-list">
"""
    
    for f in results.get("files", []):
        html += f'            <div class="file-item">✓ {f}</div>\n'
    
    html += """
        </div>
    </div>
"""
    
    # Pylint results
    pylint_data = results.get("pylint", {})
    if pylint_data.get("status") == "success":
        html += f"""
    <div class="section">
        <h2>🔍 Pylint Analysis</h2>
        <p><strong>Score:</strong> <span style="font-size: 24px; color: #28a745;">{pylint_data.get('score', 'N/A')}/10.0</span></p>
        <p><strong>Issues Found:</strong> {pylint_data.get('issue_count', 0)}</p>
"""
        
        issues = pylint_data.get("issues", [])
        if issues:
            html += '        <div style="margin-top: 20px;">\n'
            for issue in issues[:50]:  # Show first 50
                severity_class = issue.get("type", "info").lower()
                html += f"""
            <div class="issue {severity_class}">
                <div class="issue-header">{issue.get('symbol', 'unknown')}: {issue.get('message', '')}</div>
                <div class="issue-location">{issue.get('path', '')}:{issue.get('line', '')}:{issue.get('column', '')}</div>
            </div>
"""
            if len(issues) > 50:
                html += f'            <p><em>...and {len(issues) - 50} more issues</em></p>\n'
            html += '        </div>\n'
        
        html += "    </div>\n"
    
    # Flake8 results
    flake8_data = results.get("flake8", {})
    if flake8_data.get("status") == "success" and flake8_data.get("issues"):
        html += f"""
    <div class="section">
        <h2>🎯 Flake8 Analysis</h2>
        <p><strong>Issues Found:</strong> {flake8_data.get('issue_count', 0)}</p>
        <div style="margin-top: 20px;">
"""
        for issue in flake8_data.get("issues", [])[:30]:
            html += f'            <div class="issue warning">{issue}</div>\n'
        
        if len(flake8_data.get("issues", [])) > 30:
            html += f'            <p><em>...and {len(flake8_data.get("issues", [])) - 30} more issues</em></p>\n'
        
        html += """
        </div>
    </div>
"""
    
    # Bandit security results
    bandit_data = results.get("bandit", {})
    if bandit_data.get("status") == "success":
        html += f"""
    <div class="section">
        <h2>🔒 Bandit Security Analysis</h2>
        <p><strong>Security Issues:</strong> {bandit_data.get('issue_count', 0)}</p>
"""
        
        issues = bandit_data.get("issues", [])
        if issues:
            html += '        <div style="margin-top: 20px;">\n'
            for issue in issues[:30]:
                severity = issue.get("issue_severity", "LOW")
                severity_class = "error" if severity == "HIGH" else ("warning" if severity == "MEDIUM" else "info")
                html += f"""
            <div class="issue {severity_class}">
                <div class="issue-header">[{severity}] {issue.get('issue_text', '')}</div>
                <div class="issue-location">{issue.get('filename', '')}:{issue.get('line_number', '')}</div>
                <div style="margin-top: 10px; color: #666;">{issue.get('issue_confidence', '')}</div>
            </div>
"""
            if len(issues) > 30:
                html += f'            <p><em>...and {len(issues) - 30} more issues</em></p>\n'
            html += '        </div>\n'
        
        html += "    </div>\n"
    
    # Radon complexity results
    radon_data = results.get("radon", {})
    if radon_data.get("status") == "success" and radon_data.get("high_complexity_functions"):
        html += """
    <div class="section">
        <h2>📈 Cyclomatic Complexity</h2>
        <p><strong>High Complexity Functions:</strong> Functions with complexity > 10</p>
        <table>
            <thead>
                <tr>
                    <th>File</th>
                    <th>Function</th>
                    <th>Complexity</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for func in radon_data.get("high_complexity_functions", []):
            complexity = func.get("complexity", 0)
            complexity_class = "complexity-high" if complexity > 20 else ("complexity-medium" if complexity > 15 else "complexity-low")
            html += f"""
                <tr>
                    <td>{Path(func.get('file', '')).name}</td>
                    <td><code>{func.get('function', '')}</code></td>
                    <td class="{complexity_class}"><strong>{complexity}</strong></td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
    </div>
"""
    
    # MyPy results
    mypy_data = results.get("mypy", {})
    if mypy_data.get("status") == "success" and mypy_data.get("issues"):
        html += f"""
    <div class="section">
        <h2>🔤 MyPy Type Checking</h2>
        <p><strong>Type Issues:</strong> {mypy_data.get('issue_count', 0)}</p>
        <div style="margin-top: 20px;">
"""
        for issue in mypy_data.get("issues", [])[:30]:
            html += f'            <div class="issue info">{issue}</div>\n'
        
        if len(mypy_data.get("issues", [])) > 30:
            html += f'            <p><em>...and {len(mypy_data.get("issues", [])) - 30} more issues</em></p>\n'
        
        html += """
        </div>
    </div>
"""
    
    # Tool status summary
    html += """
    <div class="section">
        <h2>🛠️ Analysis Tools Status</h2>
        <table>
            <thead>
                <tr>
                    <th>Tool</th>
                    <th>Status</th>
                    <th>Details</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for tool_name in ["pylint", "flake8", "bandit", "radon", "mypy"]:
        tool_data = results.get(tool_name, {})
        status = tool_data.get("status", "not_run")
        status_class = f"status-{status}"
        
        details = ""
        if status == "success":
            if tool_name == "pylint":
                details = f"Score: {tool_data.get('score', 'N/A')}, Issues: {tool_data.get('issue_count', 0)}"
            else:
                details = f"Issues: {tool_data.get('issue_count', 0)}"
        elif status == "error":
            details = tool_data.get("error", "Unknown error")
        
        html += f"""
                <tr>
                    <td><strong>{tool_name.upper()}</strong></td>
                    <td class="{status_class}">{status.upper()}</td>
                    <td>{details}</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
    </div>
    
    <div class="section">
        <h2>📋 Summary</h2>
        <p>This analysis was performed locally without requiring a SonarQube server.</p>
        <p>For continuous integration and more features, consider setting up SonarQube server (see SONARQUBE_SETUP_GUIDE.md).</p>
        <p><strong>JSON Report:</strong> analysis_results.json</p>
    </div>
</body>
</html>
"""
    
    output_path.write_text(html, encoding="utf-8")
    print(f"✓ HTML report saved to: {output_path}")

def main() -> int:
    """Main analysis workflow."""
    print("="*60)
    print("EmailOps Local Code Quality Analysis")
    print("="*60)
    print()
    
    project_dir = Path(__file__).parent
    
    # Verify files exist
    files = [project_dir / f for f in TARGET_FILES]
    existing_files = [f for f in files if f.exists()]
    missing_files = [f for f in files if not f.exists()]
    
    if missing_files:
        print(f"⚠ Warning: {len(missing_files)} files not found:")
        for f in missing_files:
            print(f"  - {f}")
        print()
    
    if not existing_files:
        print("✗ No files found to analyze!")
        return 1
    
    print(f"Files to analyze: {len(existing_files)}")
    print()
    
    # Check/install dependencies
    missing_tools = check_dependencies()
    if missing_tools:
        print(f"⚠ Some tools could not be installed: {missing_tools}")
        print("Continuing with available tools...\n")
    
    # Count lines
    total_lines = 0
    for f in existing_files:
        try:
            total_lines += len(f.read_text(encoding="utf-8", errors="ignore").splitlines())
        except Exception:
            pass
    
    # Run analyses
    results: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "files": [str(f.relative_to(project_dir)) for f in existing_files],
        "files_analyzed": len(existing_files),
        "total_lines": total_lines,
    }
    
    # Run each tool
    results["pylint"] = run_pylint(existing_files)
    results["flake8"] = run_flake8(existing_files)
    results["bandit"] = run_bandit(existing_files)
    results["radon"] = run_radon_complexity(existing_files)
    results["mypy"] = run_mypy(existing_files)
    
    # Calculate totals
    results["total_issues"] = sum([
        results.get("pylint", {}).get("issue_count", 0),
        results.get("flake8", {}).get("issue_count", 0),
        results.get("mypy", {}).get("issue_count", 0),
    ])
    results["security_issues"] = results.get("bandit", {}).get("issue_count", 0)
    results["high_complexity_count"] = len(results.get("radon", {}).get("high_complexity_functions", []))
    
    # Save JSON report
    json_output = project_dir / "analysis_results.json"
    json_output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ JSON report saved to: {json_output}")
    
    # Generate HTML report
    html_output = project_dir / "analysis_report.html"
    generate_html_report(results, html_output)
    
    # Print summary
    print("\n" + "="*60)
    print("Analysis Summary")
    print("="*60)
    print(f"Pylint Score:        {results.get('pylint', {}).get('score', 'N/A')}/10.0")
    print(f"Total Issues:        {results.get('total_issues', 0)}")
    print(f"Security Issues:     {results.get('security_issues', 0)}")
    print(f"High Complexity:     {results.get('high_complexity_count', 0)} functions")
    print("="*60)
    print(f"\n✓ Analysis complete! Open {html_output.name} in your browser.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())