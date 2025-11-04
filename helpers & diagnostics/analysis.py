#!/usr/bin/env python3
"""
Consolidated code analysis and package generation utilities for EmailOps.
Combines functionality from run_local_analysis.py, create_production_packages.py,
generate_remediation_packages.py, and batch_prompt.py.
"""

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai

# -------------------------
# Batch Analysis with GenAI
# -------------------------


class BatchAnalyzer:
    """Batch analysis using Google GenAI."""

    def __init__(self, project: str | None = None, location: str = "global"):
        self.project = project or os.getenv("GOOGLE_CLOUD_PROJECT", "your-project-id")
        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "global")
        self.model_name = "gemini-2.5-pro"
        self.embed_model = "gemini-embedding-001"
        self.output_dir = Path("analysis_results")

    async def analyze_file(
        self, file_path: Path, client, prompt_template: str
    ) -> dict[str, Any]:
        """Analyze a single file with GenAI."""
        print(f"[STARTING] Analysis for {file_path}")
        result = {
            "file": str(file_path),
            "success": False,
            "analysis": None,
            "error": None,
        }

        try:
            file_content = file_path.read_text(encoding="utf-8")
            prompt = prompt_template.format(file_content=file_content)

            # Generate content using Gemini

            response = await client.aio.models.generate_content(
                model=self.model_name, contents=prompt
            )

            if response and response.text:
                self.output_dir.mkdir(exist_ok=True)
                output_filename = self.output_dir / f"{file_path.stem}.md"
                output_filename.write_text(response.text, encoding="utf-8")

                result["success"] = True
                result["analysis"] = response.text
                print(f"[SUCCESS] Analysis for {file_path} saved to {output_filename}")
            else:
                result["error"] = "Empty response from model"
                print(f"[FAILED] Analysis for {file_path}: Empty response")

        except Exception as e:
            result["error"] = str(e)
            print(f"[FAILED] Analysis for {file_path}. Error: {e}")

        return result

    async def batch_analyze(
        self, files: list[Path], prompt_template: str, max_concurrent: int = 3
    ) -> list[dict]:
        """Analyze multiple files concurrently."""
        try:
            client = genai.Client(
                vertexai=True, project=self.project, location=self.location
            )
        except ImportError:
            print("Error: google-genai not installed")
            return []

        # Create tasks for concurrent processing
        tasks = []
        for file_path in files:
            tasks.append(self.analyze_file(file_path, client, prompt_template))

        # Run with limited concurrency
        results = []
        for i in range(0, len(tasks), max_concurrent):
            batch = tasks[i : i + max_concurrent]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)

        return results


# -------------------------
# Local Code Analysis
# -------------------------


class LocalCodeAnalyzer:
    """Run local code quality analysis without SonarQube."""

    def __init__(self, target_files: list[str] | None = None):
        self.target_files = target_files or [
            "emailops/config.py",
            "emailops/doctor.py",
            "emailops/email_indexer.py",
            "emailops/llm_client.py",
            "emailops/llm_runtime.py",
            "emailops/processor.py",
            "emailops/search_and_draft.py",
            "emailops/summarize_email_thread.py",
            "emailops/utils.py",
            "emailops/validators.py",
        ]

    def check_dependencies(self) -> list[str]:
        """Check and install required analysis tools."""
        tools = {
            "pylint": "pylint",
            "flake8": "flake8",
            "bandit": "bandit",
            "radon": "radon",
            "mypy": "mypy",
        }

        missing = []
        for _tool_name, package in tools.items():
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "show", package],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    missing.append(package)
            except Exception:
                missing.append(package)

        if missing:
            print(f"Installing missing tools: {', '.join(missing)}")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", *missing],
                    check=True,
                    timeout=300,
                )
                print("✓ Tools installed successfully\n")
            except Exception as e:
                print(f"⚠ Warning: Failed to install some tools: {e}\n")

        return missing

    def run_pylint(self, files: list[Path]) -> dict[str, Any]:
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
                sys.executable,
                "-m",
                "pylint",
                "--output-format=json",
                "--disable=C0103,C0114,C0115,C0116",
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

    def run_flake8(self, files: list[Path]) -> dict[str, Any]:
        """Run Flake8 analysis."""
        print("Running Flake8 analysis...")
        result = {
            "tool": "flake8",
            "status": "success",
            "issues": [],
        }

        try:
            cmd = [
                sys.executable,
                "-m",
                "flake8",
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

    def run_bandit(self, files: list[Path]) -> dict[str, Any]:
        """Run Bandit security analysis."""
        print("Running Bandit security analysis...")
        result = {
            "tool": "bandit",
            "status": "success",
            "issues": [],
        }

        try:
            cmd = [
                sys.executable,
                "-m",
                "bandit",
                "-r",
                "-f",
                "json",
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

    def run_analysis(self) -> dict[str, Any]:
        """Run complete analysis."""
        print("=" * 60)
        print("EmailOps Local Code Quality Analysis")
        print("=" * 60)
        print()

        project_dir = Path.cwd()

        # Verify files exist
        files = [project_dir / f for f in self.target_files]
        existing_files = [f for f in files if f.exists()]
        missing_files = [f for f in files if not f.exists()]

        if missing_files:
            print(f"⚠ Warning: {len(missing_files)} files not found")

        if not existing_files:
            print("✗ No files found to analyze!")
            return {}

        print(f"Files to analyze: {len(existing_files)}")
        print()

        # Check/install dependencies
        self.check_dependencies()

        # Run analyses
        results = {
            "timestamp": datetime.now().isoformat(),
            "files": [str(f.relative_to(project_dir)) for f in existing_files],
            "files_analyzed": len(existing_files),
        }

        # Run each tool
        results["pylint"] = self.run_pylint(existing_files)
        results["flake8"] = self.run_flake8(existing_files)
        results["bandit"] = self.run_bandit(existing_files)

        # Calculate totals
        results["total_issues"] = sum(
            [
                results.get("pylint", {}).get("issue_count", 0),
                results.get("flake8", {}).get("issue_count", 0),
            ]
        )
        results["security_issues"] = results.get("bandit", {}).get("issue_count", 0)

        # Save JSON report
        json_output = Path("analysis_results.json")
        json_output.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n✓ JSON report saved to: {json_output}")

        # Print summary
        print("\n" + "=" * 60)
        print("Analysis Summary")
        print("=" * 60)
        print(
            f"Pylint Score:        {results.get('pylint', {}).get('score', 'N/A')}/10.0"
        )
        print(f"Total Issues:        {results.get('total_issues', 0)}")
        print(f"Security Issues:     {results.get('security_issues', 0)}")
        print("=" * 60)

        return results


# -------------------------
# Remediation Package Generator
# -------------------------


class RemediationPackageGenerator:
    """Generate remediation packages for EmailOps modules."""

    def __init__(self, output_dir: Path = Path("remediation_packages")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

        # Module dependencies
        self.module_dependencies = {
            "config.py": ["exceptions.py"],
            "conversation_loader.py": [
                "file_utils.py",
                "processing_utils.py",
                "text_extraction.py",
                "utils.py",
            ],
            "doctor.py": ["index_metadata.py", "llm_client.py", "config.py"],
            "email_indexer.py": [
                "index_metadata.py",
                "config.py",
                "llm_client.py",
                "text_chunker.py",
                "utils.py",
            ],
            "llm_client.py": ["llm_runtime.py"],
            "llm_runtime.py": [
                "config.py",
                "exceptions.py",
                "file_utils.py",
                "utils.py",
            ],
            "processor.py": [
                "config.py",
                "index_metadata.py",
                "search_and_draft.py",
                "summarize_email_thread.py",
            ],
            "search_and_draft.py": [
                "config.py",
                "index_metadata.py",
                "llm_client.py",
                "llm_runtime.py",
                "utils.py",
            ],
            "summarize_email_thread.py": ["llm_client.py", "utils.py"],
            "utils.py": [
                "conversation_loader.py",
                "email_processing.py",
                "file_utils.py",
            ],
            "validators.py": [],
        }

    def extract_module_issues(self, module_name: str, analysis_docs: list[Path]) -> str:
        """Extract all issues for a specific module from analysis documents."""
        issues = []

        for doc_path in analysis_docs:
            if not doc_path.exists():
                continue

            content = doc_path.read_text(encoding="utf-8")

            # Look for sections mentioning this module
            pattern1 = rf"(?:^|\n)##+ .*{re.escape(module_name)}.*?\n(.*?)(?=\n##+ |$)"
            matches1 = re.findall(pattern1, content, re.DOTALL | re.IGNORECASE)

            pattern2 = rf"(?:^|\n)((?:CRITICAL|HIGH|MEDIUM|LOW)-\d+).*?{re.escape(module_name)}.*?\n(.*?)(?=\n(?:CRITICAL|HIGH|MEDIUM|LOW)-\d+|$)"
            matches2 = re.findall(pattern2, content, re.DOTALL | re.IGNORECASE)

            for match in matches1:
                if match.strip():
                    issues.append(f"From {doc_path.name}:\n{match.strip()}\n")

            for issue_id, issue_text in matches2:
                issues.append(
                    f"[{issue_id}] From {doc_path.name}:\n{issue_text.strip()}\n"
                )

        if not issues:
            return f"# No specific issues found for {module_name}\n"

        return "\n\n".join(issues)

    def create_package(self, module_name: str) -> Path | None:
        """Create a complete remediation package for one module."""
        print(f"Creating package for {module_name}...")

        # Create package directory
        package_dir = self.output_dir / f"REMEDIATE_{module_name.replace('.py', '')}"
        package_dir.mkdir(parents=True, exist_ok=True)

        base_dir = Path.cwd()
        emailops_dir = base_dir / "emailops"
        docs_dir = base_dir / "emailops_docs"

        # Copy target module
        target_file = emailops_dir / module_name
        if target_file.exists():
            shutil.copy2(target_file, package_dir / f"TARGET_{module_name}")

        # Extract issues
        analysis_docs = [
            docs_dir / "DEEP_CRITICAL_ANALYSIS.md",
            docs_dir / "SECOND_PASS_ULTRA_DEEP_ANALYSIS.md",
            docs_dir / "COMPREHENSIVE_CRITICAL_ANALYSIS.md",
        ]

        issues_content = self.extract_module_issues(module_name, analysis_docs)
        (package_dir / f"ISSUES_{module_name.replace('.py', '.md')}").write_text(
            issues_content, encoding="utf-8"
        )

        # Copy dependent modules
        deps_dir = package_dir / "DEPENDENCIES"
        deps_dir.mkdir(exist_ok=True)

        dependencies = self.module_dependencies.get(module_name, [])
        for dep in dependencies:
            dep_file = emailops_dir / dep
            if dep_file.exists():
                shutil.copy2(dep_file, deps_dir / dep)

        # Create ZIP archive
        zip_path = self.output_dir / f"REMEDIATE_{module_name.replace('.py', '')}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _dirs, files in os.walk(package_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(package_dir)
                    zipf.write(file_path, arcname)

        print(f"✅ Created {zip_path.name}")

        # Clean up directory
        shutil.rmtree(package_dir)

        return zip_path

    def generate_all(self, modules: list[str] | None = None) -> list[Path]:
        """Generate packages for all modules."""
        if modules is None:
            modules = list(self.module_dependencies.keys())

        print("=" * 80)
        print("EMAILOPS REMEDIATION PACKAGE GENERATOR")
        print("=" * 80)
        print()
        print(f"Generating {len(modules)} remediation packages...")
        print()

        created_packages = []

        for i, module in enumerate(modules, 1):
            print(f"[{i}/{len(modules)}] Processing {module}...")
            try:
                zip_path = self.create_package(module)
                if zip_path:
                    created_packages.append(zip_path)
            except Exception as e:
                print(f"❌ Failed to create package for {module}: {e}")

        print()
        print("=" * 80)
        print(f"✅ COMPLETE: Generated {len(created_packages)}/{len(modules)} packages")
        print("=" * 80)
        print()
        print(f"Output directory: {self.output_dir.absolute()}")

        return created_packages


# -------------------------
# Main CLI
# -------------------------


def main():
    """Main entry point with command-line interface."""

    parser = argparse.ArgumentParser(
        description="Code analysis and package generation utilities"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Batch analysis command
    batch_parser = subparsers.add_parser("batch", help="Batch analyze files with GenAI")
    batch_parser.add_argument("--files", nargs="+", help="Files to analyze")
    batch_parser.add_argument(
        "--prompt",
        help="Prompt template",
        default="""
As a Senior Software Engineer, please review the following code for quality, potential bugs, and adherence to best practices.
Provide a detailed analysis and suggest specific improvements.

Here is the code:
---
{file_content}
---
""",
    )

    # Local analysis command
    subparsers.add_parser("local", help="Run local code analysis")

    # Remediation command
    remediate_parser = subparsers.add_parser(
        "remediate", help="Generate remediation packages"
    )
    remediate_parser.add_argument(
        "--modules", nargs="+", help="Specific modules to process"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "batch":
        if not args.files:
            print("Error: --files required for batch analysis")
            return 1

        files = [Path(f) for f in args.files]
        analyzer = BatchAnalyzer()

        async def run_batch():
            return await analyzer.batch_analyze(files, args.prompt)

        results = asyncio.run(run_batch())

        # Print summary
        success_count = sum(1 for r in results if r["success"])
        print(f"\n✓ Analyzed {success_count}/{len(results)} files successfully")

    elif args.command == "local":
        analyzer = LocalCodeAnalyzer()
        analyzer.run_analysis()

    elif args.command == "remediate":
        generator = RemediationPackageGenerator()
        generator.generate_all(modules=args.modules)

    return 0


if __name__ == "__main__":
    sys.exit(main())
