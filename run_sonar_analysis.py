#!/usr/bin/env python3
"""
Run SonarQube analysis on specified EmailOps files.
This script will download SonarQube Scanner if needed and execute the analysis.
"""

import os
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# SonarQube Scanner configuration
SCANNER_VERSION = "5.0.1.3006"
SCANNER_ZIP_URL = f"https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-{SCANNER_VERSION}-windows.zip"
SCANNER_DIR_NAME = f"sonar-scanner-{SCANNER_VERSION}-windows"

def download_scanner(target_dir: Path) -> Path:
    """Download and extract SonarQube Scanner."""
    scanner_zip = target_dir / f"sonar-scanner-{SCANNER_VERSION}.zip"
    scanner_extracted = target_dir / SCANNER_DIR_NAME
    
    if scanner_extracted.exists():
        print(f"✓ Scanner already exists at: {scanner_extracted}")
        return scanner_extracted
    
    print(f"Downloading SonarQube Scanner {SCANNER_VERSION}...")
    try:
        urlretrieve(SCANNER_ZIP_URL, scanner_zip)
        print(f"✓ Downloaded to: {scanner_zip}")
        
        print("Extracting scanner...")
        with zipfile.ZipFile(scanner_zip, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print(f"✓ Extracted to: {scanner_extracted}")
        
        # Clean up zip file
        scanner_zip.unlink()
        
        return scanner_extracted
    except Exception as e:
        print(f"✗ Failed to download/extract scanner: {e}")
        sys.exit(1)

def run_sonar_analysis(scanner_path: Path, project_dir: Path) -> int:
    """Execute SonarQube analysis."""
    scanner_bin = scanner_path / "bin" / "sonar-scanner.bat"
    
    if not scanner_bin.exists():
        print(f"✗ Scanner binary not found at: {scanner_bin}")
        return 1
    
    # Check if sonar-project.properties exists
    props_file = project_dir / "sonar-project.properties"
    if not props_file.exists():
        print(f"✗ sonar-project.properties not found at: {props_file}")
        return 1
    
    print("\n" + "="*60)
    print("Running SonarQube Analysis (Force Full Scan)")
    print("="*60 + "\n")
    
    # Build command with force scan flags
    cmd = [
        str(scanner_bin),
        f"-Dproject.settings={props_file}",
        "-Dsonar.scanAllFiles=true",
        "-Dsonar.qualitygate.wait=false",
    ]
    
    # Check for SonarQube server configuration
    sonar_host = os.getenv("SONAR_HOST_URL", "http://localhost:9000")
    sonar_token = os.getenv("SONAR_TOKEN", "")
    
    if sonar_token:
        cmd.append(f"-Dsonar.token={sonar_token}")
    else:
        print("⚠ Warning: SONAR_TOKEN not set. Analysis may fail without authentication.")
        print("  Set SONAR_TOKEN environment variable or use SONAR_LOGIN")
    
    cmd.append(f"-Dsonar.host.url={sonar_host}")
    
    print(f"Scanner: {scanner_bin}")
    print(f"Project: {project_dir}")
    print(f"Server: {sonar_host}")
    print(f"\nExecuting command...")
    print(f"  {' '.join(cmd[:2])}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=project_dir,
            capture_output=False,
            text=True,
            check=False
        )
        return result.returncode
    except Exception as e:
        print(f"✗ Failed to execute scanner: {e}")
        return 1

def main():
    """Main entry point."""
    project_dir = Path(__file__).parent.resolve()
    tools_dir = project_dir / ".sonar-tools"
    tools_dir.mkdir(exist_ok=True)
    
    print("EmailOps SonarQube Analysis Runner")
    print("=" * 60)
    print(f"Project directory: {project_dir}")
    print(f"Tools directory: {tools_dir}")
    print()
    
    # Check for SonarQube server
    sonar_host = os.getenv("SONAR_HOST_URL", "http://localhost:9000")
    print(f"SonarQube Server: {sonar_host}")
    
    if not os.getenv("SONAR_TOKEN"):
        print("\n⚠ WARNING: SONAR_TOKEN environment variable is not set!")
        print("  You may need to set this for authentication with SonarQube server.")
        print("  Example: set SONAR_TOKEN=your_token_here")
        print("\nContinue anyway? (y/n): ", end="")
        
        response = input().strip().lower()
        if response != 'y':
            print("Aborted.")
            return 1
    
    # Download scanner if needed
    scanner_path = download_scanner(tools_dir)
    
    # Run analysis
    exit_code = run_sonar_analysis(scanner_path, project_dir)
    
    if exit_code == 0:
        print("\n" + "="*60)
        print("✓ SonarQube analysis completed successfully!")
        print("="*60)
        print(f"\nView results at: {sonar_host}/dashboard?id=emailops_vertex_ai")
    else:
        print("\n" + "="*60)
        print(f"✗ SonarQube analysis failed with exit code: {exit_code}")
        print("="*60)
        print("\nCommon issues:")
        print("  1. SonarQube server not running")
        print("  2. Invalid SONAR_TOKEN")
        print("  3. Network connectivity issues")
        print("  4. Project key already exists with different settings")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())