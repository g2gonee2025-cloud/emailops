#!/usr/bin/env python3
"""
Wait for SonarQube server to be ready, then run analysis.
"""

import subprocess
import sys
import time
from pathlib import Path

def check_server_status() -> bool:
    """Check if SonarQube server is ready."""
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:9000/api/system/status"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and "UP" in result.stdout:
            return True
    except Exception:
        pass
    return False

def wait_for_server(max_wait: int = 180) -> bool:
    """Wait for SonarQube server to be ready."""
    print("Waiting for SonarQube server to be ready...")
    print("(This typically takes 60-120 seconds on first start)")
    print()
    
    start_time = time.time()
    dots = 0
    
    while time.time() - start_time < max_wait:
        if check_server_status():
            print("\n✓ SonarQube server is ready!")
            return True
        
        # Print progress dots
        print(".", end="", flush=True)
        dots += 1
        if dots % 60 == 0:
            elapsed = int(time.time() - start_time)
            print(f" [{elapsed}s]")
        
        time.sleep(1)
    
    print(f"\n✗ Server did not become ready within {max_wait} seconds")
    return False

def run_analysis() -> int:
    """Run SonarQube analysis."""
    scanner_path = Path(".sonar-tools/sonar-scanner-5.0.1.3006-windows/bin/sonar-scanner.bat")
    
    if not scanner_path.exists():
        print(f"✗ Scanner not found at: {scanner_path}")
        print("Run 'python run_sonar_analysis.py' first to download the scanner.")
        return 1
    
    print("\n" + "="*60)
    print("Starting SonarQube Analysis")
    print("="*60)
    print()
    
    cmd = [
        str(scanner_path.resolve()),
        "-Dsonar.scanAllFiles=true",
        "-Dsonar.qualitygate.wait=false",
    ]
    
    try:
        result = subprocess.run(cmd, cwd=Path.cwd())
        return result.returncode
    except Exception as e:
        print(f"✗ Failed to run analysis: {e}")
        return 1

def main() -> int:
    """Main entry point."""
    print("="*60)
    print("SonarQube Analysis - Wait for Server and Run")
    print("="*60)
    print()
    
    # Check if server is already ready
    if check_server_status():
        print("✓ SonarQube server is already running!")
    else:
        print("SonarQube server is not ready yet.")
        print("Make sure the server is starting in another terminal.")
        print()
        
        if not wait_for_server(max_wait=180):
            print("\n✗ Server failed to start. Please check:")
            print("  1. Is StartSonar.bat running?")
            print("  2. Check logs in: ../sonarqube-*/logs/sonar.log")
            print("  3. Ensure Java is installed (Java 17+ required)")
            return 1
    
    # Server is ready, run analysis
    print()
    return run_analysis()

if __name__ == "__main__":
    sys.exit(main())