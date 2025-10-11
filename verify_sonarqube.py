#!/usr/bin/env python3
"""Verify SonarQube accessibility and status."""

import requests
import json
import time
from datetime import datetime

def check_sonarqube_health():
    """Check if SonarQube is healthy and ready."""
    base_url = "http://localhost:9000"
    results = {
        "web_ui_accessible": False,
        "api_accessible": False,
        "system_status": None,
        "version": None,
        "database_status": None,
        "errors": []
    }
    
    # Check web UI accessibility
    try:
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            results["web_ui_accessible"] = True
            print(f"✓ SonarQube Web UI is accessible at {base_url}")
        else:
            results["errors"].append(f"Web UI returned status code: {response.status_code}")
    except Exception as e:
        results["errors"].append(f"Web UI connection error: {str(e)}")
        print(f"✗ Cannot connect to SonarQube Web UI: {e}")
    
    # Check API health endpoint
    try:
        health_response = requests.get(f"{base_url}/api/system/health", timeout=10)
        if health_response.status_code == 200:
            results["api_accessible"] = True
            health_data = health_response.json()
            results["system_status"] = health_data.get("health", "Unknown")
            print(f"✓ SonarQube API is accessible")
            print(f"  System Health: {results['system_status']}")
        else:
            results["errors"].append(f"API health check returned: {health_response.status_code}")
    except Exception as e:
        results["errors"].append(f"API health check error: {str(e)}")
    
    # Check system status
    try:
        status_response = requests.get(f"{base_url}/api/system/status", timeout=10)
        if status_response.status_code == 200:
            status_data = status_response.json()
            results["system_status"] = status_data.get("status", "Unknown")
            results["version"] = status_data.get("version", "Unknown")
            print(f"✓ System Status: {results['system_status']}")
            print(f"  Version: {results['version']}")
        else:
            results["errors"].append(f"System status check returned: {status_response.status_code}")
    except Exception as e:
        results["errors"].append(f"System status error: {str(e)}")
    
    # Check database migration status
    try:
        db_response = requests.get(f"{base_url}/api/system/db_migration_status", timeout=10)
        if db_response.status_code == 200:
            db_data = db_response.json()
            results["database_status"] = db_data.get("state", "Unknown")
            print(f"✓ Database Status: {results['database_status']}")
        else:
            results["errors"].append(f"Database status check returned: {db_response.status_code}")
    except Exception as e:
        results["errors"].append(f"Database status error: {str(e)}")
    
    return results

def wait_for_sonarqube(max_wait=60):
    """Wait for SonarQube to be fully ready."""
    print(f"Waiting for SonarQube to be ready (max {max_wait} seconds)...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get("http://localhost:9000/api/system/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "UP":
                    print("✓ SonarQube is ready!")
                    return True
                else:
                    print(f"  Status: {data.get('status', 'Unknown')} - waiting...")
        except:
            print("  Not ready yet - waiting...")
        
        time.sleep(5)
    
    return False

def check_docker_container():
    """Check if SonarQube Docker container is running."""
    import subprocess
    
    try:
        result = subprocess.run(
            ["wsl", "docker", "ps", "--filter", "name=sonarqube", "--format", "{{.Names}}\t{{.Status}}"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.startswith('sonarqube'):
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        return True, parts[1]
            return False, "Container not found"
        else:
            return False, "Failed to check container status"
    except Exception as e:
        return False, str(e)

def main():
    """Main verification function."""
    print("=" * 60)
    print("SonarQube Verification")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    
    # Check Docker container status
    print("Checking Docker container status...")
    print("-" * 60)
    container_running, container_status = check_docker_container()
    
    if container_running:
        print(f"✓ SonarQube container is running")
        print(f"  Status: {container_status}")
    else:
        print(f"✗ SonarQube container issue: {container_status}")
    
    print()
    
    # Wait for SonarQube to be ready
    if container_running:
        if not wait_for_sonarqube():
            print("⚠️  SonarQube is taking longer than expected to start")
    
    print()
    
    # Check SonarQube health
    print("Checking SonarQube services...")
    print("-" * 60)
    results = check_sonarqube_health()
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Docker Container: {'✓ RUNNING' if container_running else '✗ NOT RUNNING'}")
    print(f"Web UI Accessible: {'✓ YES' if results['web_ui_accessible'] else '✗ NO'}")
    print(f"API Accessible: {'✓ YES' if results['api_accessible'] else '✗ NO'}")
    print(f"System Status: {results['system_status']}")
    print(f"Version: {results['version']}")
    print(f"Database Status: {results['database_status']}")
    
    # Overall status
    all_good = (
        container_running and 
        results['web_ui_accessible'] and 
        results['api_accessible'] and
        results['system_status'] in ['UP', 'GREEN']
    )
    
    print()
    print(f"Overall Status: {'✓ READY FOR USE' if all_good else '✗ ISSUES DETECTED'}")
    
    if results['errors']:
        print("\nErrors encountered:")
        for error in results['errors']:
            print(f"  - {error}")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "container_status": {
            "running": container_running,
            "status": container_status
        },
        "sonarqube_health": results,
        "overall_status": "READY" if all_good else "ISSUES_DETECTED"
    }
    
    with open('sonarqube_verification_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to: sonarqube_verification_results.json")
    
    # Additional info
    if all_good:
        print("\n✓ SonarQube is ready for code analysis!")
        print(f"  Access the web interface at: http://localhost:9000")
        print(f"  Default credentials: admin/admin (change after first login)")
    
if __name__ == "__main__":
    main()