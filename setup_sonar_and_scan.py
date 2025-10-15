#!/usr/bin/env python3
"""
Automated script to setup SonarQube token and run scan.
"""

import subprocess
import sys
import requests

SONAR_URL = "http://localhost:9099"
USERNAME = "admin"
PASSWORD = "!Sharjah2050"

def revoke_and_create_token(token_name: str = "emailops-scan") -> str | None:
    """Revoke existing token and create new one."""
    try:
        # First, try to revoke existing token (ignore if it doesn't exist)
        revoke_url = f"{SONAR_URL}/api/user_tokens/revoke"
        requests.post(
            revoke_url,
            auth=(USERNAME, PASSWORD),
            data={"name": token_name},
            timeout=10
        )
        
        # Now create new token
        gen_url = f"{SONAR_URL}/api/user_tokens/generate"
        response = requests.post(
            gen_url,
            auth=(USERNAME, PASSWORD),
            data={"name": token_name},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("token")
        else:
            print(f"Failed to generate token: HTTP {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None

def run_sonar_scan(token: str) -> bool:
    """Run SonarQube scanner with token."""
    try:
        scanner_path = r".sonar-tools\sonar-scanner-5.0.1.3006-windows\bin\sonar-scanner.bat"
        
        # Run scanner with token as parameter
        result = subprocess.run(
            [scanner_path, f"-Dsonar.token={token}"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("Scanner timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"Error running scanner: {e}")
        return False

def main():
    print("=" * 70)
    print("SonarQube Automated Setup and Scan")
    print("=" * 70)
    print()
    
    # Step 1: Get token
    print("Step 1: Getting authentication token...")
    token = revoke_and_create_token()
    
    if not token:
        print("\n✗ Failed to get authentication token")
        print("\nManual steps:")
        print(f"1. Go to {SONAR_URL}")
        print("2. Login with admin credentials")
        print("3. Go to: My Account > Security > Tokens")
        print("4. Generate a token and run:")
        print('   .sonar-tools\\sonar-scanner-5.0.1.3006-windows\\bin\\sonar-scanner.bat -Dsonar.token=YOUR_TOKEN')
        return 1
    
    print(f"✓ Token obtained: {token[:10]}...")
    
    # Step 2: Run scan
    print("\nStep 2: Running SonarQube scan...")
    print("-" * 70)
    
    if run_sonar_scan(token):
        print("\n" + "=" * 70)
        print("✓ Scan completed successfully!")
        print("=" * 70)
        print(f"\nView results at: {SONAR_URL}/dashboard?id=emailops_vertex_ai")
        return 0
    else:
        print("\n✗ Scan failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())