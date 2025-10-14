#!/usr/bin/env python3
"""
Helper script to generate SonarQube authentication token.
"""

import sys
import requests
import json

SONAR_URL = "http://localhost:9099"
DEFAULT_USER = "admin"
DEFAULT_PASS = "admin"

def generate_token(username: str, password: str, token_name: str = "emailops-analysis") -> tuple[bool, str]:
    """Generate a new token for SonarQube."""
    try:
        # Try to generate token
        url = f"{SONAR_URL}/api/user_tokens/generate"
        response = requests.post(
            url,
            auth=(username, password),
            data={"name": token_name},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            token = data.get("token")
            if token:
                return True, token
            return False, "No token in response"
        elif response.status_code == 401:
            return False, "Invalid credentials"
        else:
            return False, f"HTTP {response.status_code}: {response.text[:200]}"
    
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to SonarQube server"
    except Exception as e:
        return False, str(e)

def check_existing_token() -> str | None:
    """Check if token is already set in environment."""
    import os
    return os.getenv("SONAR_TOKEN")

def main() -> int:
    """Main entry point."""
    print("="*60)
    print("SonarQube Token Setup")
    print("="*60)
    print()
    
    # Check if token already exists
    existing = check_existing_token()
    if existing:
        print(f"✓ SONAR_TOKEN is already set: {existing[:10]}...")
        print("\nTo use this token, run:")
        print("  .sonar-tools\\sonar-scanner-5.0.1.3006-windows\\bin\\sonar-scanner.bat")
        return 0
    
    print(f"SonarQube Server: {SONAR_URL}")
    print()
    print("Options:")
    print("  1. Generate new token (requires admin credentials)")
    print("  2. Enter existing token manually")
    print("  3. Exit")
    print()
    
    choice = input("Choose option (1-3): ").strip()
    
    if choice == "1":
        print("\nDefault credentials: admin/admin")
        username = input(f"Username [{DEFAULT_USER}]: ").strip() or DEFAULT_USER
        password = input(f"Password [{DEFAULT_PASS}]: ").strip() or DEFAULT_PASS
        
        print("\nGenerating token...")
        success, result = generate_token(username, password)
        
        if success:
            print(f"\n✓ Token generated successfully!")
            print(f"\nToken: {result}")
            print("\n" + "="*60)
            print("Set this token in your environment:")
            print("="*60)
            print(f"\nPowerShell:")
            print(f'  $env:SONAR_TOKEN="{result}"')
            print(f"\nCmd:")
            print(f'  set SONAR_TOKEN={result}')
            print("\nThen run the analysis:")
            print("  .sonar-tools\\sonar-scanner-5.0.1.3006-windows\\bin\\sonar-scanner.bat")
            print()
            return 0
        else:
            print(f"\n✗ Failed to generate token: {result}")
            return 1
    
    elif choice == "2":
        token = input("\nEnter your SonarQube token: ").strip()
        if token:
            print("\n" + "="*60)
            print("Set this token in your environment:")
            print("="*60)
            print(f"\nPowerShell:")
            print(f'  $env:SONAR_TOKEN="{token}"')
            print(f"\nCmd:")
            print(f'  set SONAR_TOKEN={token}')
            print("\nThen run the analysis:")
            print("  .sonar-tools\\sonar-scanner-5.0.1.3006-windows\\bin\\sonar-scanner.bat")
            return 0
        else:
            print("✗ No token provided")
            return 1
    
    else:
        print("Exiting...")
        return 0

if __name__ == "__main__":
    sys.exit(main())