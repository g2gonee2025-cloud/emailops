#!/usr/bin/env python3
"""
Retrieve or create SonarQube authentication token.
"""

import sys
import requests

SONAR_URL = "http://localhost:9099"
DEFAULT_USER = "admin"
DEFAULT_PASS = "!Sharjah2050"

def list_tokens(username: str, password: str) -> tuple[bool, list]:
    """List existing tokens."""
    try:
        url = f"{SONAR_URL}/api/user_tokens/search"
        response = requests.get(
            url,
            auth=(username, password),
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            tokens = data.get("userTokens", [])
            return True, tokens
        else:
            return False, []
    except Exception as e:
        return False, []

def revoke_token(username: str, password: str, token_name: str) -> bool:
    """Revoke an existing token."""
    try:
        url = f"{SONAR_URL}/api/user_tokens/revoke"
        response = requests.post(
            url,
            auth=(username, password),
            data={"name": token_name},
            timeout=10
        )
        return response.status_code == 204
    except:
        return False

def generate_token(username: str, password: str, token_name: str) -> tuple[bool, str]:
    """Generate a new token."""
    try:
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
        return False, response.text
    except Exception as e:
        return False, str(e)

def main():
    username = DEFAULT_USER
    password = DEFAULT_PASS
    
    print("=" * 70)
    print("SonarQube Token Retrieval")
    print("=" * 70)
    print()
    
    # List existing tokens
    print("Checking for existing tokens...")
    success, tokens = list_tokens(username, password)
    
    if success and tokens:
        print(f"\nFound {len(tokens)} existing token(s):")
        for token in tokens:
            print(f"  - {token['name']} (created: {token.get('createdAt', 'unknown')})")
        
        # Check if emailops-analysis exists
        existing_names = [t['name'] for t in tokens]
        if 'emailops-analysis' in existing_names:
            print("\n'emailops-analysis' token exists.")
            print("Options:")
            print("  1. Revoke and regenerate 'emailops-analysis'")
            print("  2. Create new token with different name")
            print("  3. Exit (you need to retrieve token from SonarQube UI)")
            
            choice = input("\nChoose option (1-3): ").strip()
            
            if choice == "1":
                print("\nRevoking existing token...")
                if revoke_token(username, password, "emailops-analysis"):
                    print("✓ Token revoked")
                    print("\nGenerating new token...")
                    success, token = generate_token(username, password, "emailops-analysis")
                    if success:
                        print(f"\n✓ New token generated: {token}")
                        print("\n" + "=" * 70)
                        print("Add token to sonar-project.properties:")
                        print("=" * 70)
                        print(f"\nsonar.token={token}")
                        print("\nOr set as environment variable in PowerShell:")
                        print(f'$env:SONAR_TOKEN="{token}"')
                        print("\nThen run:")
                        print(".sonar-tools\\sonar-scanner-5.0.1.3006-windows\\bin\\sonar-scanner.bat")
                        return 0
                    else:
                        print(f"✗ Failed to generate: {token}")
                        return 1
                else:
                    print("✗ Failed to revoke token")
                    return 1
            
            elif choice == "2":
                token_name = input("\nEnter new token name: ").strip()
                if token_name:
                    print(f"\nGenerating token '{token_name}'...")
                    success, token = generate_token(username, password, token_name)
                    if success:
                        print(f"\n✓ Token generated: {token}")
                        print("\n" + "=" * 70)
                        print("Add token to sonar-project.properties:")
                        print("=" * 70)
                        print(f"\nsonar.token={token}")
                        print("\nOr set as environment variable in PowerShell:")
                        print(f'$env:SONAR_TOKEN="{token}"')
                        print("\nThen run:")
                        print(".sonar-tools\\sonar-scanner-5.0.1.3006-windows\\bin\\sonar-scanner.bat")
                        return 0
                    else:
                        print(f"✗ Failed: {token}")
                        return 1
            else:
                print("\nTo retrieve token manually:")
                print(f"1. Go to {SONAR_URL}")
                print("2. Login with admin credentials")
                print("3. Go to: My Account > Security > Tokens")
                print("4. Copy existing token or generate new one")
                return 0
    else:
        print("\nNo existing tokens found. Creating new one...")
        success, token = generate_token(username, password, "emailops-analysis-2")
        if success:
            print(f"\n✓ Token generated: {token}")
            print("\n" + "=" * 70)
            print("Add token to sonar-project.properties:")
            print("=" * 70)
            print(f"\nsonar.token={token}")
            print("\nOr set as environment variable in PowerShell:")
            print(f'$env:SONAR_TOKEN="{token}"')
            print("\nThen run:")
            print(".sonar-tools\\sonar-scanner-5.0.1.3006-windows\\bin\\sonar-scanner.bat")
            return 0
        else:
            print(f"✗ Failed: {token}")
            return 1

if __name__ == "__main__":
    sys.exit(main())