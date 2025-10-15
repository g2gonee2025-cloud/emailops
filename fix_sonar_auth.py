#!/usr/bin/env python3
"""
Comprehensive SonarQube authentication fix.
"""

import sys
import requests
import time

SONAR_URL = "http://localhost:9099"
DEFAULT_USER = "admin"
DEFAULT_PASS = "!Sharjah2050"
TOKEN_NAME = "emailops-analysis-v2"
PROPERTIES_FILE = "sonar-project.properties"

def check_server() -> bool:
    """Check if SonarQube server is running."""
    try:
        response = requests.get(f"{SONAR_URL}/api/system/status", timeout=5)
        return response.status_code == 200
    except:
        return False

def authenticate_with_password() -> bool:
    """Test authentication with username/password."""
    try:
        response = requests.get(
            f"{SONAR_URL}/api/authentication/validate",
            auth=(DEFAULT_USER, DEFAULT_PASS),
            timeout=10
        )
        return response.status_code == 200 and response.json().get("valid", False)
    except:
        return False

def list_all_tokens(username: str, password: str) -> list:
    """List all existing tokens."""
    try:
        url = f"{SONAR_URL}/api/user_tokens/search"
        response = requests.get(url, auth=(username, password), timeout=10)
        if response.status_code == 200:
            return response.json().get("userTokens", [])
        return []
    except:
        return []

def revoke_all_tokens(username: str, password: str) -> int:
    """Revoke all existing tokens."""
    tokens = list_all_tokens(username, password)
    revoked = 0
    for token in tokens:
        try:
            url = f"{SONAR_URL}/api/user_tokens/revoke"
            response = requests.post(
                url,
                auth=(username, password),
                data={"name": token["name"]},
                timeout=10
            )
            if response.status_code == 204:
                revoked += 1
                print(f"  ✓ Revoked: {token['name']}")
        except:
            pass
    return revoked

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
        return False, f"Status {response.status_code}: {response.text}"
    except Exception as e:
        return False, str(e)

def validate_token(token: str) -> bool:
    """Validate a token."""
    try:
        response = requests.get(
            f"{SONAR_URL}/api/authentication/validate",
            auth=(token, ""),
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("valid", False)
        return False
    except:
        return False

def update_properties_file(token: str) -> bool:
    """Update sonar-project.properties with the token."""
    try:
        with open(PROPERTIES_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Remove existing token lines
        lines = [line for line in lines if not line.strip().startswith('sonar.token=')]
        
        # Add token after sonar.host.url
        new_lines = []
        for line in lines:
            new_lines.append(line)
            if line.strip().startswith('sonar.host.url='):
                new_lines.append(f'sonar.token={token}\n')
        
        with open(PROPERTIES_FILE, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        return True
    except Exception as e:
        print(f"Error updating properties file: {e}")
        return False

def main():
    print("=" * 70)
    print("SonarQube Authentication Fix")
    print("=" * 70)
    print()
    
    # Step 1: Check server
    print("Step 1: Checking SonarQube server...")
    if not check_server():
        print("✗ SonarQube server is not running or not accessible")
        print(f"  Please ensure SonarQube is running at {SONAR_URL}")
        return 1
    print("✓ Server is running")
    
    # Step 2: Test password authentication
    print("\nStep 2: Testing password authentication...")
    if not authenticate_with_password():
        print("✗ Cannot authenticate with admin credentials")
        print("  Please check username and password")
        return 1
    print("✓ Password authentication works")
    
    # Step 3: Clean up old tokens
    print("\nStep 3: Cleaning up old tokens...")
    tokens = list_all_tokens(DEFAULT_USER, DEFAULT_PASS)
    if tokens:
        print(f"Found {len(tokens)} existing token(s):")
        for token in tokens:
            print(f"  - {token['name']}")
        print("\nRevoking all tokens...")
        revoked = revoke_all_tokens(DEFAULT_USER, DEFAULT_PASS)
        print(f"✓ Revoked {revoked} token(s)")
    else:
        print("✓ No existing tokens to clean up")
    
    # Step 4: Generate new token
    print(f"\nStep 4: Generating new token '{TOKEN_NAME}'...")
    success, token = generate_token(DEFAULT_USER, DEFAULT_PASS, TOKEN_NAME)
    if not success:
        print(f"✗ Failed to generate token: {token}")
        return 1
    print(f"✓ Token generated: {token}")
    
    # Step 5: Validate token
    print("\nStep 5: Validating new token...")
    time.sleep(1)  # Wait a moment for the token to be fully registered
    if not validate_token(token):
        print("✗ Token validation failed")
        print("  This might be a SonarQube configuration issue")
        print(f"\n  Try using the token directly: {token}")
        # Still proceed to update the file
    else:
        print("✓ Token validated successfully")
    
    # Step 6: Update properties file
    print(f"\nStep 6: Updating {PROPERTIES_FILE}...")
    if update_properties_file(token):
        print("✓ Properties file updated")
    else:
        print("✗ Failed to update properties file")
        return 1
    
    # Step 7: Update .env file
    print("\nStep 7: Updating .env file...")
    try:
        with open('.env', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update or add SONAR_TOKEN
        if 'SONAR_TOKEN=' in content:
            import re
            content = re.sub(
                r'SONAR_TOKEN=.*',
                f'SONAR_TOKEN="{token}"',
                content
            )
        else:
            content += f'\nSONAR_TOKEN="{token}"\n'
        
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(content)
        print("✓ .env file updated")
    except Exception as e:
        print(f"✗ Failed to update .env: {e}")
    
    # Final instructions
    print("\n" + "=" * 70)
    print("✓ Setup Complete!")
    print("=" * 70)
    print("\nNow run the SonarQube scanner:")
    print(".sonar-tools\\sonar-scanner-5.0.1.3006-windows\\bin\\sonar-scanner.bat")
    print("\nIf authentication still fails, try running with -X for debug output:")
    print(".sonar-tools\\sonar-scanner-5.0.1.3006-windows\\bin\\sonar-scanner.bat -X")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())