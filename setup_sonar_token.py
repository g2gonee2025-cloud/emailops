#!/usr/bin/env python3
"""
Automated SonarQube token setup and configuration.
"""

import sys
import requests
import re

SONAR_URL = "http://localhost:9099"
DEFAULT_USER = "admin"
DEFAULT_PASS = "!Sharjah2050"
TOKEN_NAME = "emailops-analysis"
PROPERTIES_FILE = "sonar-project.properties"

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

def update_properties_file(token: str) -> bool:
    """Update sonar-project.properties with the token."""
    try:
        with open(PROPERTIES_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if token already exists
        if re.search(r'^sonar\.token=', content, re.MULTILINE):
            # Replace existing token
            content = re.sub(
                r'^sonar\.token=.*$',
                f'sonar.token={token}',
                content,
                flags=re.MULTILINE
            )
        else:
            # Add token after host.url
            content = re.sub(
                r'(sonar\.host\.url=.*\n)',
                f'\\1sonar.token={token}\n',
                content
            )
        
        with open(PROPERTIES_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    except Exception as e:
        print(f"Error updating properties file: {e}")
        return False

def main():
    username = DEFAULT_USER
    password = DEFAULT_PASS
    
    print("=" * 70)
    print("SonarQube Token Automated Setup")
    print("=" * 70)
    print()
    
    # Step 1: Revoke existing token if it exists
    print(f"Step 1: Revoking existing '{TOKEN_NAME}' token (if exists)...")
    revoke_token(username, password, TOKEN_NAME)  # Don't care if it fails
    print("✓ Done")
    
    # Step 2: Generate new token
    print(f"\nStep 2: Generating new '{TOKEN_NAME}' token...")
    success, token = generate_token(username, password, TOKEN_NAME)
    
    if not success:
        print(f"✗ Failed to generate token: {token}")
        print("\nPlease check:")
        print(f"1. SonarQube is running at {SONAR_URL}")
        print(f"2. Admin credentials are correct")
        return 1
    
    print(f"✓ Token generated: {token}")
    
    # Step 3: Update properties file
    print(f"\nStep 3: Updating {PROPERTIES_FILE}...")
    if update_properties_file(token):
        print("✓ Properties file updated")
    else:
        print("✗ Failed to update properties file")
        print("\nManually add this line to sonar-project.properties:")
        print(f"sonar.token={token}")
        return 1
    
    # Step 4: Instructions
    print("\n" + "=" * 70)
    print("✓ Setup Complete!")
    print("=" * 70)
    print("\nYou can now run the SonarQube scanner:")
    print(".sonar-tools\\sonar-scanner-5.0.1.3006-windows\\bin\\sonar-scanner.bat")
    print("\nOr use the run script:")
    print("python run_sonar_analysis.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
