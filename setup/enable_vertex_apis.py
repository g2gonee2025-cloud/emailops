#!/usr/bin/env python3
"""
Unified script to enable Vertex AI API for all projects

Consolidates functionality from:
- enable_apis_batch.py
- enable_vertex_api_all_projects.py
- enable_apis_all_projects.sh
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Colors
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(text: str, color: str = Colors.ENDC):
    """Print colored text"""
    print(f"{color}{text}{Colors.ENDC}")

def run_command(cmd: str) -> Tuple[bool, str, str]:
    """Run command and return (success, stdout, stderr)"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def load_accounts() -> List[Dict]:
    """Load account configurations"""
    # First try validated_accounts.json
    validated_file = Path("validated_accounts.json")
    if validated_file.exists():
        try:
            with open(validated_file, 'r') as f:
                data = json.load(f)
                return data.get("accounts", [])
        except:
            pass
    
    # Default accounts
    return [
        {"project_id": "api-agent-470921", "credentials_path": "secrets/api-agent-470921-4e2065b2ecf9.json"},
        {"project_id": "apt-arcana-470409-i7", "credentials_path": "secrets/apt-arcana-470409-i7-ce42b76061bf.json"},
        {"project_id": "crafty-airfoil-474021-s2", "credentials_path": "secrets/crafty-airfoil-474021-s2-34159960925b.json"},
        {"project_id": "embed2-474114", "credentials_path": "secrets/embed2-474114-fca38b4d2068.json"},
        {"project_id": "my-project-31635v", "credentials_path": "secrets/my-project-31635v-8ec357ac35b2.json"},
        {"project_id": "semiotic-nexus-470620-f3", "credentials_path": "secrets/semiotic-nexus-470620-f3-3240cfaf6036.json"}
    ]

def check_gcloud() -> bool:
    """Check if gcloud CLI is installed"""
    success, output, _ = run_command("gcloud --version")
    return success

def enable_api_for_project(project_id: str, credentials_path: Optional[str] = None) -> bool:
    """Enable Vertex AI API for a specific project"""
    print(f"\n{Colors.BLUE}Processing project: {project_id}{Colors.ENDC}")
    
    # Authenticate with service account if provided
    if credentials_path and Path(credentials_path).exists():
        auth_cmd = f'gcloud auth activate-service-account --key-file="{credentials_path}"'
        print(f"  Authenticating with service account...")
        success, _, error = run_command(auth_cmd)
        if not success:
            print(f"{Colors.YELLOW}  Warning: Could not authenticate with service account{Colors.ENDC}")
    
    # Set the project
    cmd = f"gcloud config set project {project_id}"
    print(f"  Setting project...")
    success, _, error = run_command(cmd)
    if not success:
        print(f"{Colors.RED}  Failed to set project: {error}{Colors.ENDC}")
        return False
    
    # Enable the API
    cmd = f"gcloud services enable aiplatform.googleapis.com --project={project_id}"
    print(f"  Enabling Vertex AI API...")
    success, output, error = run_command(cmd)
    
    if success or "already enabled" in str(output).lower() or "already enabled" in str(error).lower():
        print(f"{Colors.GREEN}  ✓ API enabled successfully!{Colors.ENDC}")
        return True
    else:
        print(f"{Colors.RED}  ✗ Failed to enable API: {error}{Colors.ENDC}")
        return False

def generate_manual_urls(accounts: List[Dict]) -> None:
    """Generate manual enablement URLs for all projects"""
    print(f"\n{Colors.BOLD}Manual Enable URLs:{Colors.ENDC}")
    print("Visit these URLs to enable the API manually in the Google Cloud Console:\n")
    
    for account in accounts:
        project_id = account['project_id']
        url = f"https://console.cloud.google.com/apis/library/aiplatform.googleapis.com?project={project_id}"
        print(f"{Colors.BLUE}{project_id}:{Colors.ENDC}")
        print(f"  {url}\n")

def main():
    """Main function"""
    print_colored("="*80, Colors.BOLD)
    print_colored("VERTEX AI API ENABLER", Colors.BOLD)
    print_colored("="*80, Colors.BOLD)
    print(f"Timestamp: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    
    # Load accounts
    accounts = load_accounts()
    print(f"Found {Colors.BLUE}{len(accounts)}{Colors.ENDC} projects to configure\n")
    
    # Show options
    print(f"{Colors.BOLD}Options:{Colors.ENDC}")
    print("1. Automatic enable with gcloud CLI")
    print("2. Show manual enable URLs")
    print("3. Both (try automatic, then show URLs)")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "4":
        print("Exiting...")
        return 0
    
    if choice in ["1", "3"]:
        # Check for gcloud
        if not check_gcloud():
            print(f"\n{Colors.RED}gcloud CLI not found!{Colors.ENDC}")
            print("Install from: https://cloud.google.com/sdk/docs/install")
            
            if choice == "1":
                return 1
            else:
                print("\nShowing manual URLs instead...")
        else:
            # Process all projects
            print(f"\n{Colors.YELLOW}Enabling Vertex AI API for all projects...{Colors.ENDC}")
            
            success_count = 0
            failed_projects = []
            
            for account in accounts:
                if enable_api_for_project(account['project_id'], account.get('credentials_path')):
                    success_count += 1
                else:
                    failed_projects.append(account)
                time.sleep(2)  # Small delay between projects
            
            # Print summary
            print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
            print(f"{Colors.BOLD}SUMMARY{Colors.ENDC}")
            print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}")
            print(f"Successfully enabled API for {Colors.GREEN}{success_count}{Colors.ENDC} out of {len(accounts)} projects")
            
            if success_count == len(accounts):
                print(f"\n{Colors.GREEN}✓ All APIs enabled successfully!{Colors.ENDC}")
                print("\nNext step: Run the test suite to verify everything works:")
                print(f"  {Colors.BLUE}python test_vertex_suite.py{Colors.ENDC}")
                return 0
            elif failed_projects:
                print(f"\n{Colors.YELLOW}⚠ {len(failed_projects)} projects need manual enabling:{Colors.ENDC}")
                generate_manual_urls(failed_projects)
    
    if choice in ["2", "3"]:
        generate_manual_urls(accounts)
    
    print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
    print("1. Enable the APIs using the URLs above")
    print("2. Wait 2-3 minutes for changes to propagate")
    print("3. Run the test suite:")
    print(f"   {Colors.BLUE}python test_vertex_suite.py{Colors.ENDC}")
    print("4. Start indexing:")
    print(f"   {Colors.BLUE}python vertex_indexer.py --root C:/Users/ASUS/outlook_export --mode parallel{Colors.ENDC}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
