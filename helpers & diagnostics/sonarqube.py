#!/usr/bin/env python3
"""
Consolidated SonarQube utilities for EmailOps.
Combines functionality from all SonarQube-related scripts.
"""

import argparse
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import requests

# Configuration
SONAR_URL = os.getenv("SONAR_HOST_URL", "http://localhost:9099")
DEFAULT_USER = "admin"
DEFAULT_PASS = "!Sharjah2050"
TOKEN_NAME = "emailops-analysis"
PROPERTIES_FILE = "sonar-project.properties"

# Scanner configuration
SCANNER_VERSION = "5.0.1.3006"
SCANNER_ZIP_URL = f"https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-{SCANNER_VERSION}-windows.zip"
SCANNER_DIR_NAME = f"sonar-scanner-{SCANNER_VERSION}-windows"


class SonarQubeManager:
    """Manages SonarQube authentication and analysis."""

    def __init__(
        self,
        url: str = SONAR_URL,
        username: str = DEFAULT_USER,
        password: str = DEFAULT_PASS,
    ):
        self.url = url
        self.username = username
        self.password = password
        self.token_name = TOKEN_NAME

    def check_server(self) -> bool:
        """Check if SonarQube server is running."""
        try:
            response = requests.get(f"{self.url}/api/system/status", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def wait_for_server(self, max_wait: int = 180) -> bool:
        """Wait for SonarQube server to be ready."""
        print("Waiting for SonarQube server to be ready...")
        print("(This typically takes 60-120 seconds on first start)")

        start_time = time.time()
        dots = 0

        while time.time() - start_time < max_wait:
            if self.check_server():
                print("\n✓ SonarQube server is ready!")
                return True

            print(".", end="", flush=True)
            dots += 1
            if dots % 60 == 0:
                elapsed = int(time.time() - start_time)
                print(f" [{elapsed}s]")

            time.sleep(1)

        print(f"\n✗ Server did not become ready within {max_wait} seconds")
        return False

    def authenticate_with_password(self) -> bool:
        """Test authentication with username/password."""
        try:
            response = requests.get(
                f"{self.url}/api/authentication/validate",
                auth=(self.username, self.password),
                timeout=10,
            )
            return response.status_code == 200 and response.json().get("valid", False)
        except Exception:
            return False

    def list_tokens(self) -> list:
        """List all existing tokens."""
        try:
            url = f"{self.url}/api/user_tokens/search"
            response = requests.get(
                url, auth=(self.username, self.password), timeout=10
            )
            if response.status_code == 200:
                return response.json().get("userTokens", [])
            return []
        except Exception:
            return []

    def revoke_token(self, token_name: str) -> bool:
        """Revoke an existing token."""
        try:
            url = f"{self.url}/api/user_tokens/revoke"
            response = requests.post(
                url,
                auth=(self.username, self.password),
                data={"name": token_name},
                timeout=10,
            )
            return response.status_code == 204
        except Exception:
            return False

    def generate_token(self, token_name: str | None = None) -> tuple[bool, str]:
        """Generate a new token."""
        if token_name is None:
            token_name = self.token_name

        try:
            url = f"{self.url}/api/user_tokens/generate"
            response = requests.post(
                url,
                auth=(self.username, self.password),
                data={"name": token_name},
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                token = data.get("token")
                if token:
                    return True, token
            return False, f"Status {response.status_code}: {response.text}"
        except Exception as e:
            return False, str(e)

    def setup_token(self) -> str | None:
        """Complete token setup: revoke old, create new, update properties."""
        print("=" * 70)
        print("SonarQube Authentication Setup")
        print("=" * 70)
        print()

        # Check server
        print("Step 1: Checking SonarQube server...")
        if not self.check_server() and not self.wait_for_server():
            print("✗ SonarQube server is not accessible")
            return None
        print("✓ Server is running")

        # Test authentication
        print("\nStep 2: Testing password authentication...")
        if not self.authenticate_with_password():
            print("✗ Cannot authenticate with admin credentials")
            return None
        print("✓ Password authentication works")

        # Clean up old tokens
        print("\nStep 3: Cleaning up old tokens...")
        tokens = self.list_tokens()
        for token in tokens:
            if token["name"] == self.token_name:
                print(f"Revoking existing token: {token['name']}")
                self.revoke_token(token["name"])

        # Generate new token
        print(f"\nStep 4: Generating new token '{self.token_name}'...")
        success, token = self.generate_token()
        if not success:
            print(f"✗ Failed to generate token: {token}")
            return None
        print(f"✓ Token generated: {token}")

        # Update properties file
        print(f"\nStep 5: Updating {PROPERTIES_FILE}...")
        if self.update_properties_file(token):
            print("✓ Properties file updated")
        else:
            print("✗ Failed to update properties file")

        # Update .env file
        self.update_env_file(token)

        return token

    def update_properties_file(self, token: str) -> bool:
        """Update sonar-project.properties with the token."""
        try:
            props_path = Path(PROPERTIES_FILE)
            if not props_path.exists():
                print(f"Creating {PROPERTIES_FILE}...")
                content = f"""# SonarQube project configuration
sonar.projectKey=emailops_vertex_ai
sonar.projectName=EmailOps Vertex AI
sonar.projectVersion=1.0
sonar.sources=emailops
sonar.python.version=3
sonar.host.url={self.url}
sonar.token={token}
"""
                props_path.write_text(content, encoding="utf-8")
            else:
                content = props_path.read_text(encoding="utf-8")

                # Update or add token
                if re.search(r"^sonar\.token=", content, re.MULTILINE):
                    content = re.sub(
                        r"^sonar\.token=.*$",
                        f"sonar.token={token}",
                        content,
                        flags=re.MULTILINE,
                    )
                else:
                    content = re.sub(
                        r"(sonar\.host\.url=.*\n)", f"\\1sonar.token={token}\n", content
                    )

                props_path.write_text(content, encoding="utf-8")

            return True
        except Exception as e:
            print(f"Error updating properties file: {e}")
            return False

    def update_env_file(self, token: str) -> bool:
        """Update .env file with the token."""
        try:
            env_path = Path(".env")
            if env_path.exists():
                content = env_path.read_text(encoding="utf-8")

                if "SONAR_TOKEN=" in content:
                    content = re.sub(
                        r"SONAR_TOKEN=.*", f'SONAR_TOKEN="{token}"', content
                    )
                else:
                    content += f'\nSONAR_TOKEN="{token}"\n'

                env_path.write_text(content, encoding="utf-8")
                print("✓ .env file updated")
                return True
        except Exception as e:
            print(f"✗ Failed to update .env: {e}")
        return False


class SonarScanner:
    """Manages SonarQube Scanner download and execution."""

    def __init__(self, tools_dir: Path | None = None):
        self.tools_dir = tools_dir or Path(".sonar-tools")
        self.tools_dir.mkdir(exist_ok=True)
        self.scanner_path = self.tools_dir / SCANNER_DIR_NAME

    def download_scanner(self) -> Path:
        """Download and extract SonarQube Scanner."""
        scanner_zip = self.tools_dir / f"sonar-scanner-{SCANNER_VERSION}.zip"

        if self.scanner_path.exists():
            print(f"✓ Scanner already exists at: {self.scanner_path}")
            return self.scanner_path

        print(f"Downloading SonarQube Scanner {SCANNER_VERSION}...")
        try:
            urlretrieve(SCANNER_ZIP_URL, scanner_zip)
            print(f"✓ Downloaded to: {scanner_zip}")

            print("Extracting scanner...")
            with zipfile.ZipFile(scanner_zip, "r") as zip_ref:
                zip_ref.extractall(self.tools_dir)
            print(f"✓ Extracted to: {self.scanner_path}")

            # Clean up zip file
            scanner_zip.unlink()

            return self.scanner_path
        except Exception as e:
            print(f"✗ Failed to download/extract scanner: {e}")
            sys.exit(1)

    def run_analysis(
        self, project_dir: Path | None = None, token: str | None = None
    ) -> int:
        """Execute SonarQube analysis."""
        project_dir = project_dir or Path.cwd()
        scanner_bin = self.scanner_path / "bin" / "sonar-scanner.bat"

        if not scanner_bin.exists():
            print(f"✗ Scanner binary not found at: {scanner_bin}")
            return 1

        # Check if sonar-project.properties exists
        props_file = project_dir / "sonar-project.properties"
        if not props_file.exists():
            print(f"✗ sonar-project.properties not found at: {props_file}")
            return 1

        print("\n" + "=" * 60)
        print("Running SonarQube Analysis")
        print("=" * 60 + "\n")

        # Build command
        cmd = [
            str(scanner_bin),
            f"-Dproject.settings={props_file}",
            "-Dsonar.scanAllFiles=true",
            "-Dsonar.qualitygate.wait=false",
        ]

        # Add token if provided
        if token:
            cmd.append(f"-Dsonar.token={token}")
        elif os.getenv("SONAR_TOKEN"):
            cmd.append(f"-Dsonar.token={os.getenv('SONAR_TOKEN')}")

        print(f"Scanner: {scanner_bin}")
        print(f"Project: {project_dir}")
        print("\nExecuting analysis...")

        try:
            result = subprocess.run(
                cmd, cwd=project_dir, capture_output=False, text=True, check=False
            )
            return result.returncode
        except Exception as e:
            print(f"✗ Failed to execute scanner: {e}")
            return 1


def setup_and_scan():
    """Complete setup and scan workflow."""
    manager = SonarQubeManager()
    scanner = SonarScanner()

    # Setup token
    token = manager.setup_token()
    if not token:
        print("\n✗ Failed to setup authentication")
        return 1

    # Download scanner if needed
    scanner.download_scanner()

    # Run analysis
    print("\n" + "=" * 70)
    print("Starting SonarQube Analysis")
    print("=" * 70)

    exit_code = scanner.run_analysis(token=token)

    if exit_code == 0:
        print("\n" + "=" * 60)
        print("✓ SonarQube analysis completed successfully!")
        print("=" * 60)
        print(f"\nView results at: {SONAR_URL}/dashboard?id=emailops_vertex_ai")
    else:
        print("\n" + "=" * 60)
        print(f"✗ SonarQube analysis failed with exit code: {exit_code}")
        print("=" * 60)

    return exit_code


def main():
    """Main entry point with command-line interface."""

    parser = argparse.ArgumentParser(description="SonarQube utilities for EmailOps")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Setup command
    subparsers.add_parser("setup", help="Setup SonarQube authentication")

    # Scan command
    subparsers.add_parser("scan", help="Run SonarQube analysis")

    # Full command
    subparsers.add_parser("full", help="Setup and scan")

    # Token command
    token_parser = subparsers.add_parser("token", help="Generate new token")
    token_parser.add_argument("--name", help="Token name", default=TOKEN_NAME)

    args = parser.parse_args()

    if not args.command:
        args.command = "full"

    if args.command == "setup":
        manager = SonarQubeManager()
        token = manager.setup_token()
        return 0 if token else 1

    elif args.command == "scan":
        scanner = SonarScanner()
        scanner.download_scanner()
        return scanner.run_analysis()

    elif args.command == "full":
        return setup_and_scan()

    elif args.command == "token":
        manager = SonarQubeManager()
        if manager.check_server():
            success, token = manager.generate_token(args.name)
            if success:
                print(f"✓ Token generated: {token}")
                return 0
            else:
                print(f"✗ Failed: {token}")
                return 1
        else:
            print("✗ Server not available")
            return 1


if __name__ == "__main__":
    sys.exit(main())
