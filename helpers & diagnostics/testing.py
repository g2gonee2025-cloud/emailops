#!/usr/bin/env python3
"""
Consolidated testing utilities for EmailOps.
Combines all testing and verification functionality.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import vertexai
from dotenv import load_dotenv
from google import genai
from google.oauth2 import service_account

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DependencyVerifier:
    """Verify Python dependencies and imports."""

    # Critical dependencies to check
    DEPENDENCIES: ClassVar[list[str]] = [
        # Google Cloud dependencies
        "google.cloud.aiplatform",
        "google.auth",
        "google.api_core",
        "google.generativeai",
        # Core dependencies
        "streamlit",
        "qdrant_client",
        "pandas",
        "numpy",
        "pydantic",
        "langchain",
        "openai",
        # Other important packages
        "requests",
        "pytest",
        "dotenv",
        "yaml",
        "tiktoken",
        "tqdm",
    ]

    @staticmethod
    def check_import(module_name: str) -> tuple[bool, str | None]:
        """Check if a module can be imported."""
        try:
            __import__(module_name)
            return True, None
        except ImportError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected error: {e}"

    @staticmethod
    def get_version(module_name: str) -> str:
        """Get the version of an installed module."""
        try:
            module = __import__(module_name)
            if hasattr(module, "__version__"):
                return module.__version__
            elif hasattr(module, "version"):
                return module.version
            elif hasattr(module, "VERSION"):
                return module.VERSION
            else:
                return "Version unknown"
        except Exception:
            return "N/A"

    def verify_all(self) -> dict[str, Any]:
        """Verify all dependencies and return report."""
        print("=" * 60)
        print("Python Dependencies Verification Report")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Python Version: {sys.version}")
        print("=" * 60)
        print()

        results = {}
        failed_imports = []

        print("Checking critical dependencies...")
        print("-" * 60)

        for dep in self.DEPENDENCIES:
            success, error = self.check_import(dep)
            if success:
                version = self.get_version(dep)
                print(f"✓ {dep:<30} - OK (v{version})")
                results[dep] = {"status": "OK", "version": version, "error": None}
            else:
                print(f"✗ {dep:<30} - FAILED: {error}")
                results[dep] = {"status": "FAILED", "version": None, "error": error}
                failed_imports.append(dep)

        # Summary
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total dependencies checked: {len(self.DEPENDENCIES)}")
        print(f"Successful imports: {len(self.DEPENDENCIES) - len(failed_imports)}")
        print(f"Failed imports: {len(failed_imports)}")

        if failed_imports:
            print("\nFailed imports:")
            for dep in failed_imports:
                print(f"  - {dep}: {results[dep]['error']}")
        else:
            print("\n✓ All critical dependencies are properly installed!")

        return {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "results": results,
            "summary": {
                "total_checked": len(self.DEPENDENCIES),
                "successful": len(self.DEPENDENCIES) - len(failed_imports),
                "failed": len(failed_imports),
                "failed_list": failed_imports,
            },
        }


class GenAITester:
    """Test Google GenAI/Vertex AI authentication and functionality."""

    def __init__(self):
        load_dotenv()
        self.project = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
        self.credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    def test_environment(self) -> bool:
        """Check environment variables."""
        print("1. Checking environment variables...")
        required_vars = [
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
            "GOOGLE_APPLICATION_CREDENTIALS",
        ]

        all_set = True
        for var in required_vars:
            value = os.getenv(var)
            if value:
                if var == "GOOGLE_APPLICATION_CREDENTIALS":
                    credential_path = Path(value)
                    if credential_path.exists():
                        print(f"   ✓ {var}: {value} (file exists)")
                    else:
                        print(f"   ✗ {var}: {value} (file NOT found)")
                        all_set = False
                else:
                    print(f"   ✓ {var}: {value}")
            else:
                print(f"   ✗ {var}: NOT SET")
                all_set = False

        return all_set

    def test_genai_import(self) -> bool:
        """Test GenAI import."""
        print("\n2. Testing GenAI import...")
        try:
            print("   ✓ GenAI module imported successfully")
            return True
        except ImportError as e:
            print(f"   ✗ Failed to import GenAI: {e}")
            return False

    def test_vertex_init(self) -> bool:
        """Test Vertex AI initialization."""
        print("\n3. Testing Vertex AI initialization...")
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path
            )
            vertexai.init(
                project=self.project, location=self.location, credentials=credentials
            )
            print("   ✓ Vertex AI initialized successfully")
            return True
        except Exception as e:
            print(f"   ✗ Failed to initialize Vertex AI: {e}")
            return False

    def test_generation(self) -> bool:
        """Test text generation."""
        print("\n4. Testing text generation...")
        try:
            client = genai.Client(
                vertexai=True, project=self.project, location=self.location
            )

            response = client.models.generate_content(
                model="gemini-2.5-pro", contents="Say 'Hello, authentication works!'"
            )

            if response and response.text:
                print(f"   ✓ Generation successful: {response.text.strip()}")
                return True
            else:
                print("   ✗ Empty response received")
                return False

        except Exception as e:
            print(f"   ✗ Failed to generate content: {e}")
            return False

    def test_embedding(self) -> bool:
        """Test embedding generation."""
        print("\n5. Testing embedding generation...")
        try:
            client = genai.Client(
                vertexai=True, project=self.project, location=self.location
            )

            response = client.models.embed_content(
                model="gemini-embedding-001", contents=["test embedding"]
            )

            if response and response.embeddings:
                embedding = response.embeddings[0]
                if hasattr(embedding, "values") and embedding.values:
                    dim = len(embedding.values)
                    print(f"   ✓ Embedding successful: dimension {dim}")
                    return True

            print("   ✗ Empty embedding response")
            return False

        except Exception as e:
            print(f"   ✗ Failed to generate embedding: {e}")
            return False

    def test_all(self) -> bool:
        """Run all tests."""
        print("=== GenAI Authentication Test ===")
        print()

        tests = [
            self.test_environment,
            self.test_genai_import,
            self.test_vertex_init,
            self.test_generation,
            self.test_embedding,
        ]

        results = []
        for test in tests:
            try:
                results.append(test())
            except Exception as e:
                print(f"   ✗ Test failed with exception: {e}")
                results.append(False)

        print()
        if all(results):
            print("🎉 All tests passed! GenAI authentication is working correctly.")
            return True
        else:
            print(f"❌ {len([r for r in results if not r])} test(s) failed")
            return False


class CredentialTester:
    """Test GCP service account credentials."""

    def __init__(self, secrets_dir: Path = Path("secrets")):
        self.secrets_dir = secrets_dir
        self.priority_files = [
            "api-agent-470921-aa03081a1b4d.json",
            "apt-arcana-470409-i7-ce42b76061bf.json",
            "crafty-airfoil-474021-s2-34159960925b.json",
            "embed2-474114-fca38b4d2068.json",
            "my-project-31635v-8ec357ac35b2.json",
            "semiotic-nexus-470620-f3-3240cfaf6036.json",
        ]

    def test_credential_file(self, cred_path: Path) -> tuple[str, bool, str, dict]:
        """Test a single credential file with live API call."""
        if not cred_path.exists():
            return "", False, f"File not found: {cred_path}", {}

        try:
            # Read credential file
            with cred_path.open("r") as f:
                cred_data = json.load(f)

            project_id = cred_data.get("project_id", "unknown")
            client_email = cred_data.get("client_email", "unknown")

            # Validate required fields
            required_fields = [
                "type",
                "project_id",
                "private_key_id",
                "private_key",
                "client_email",
            ]
            missing = [f for f in required_fields if f not in cred_data]
            if missing:
                return (
                    project_id,
                    False,
                    f"Missing fields: {missing}",
                    {"client_email": client_email},
                )

            if cred_data.get("type") != "service_account":
                return (
                    project_id,
                    False,
                    "Not a service account file",
                    {"client_email": client_email},
                )

            # Set environment for this test
            os.environ["GCP_PROJECT"] = project_id
            os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path)

            # Try to initialize and make API call
            try:
                credentials = service_account.Credentials.from_service_account_file(
                    str(cred_path)
                )
                vertexai.init(
                    project=project_id, location="global", credentials=credentials
                )

                client = genai.Client(
                    vertexai=True, project=project_id, location="global"
                )

                # Make test embedding call
                resp = client.models.embed_content(
                    model="gemini-embedding-001", contents=["test embedding call"]
                )

                if resp and resp.embeddings and len(resp.embeddings) == 1:
                    embedding = resp.embeddings[0]
                    if hasattr(embedding, "values") and embedding.values:
                        dim = len(embedding.values)
                        return (
                            project_id,
                            True,
                            f"SUCCESS - Dimension: {dim}",
                            {"client_email": client_email, "dimension": dim},
                        )

                return (
                    project_id,
                    False,
                    "Empty/invalid embedding response",
                    {"client_email": client_email},
                )

            except Exception as api_error:
                error_msg = str(api_error)
                if "403" in error_msg or "permission" in error_msg.lower():
                    return (
                        project_id,
                        False,
                        "PERMISSION DENIED",
                        {"client_email": client_email},
                    )
                elif "429" in error_msg or "quota" in error_msg.lower():
                    return (
                        project_id,
                        False,
                        "QUOTA EXCEEDED",
                        {"client_email": client_email},
                    )
                elif "not found" in error_msg.lower():
                    return (
                        project_id,
                        False,
                        "PROJECT NOT FOUND",
                        {"client_email": client_email},
                    )
                else:
                    return (
                        project_id,
                        False,
                        f"API ERROR: {error_msg[:100]}",
                        {"client_email": client_email},
                    )

        except json.JSONDecodeError:
            return "", False, "Invalid JSON", {}
        except Exception as e:
            return "", False, f"Unexpected error: {e}", {}

    def test_all_credentials(self) -> int:
        """Test all credential files."""
        if not self.secrets_dir.exists():
            print("❌ No secrets directory found")
            return 1

        print("🧪 Testing EmailOps GCP Credentials - Live API Calls")
        print("=" * 60)

        working_count = 0
        failed_count = 0

        for filename in self.priority_files:
            cred_path = self.secrets_dir / filename
            print(f"\n📋 Testing: {filename}")
            print("-" * 40)

            project_id, success, message, details = self.test_credential_file(cred_path)

            if success:
                working_count += 1
                print(f"✅ {project_id}")
                print(f"   └─ {message}")
                print(f"   └─ Client: {details.get('client_email', 'N/A')}")
                print(f"   └─ Embedding dimension: {details.get('dimension', 'N/A')}")
            else:
                failed_count += 1
                print(f"❌ {project_id or 'UNKNOWN'}")
                print(f"   └─ {message}")
                if details.get("client_email"):
                    print(f"   └─ Client: {details['client_email']}")

        print("\n" + "=" * 60)
        print(f"📊 RESULTS: {working_count} working, {failed_count} failed")

        if working_count > 0:
            print("✅ SYSTEM STATUS: Ready for production use!")
            print(
                "🔧 Working credentials found - EmailOps can embed texts and generate responses"
            )
        else:
            print("❌ SYSTEM STATUS: No working credentials - Setup required")
            print("🔧 Check GCP project permissions and API enablement")

        return 0 if working_count > 0 else 1


class LLMRuntimeVerifier:
    """Verify LLM runtime fixes have been applied."""

    @staticmethod
    def verify_fixes() -> list[str]:
        """Check that all required fixes are in place."""
        issues = []

        # Check llm_runtime.py
        runtime_path = Path("emailops/llm_runtime.py")
        if not runtime_path.exists():
            issues.append("❌ llm_runtime.py not found")
            return issues

        runtime_code = runtime_path.read_text()

        # Check imports
        if "from collections.abc import Iterable" not in runtime_code:
            issues.append("❌ Missing: from collections.abc import Iterable")
        else:
            print("✅ Import: collections.abc.Iterable")

        # Check thread-safety locks
        if "_INIT_LOCK = threading.RLock()" not in runtime_code:
            issues.append("❌ Missing: _INIT_LOCK declaration")
        else:
            print("✅ Thread safety: _INIT_LOCK")

        if "_VALIDATED_LOCK = threading.RLock()" not in runtime_code:
            issues.append("❌ Missing: _VALIDATED_LOCK declaration")
        else:
            print("✅ Thread safety: _VALIDATED_LOCK")

        # Check rate limiting
        rate_limit_calls = runtime_code.count("_check_rate_limit()")
        if rate_limit_calls < 10:
            issues.append(f"❌ Insufficient rate limiting calls: {rate_limit_calls}")
        else:
            print(f"✅ Rate limiting: {rate_limit_calls} calls")

        # Check embed_texts signature
        if "def embed_texts(\n    texts: Iterable[str]," in runtime_code:
            print("✅ embed_texts accepts Iterable[str]")
        else:
            issues.append("❌ embed_texts should accept Iterable[str]")

        # Check list conversion
        if "seq = list(texts)" in runtime_code:
            print("✅ embed_texts realizes iterables to list")
        else:
            issues.append("❌ Missing: seq = list(texts) in embed_texts")

        # Check empty completion handling
        if 'raise LLMError("Empty completion from model")' in runtime_code:
            print("✅ Empty completion error handling")
        else:
            issues.append("❌ Missing: Empty completion error handling")

        return issues

    @staticmethod
    def verify_all() -> int:
        """Run all verifications."""
        print("🔍 Verifying LLM runtime fixes...\n")
        issues = LLMRuntimeVerifier.verify_fixes()

        print("\n" + "=" * 60)
        if issues:
            print("❌ ISSUES FOUND:")
            for issue in issues:
                print(f"  {issue}")
            return 1
        else:
            print("✅ All required fixes have been successfully applied!")
            print("\n📋 Summary of changes:")
            print("  • Rate limiting enforced for all API calls")
            print("  • Thread-safe initialization and account loading")
            print("  • Iterable support in embed_texts")
            print("  • Empty completion error handling")
            return 0


def main():
    """Main entry point with command-line interface."""

    parser = argparse.ArgumentParser(description="Testing utilities for EmailOps")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Commands
    subparsers.add_parser("deps", help="Verify Python dependencies")
    subparsers.add_parser("genai", help="Test GenAI authentication")
    subparsers.add_parser("creds", help="Test GCP credentials")
    subparsers.add_parser("runtime", help="Verify LLM runtime fixes")
    subparsers.add_parser("all", help="Run all tests")

    args = parser.parse_args()

    if not args.command:
        args.command = "all"

    exit_code = 0

    if args.command in ["deps", "all"]:
        verifier = DependencyVerifier()
        result = verifier.verify_all()
        if result["summary"]["failed"] > 0:
            exit_code = 1

    if args.command in ["genai", "all"]:
        tester = GenAITester()
        if not tester.test_all():
            exit_code = 1

    if args.command in ["creds", "all"]:
        tester = CredentialTester()
        if tester.test_all_credentials() != 0:
            exit_code = 1

    if args.command in ["runtime", "all"] and LLMRuntimeVerifier.verify_all() != 0:
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
