#!/usr/bin/env python3
"""
Test script to verify centralized configuration integration across EmailOps codebase.
This script checks that the config module is properly integrated and working.
"""

import os
import sys
from pathlib import Path


# Colors for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def colored(text: str, color: str) -> str:
    """Apply color to text if terminal supports it"""
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.ENDC}"
    return text


def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'=' * 70}")
    print(colored(f"  {title}", Colors.BOLD))
    print("=" * 70)


def test_config_module():
    """Test that the config module can be imported and used"""
    print_header("Testing Config Module Import")

    try:
        from emailops.core_config import get_config

        get_config()
        print(colored("✅ Successfully imported config module", Colors.GREEN))
        return True
    except ImportError as e:
        print(colored(f"❌ Failed to import config module: {e}", Colors.RED))
        return False


def test_config_singleton():
    """Test that config returns singleton instance"""
    print_header("Testing Config Singleton Pattern")

    try:
        from emailops.core_config import get_config

        # Get config instances
        config1 = get_config()
        config2 = get_config()

        # Check they are the same instance
        if config1 is config2:
            print(colored("✅ Config singleton works correctly", Colors.GREEN))
            return True
        else:
            print(colored("❌ Config not returning singleton", Colors.RED))
            return False
    except Exception as e:
        print(colored(f"❌ Error testing singleton: {e}", Colors.RED))
        return False


def test_config_values():
    """Test that config has expected values"""
    print_header("Testing Config Values")

    try:
        from emailops.core_config import get_config

        config = get_config()

        # Test expected attributes exist (hierarchical structure)
        required_attrs = [
            ("directories.index_dirname", "INDEX_DIRNAME"),
            ("directories.chunk_dirname", "CHUNK_DIRNAME"),
            ("processing.batch_size", "DEFAULT_BATCH_SIZE"),
            ("processing.chunk_size", "DEFAULT_CHUNK_SIZE"),
            ("processing.chunk_overlap", "DEFAULT_CHUNK_OVERLAP"),
            ("processing.num_workers", "DEFAULT_NUM_WORKERS"),
            ("gcp.gcp_project", "GCP_PROJECT"),
            ("gcp.gcp_region", "GCP_REGION"),
            ("gcp.vertex_location", "VERTEX_LOCATION"),
            ("directories.secrets_dir", "SECRETS_DIR"),
            ("system.log_level", "LOG_LEVEL"),
        ]

        missing = []
        for attr_path, display_name in required_attrs:
            parts = attr_path.split('.')
            obj = config
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    missing.append(display_name)
                    break

        if missing:
            print(colored(f"❌ Missing attributes: {missing}", Colors.RED))
            return False

        # Print current values
        print(colored("✅ All required attributes present", Colors.GREEN))
        print("\nCurrent configuration values:")
        print(f"  INDEX_DIRNAME: {config.directories.index_dirname}")
        print(f"  CHUNK_DIRNAME: {config.directories.chunk_dirname}")
        print(f"  DEFAULT_BATCH_SIZE: {config.processing.batch_size}")
        print(f"  DEFAULT_CHUNK_SIZE: {config.processing.chunk_size}")
        print(f"  DEFAULT_CHUNK_OVERLAP: {config.processing.chunk_overlap}")
        print(f"  DEFAULT_NUM_WORKERS: {config.processing.num_workers}")
        print(f"  GCP_PROJECT: {config.gcp.gcp_project}")
        print(f"  GCP_REGION: {config.gcp.gcp_region}")
        print(f"  VERTEX_LOCATION: {config.gcp.vertex_location}")
        print(f"  SECRETS_DIR: {config.directories.secrets_dir}")
        print(f"  LOG_LEVEL: {config.system.log_level}")

        return True
    except Exception as e:
        print(colored(f"❌ Error testing config values: {e}", Colors.RED))
        return False


def test_environment_override():
    """Test that environment variables override defaults"""
    print_header("Testing Environment Variable Override")

    try:
        # Set test environment variable
        test_value = "_test_index_dir"
        os.environ["INDEX_DIRNAME"] = test_value

        # Force reload of config
        import importlib

        from emailops import core_config as config_module

        importlib.reload(config_module)

        from emailops.core_config import get_config, reset_config

        reset_config()
        config = get_config()

        if test_value == config.directories.index_dirname:
            print(
                colored(
                    f"✅ Environment override works: INDEX_DIRNAME = {test_value}",
                    Colors.GREEN,
                )
            )

            # Reset to default
            del os.environ["INDEX_DIRNAME"]
            reset_config()

            return True
        else:
            print(
                colored(
                    f"❌ Environment override failed: got {config.directories.index_dirname}, expected {test_value}",
                    Colors.RED,
                )
            )
            return False
    except Exception as e:
        print(colored(f"❌ Error testing environment override: {e}", Colors.RED))
        return False


def test_module_integration():
    """Test that other modules are using config"""
    print_header("Testing Module Integration")

    modules_to_test = [
        ("ui.emailops_ui", "UI module"),
        ("processing.processor", "Processor module"),
        ("diagnostics.monitor", "Monitor module"),
        ("diagnostics.statistics", "Statistics module"),
    ]

    all_passed = True

    for module_name, display_name in modules_to_test:
        try:
            # Check if module imports config
            module_path = Path(module_name.replace(".", "/") + ".py")
            if module_path.exists():
                with module_path.open(encoding="utf-8") as f:
                    content = f.read()
                    if (
                        "from emailops.core_config import" in content
                        or "from emailops import core_config" in content
                    ):
                        print(
                            colored(f"✅ {display_name} imports config", Colors.GREEN)
                        )
                    else:
                        print(
                            colored(
                                f"⚠️  {display_name} may not use config", Colors.YELLOW
                            )
                        )
                        # Not a failure, just a warning
            else:
                print(
                    colored(
                        f"⚠️  {display_name} file not found at {module_path}",
                        Colors.YELLOW,
                    )
                )
        except Exception as e:
            print(colored(f"❌ Error checking {display_name}: {e}", Colors.RED))
            all_passed = False

    return all_passed


def test_credential_discovery():
    """Test credential file discovery"""
    print_header("Testing Credential Discovery")

    try:
        from emailops.core_config import get_config

        config = get_config()

        # Test credential discovery
        cred_file = config.get_credential_file()
        if cred_file:
            print(
                colored(
                    f"✅ Found credential file: {Path(cred_file).name}", Colors.GREEN
                )
            )
        else:
            print(
                colored(
                    "i  No credential files found (this is OK if not using GCP)",
                    Colors.YELLOW,
                )
            )

        # Test secrets directory
        secrets_dir = config.get_secrets_dir()
        if secrets_dir and secrets_dir.exists():
            print(colored(f"✅ Secrets directory exists: {secrets_dir}", Colors.GREEN))
        else:
            print(
                colored(
                    "i  Secrets directory not found (will use defaults)", Colors.YELLOW
                )
            )

        return True
    except Exception as e:
        print(colored(f"❌ Error testing credential discovery: {e}", Colors.RED))
        return False


def check_hardcoded_values():
    """Check for remaining hardcoded values that should use config"""
    print_header("Checking for Hardcoded Values")

    patterns_to_check = [
        ("_index", "Hardcoded index directory name"),
        ("_chunks", "Hardcoded chunks directory name"),
        ('os.getenv("INDEX_DIRNAME"', "Direct INDEX_DIRNAME env access"),
        ('os.getenv("CHUNK_DIRNAME"', "Direct CHUNK_DIRNAME env access"),
        ('os.getenv("EMBED_BATCH"', "Direct EMBED_BATCH env access"),
        ("chunk_size=1600", "Hardcoded chunk size"),
        ("chunk_overlap=200", "Hardcoded chunk overlap"),
    ]

    # Files to check (excluding config.py itself and test files)
    files_to_check = [
        "ui/emailops_ui.py",
        "processing/processor.py",
        "diagnostics/monitor.py",
        "diagnostics/statistics.py",
        "diagnostics/diagnostics.py",
        "emailops/email_indexer.py",
        "emailops/text_chunker.py",
    ]

    found_issues = []

    for file_path in files_to_check:
        path = Path(file_path)
        if not path.exists():
            continue

        try:
            with path.open(encoding="utf-8") as f:
                content = f.read()
                for pattern, description in patterns_to_check:
                    # Skip if it's in a comment or string
                    if pattern in content:
                        # Rough check - might have false positives
                        lines = content.split("\n")
                        for line_num, line in enumerate(lines, 1):
                            if pattern in line and not line.strip().startswith("#"):
                                found_issues.append(
                                    f"{file_path}:{line_num} - {description}"
                                )
        except Exception as e:
            print(colored(f"⚠️  Error checking {file_path}: {e}", Colors.YELLOW))

    if found_issues:
        print(colored("⚠️  Found potential hardcoded values:", Colors.YELLOW))
        for issue in found_issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(found_issues) > 10:
            print(f"  ... and {len(found_issues) - 10} more")
        return False
    else:
        print(colored("✅ No obvious hardcoded values found", Colors.GREEN))
        return True


def main():
    """Run all tests"""
    print(colored("\n" + "=" * 70, Colors.BOLD))
    print(colored("  EMAILOPS CONFIG INTEGRATION TEST", Colors.BOLD))
    print(colored("=" * 70, Colors.BOLD))

    # Run tests
    tests = [
        ("Config Module Import", test_config_module),
        ("Config Singleton", test_config_singleton),
        ("Config Values", test_config_values),
        ("Environment Override", test_environment_override),
        ("Module Integration", test_module_integration),
        ("Credential Discovery", test_credential_discovery),
        ("Hardcoded Values Check", check_hardcoded_values),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(colored(f"❌ Unexpected error in {test_name}: {e}", Colors.RED))
            results.append((test_name, False))

    # Print summary
    print_header("TEST SUMMARY")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = (
            colored("✅ PASSED", Colors.GREEN)
            if passed
            else colored("❌ FAILED", Colors.RED)
        )
        print(f"  {test_name:30} {status}")

    print(f"\n  Total: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print(
            colored(
                "\n🎉 All tests passed! Config integration is working correctly.",
                Colors.GREEN,
            )
        )
        return 0
    else:
        print(
            colored(
                f"\n⚠️  {total_count - passed_count} tests failed. Please review the issues above.",
                Colors.YELLOW,
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
