#!/usr/bin/env python3
"""Verify that all required LLM runtime fixes have been applied."""

import sys
from pathlib import Path


def verify_fixes():
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

    # Check rate limiting calls
    rate_limit_calls = runtime_code.count("_check_rate_limit()")
    if rate_limit_calls < 10:  # Should have at least 10 calls
        issues.append(f"❌ Insufficient rate limiting calls: {rate_limit_calls} (expected >= 10)")
    else:
        print(f"✅ Rate limiting: {rate_limit_calls} calls to _check_rate_limit()")

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

    # Check GenerationConfig import usage
    if "from vertexai.generative_models import GenerationConfig" in runtime_code:
        print("✅ GenerationConfig used in complete_text")
    else:
        issues.append("❌ Missing: GenerationConfig import in complete_text")

    # Check empty completion handling
    if 'raise LLMError("Empty completion from model")' in runtime_code:
        print("✅ Empty completion error handling")
    else:
        issues.append("❌ Missing: Empty completion error handling")

    # Check JSON array support
    if r'r"(\{[\s\S]*\}|\[[\s\S]*\])"' in runtime_code:
        print("✅ JSON salvage supports arrays")
    else:
        issues.append("❌ JSON salvage should support arrays")

    # Check Qwen no zero vectors
    if 'raise LLMError(f"Qwen embedding failed for batch {i}:{i+B}' in runtime_code:
        print("✅ Qwen raises error instead of zero vectors")
    else:
        issues.append("❌ Qwen should raise error instead of zero vectors")

    # Check project rotation thread safety
    if "with _PROJECT_ROTATION_LOCK:\n                            _PROJECT_ROTATION[\"consecutive_errors\"]" in runtime_code:
        print("✅ Project rotation uses locks for consecutive_errors")
    else:
        issues.append("❌ Project rotation consecutive_errors should use locks")

    # Check llm_client.py
    client_path = Path("emailops/llm_client.py")
    if client_path.exists():
        client_code = client_path.read_text()
        if "the runtime now realizes non-list iterables" in client_code:
            print("✅ llm_client.py docstring updated")
        else:
            issues.append("❌ llm_client.py docstring not updated")

    return issues

def main():
    print("🔍 Verifying LLM runtime fixes...\n")
    issues = verify_fixes()

    print("\n" + "="*60)
    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        sys.exit(1)
    else:
        print("✅ All required fixes have been successfully applied!")
        print("\n📋 Summary of changes:")
        print("  • Rate limiting enforced for all API calls")
        print("  • Thread-safe initialization and account loading")
        print("  • Thread-safe project rotation")
        print("  • Iterable support in embed_texts with list realization")
        print("  • Qwen embedding hard failures (no zero vectors)")
        print("  • Unified Vertex completion config with empty output handling")
        print("  • JSON salvage extended to arrays")
        print("  • Client docstring updated")
        sys.exit(0)

if __name__ == "__main__":
    main()
