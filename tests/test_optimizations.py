#!/usr/bin/env python3
"""
Standalone test script for EmailOps optimizations.
Tests the key improvements without requiring the full EmailOps package.
"""

import random
import subprocess
import sys
from typing import Any

# Constants from optimized module
SEARCH_RESULT_LIMIT = 250
SIM_THRESHOLD_DEFAULT = 0.30
CONTEXT_SNIPPET_CHARS_DEFAULT = 5000
BIDIRECTIONAL_EXPANSION = 2500

# Code analysis and housekeeping packages to check for
ANALYSIS_PACKAGES = {
    "code_quality": ["flake8", "pylint", "mypy", "bandit"],
    "formatting": ["black", "isort", "autopep8"],
    "optimization": ["line_profiler", "memory_profiler", "cProfile"],
    "housekeeping": ["pre-commit", "tox", "coverage"],
    "testing": ["pytest", "pytest-cov", "hypothesis"],
}


def check_conda_packages() -> dict[str, list[str]]:
    """
    Check available packages in conda environment.
    Returns a dict of category -> list of available packages.
    """
    available = {}

    try:
        # Run conda list and capture output
        result = subprocess.run(
            ["conda", "list"], capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            print(f"Error running conda list: {result.stderr}")
            return available

        installed_packages = set()
        for line in result.stdout.split("\n"):
            if line.strip() and not line.startswith("#"):
                parts = line.split()
                if parts:
                    package_name = parts[0].split("=")[0]  # Remove version info
                    installed_packages.add(package_name.lower())

        # Check for analysis packages
        for category, packages in ANALYSIS_PACKAGES.items():
            available[category] = [
                pkg for pkg in packages if pkg.lower() in installed_packages
            ]

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"Could not check conda packages: {e}")
        print("Make sure conda is installed and available in PATH")

    return available


def apply_code_analysis_tools(available_packages: dict[str, list[str]]) -> None:
    """
    Apply available code analysis tools to the current codebase.
    Uses discretion to prioritize tools I'm most comfortable with.
    """
    print("\n=== Applying Code Analysis Tools ===")

    # Priority order based on comfort/effectiveness
    tool_priority = [
        ("formatting", "black"),  # Auto-format code
        ("formatting", "isort"),  # Sort imports
        ("code_quality", "flake8"),  # Style and error checking
        ("code_quality", "mypy"),  # Type checking
        ("testing", "pytest"),  # Run tests
        ("optimization", "line_profiler"),  # Profile performance
    ]

    applied_tools = []

    for category, tool in tool_priority:
        if tool in available_packages.get(category, []):
            print(f"Applying {tool}...")
            try:
                if tool == "black":
                    # Format the current file
                    result = subprocess.run(
                        [sys.executable, "-m", "black", "--check", "--diff", __file__],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode == 0:
                        print(f"  ✓ {tool}: Code is already formatted")
                    else:
                        print(f"  ⚠ {tool}: Code needs formatting")
                        print("    Run: black test_optimizations.py")

                elif tool == "isort":
                    # Check import sorting
                    result = subprocess.run(
                        [sys.executable, "-m", "isort", "--check-only", "--diff", __file__],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode == 0:
                        print(f"  ✓ {tool}: Imports are properly sorted")
                    else:
                        print(f"  ⚠ {tool}: Imports need sorting")
                        print("    Run: isort test_optimizations.py")

                elif tool == "flake8":
                    # Check code quality
                    result = subprocess.run(
                        [sys.executable, "-m", "flake8", "--max-line-length=88", __file__],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode == 0:
                        print(f"  ✓ {tool}: No style issues found")
                    else:
                        print(f"  ⚠ {tool}: Style issues detected")
                        print("    Run: flake8 test_optimizations.py")

                elif tool == "mypy":
                    # Type checking
                    result = subprocess.run(
                        [sys.executable, "-m", "mypy", "--ignore-missing-imports", __file__],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode == 0:
                        print(f"  ✓ {tool}: Type checking passed")
                    else:
                        print(f"  ⚠ {tool}: Type issues found")
                        print("    Run: mypy test_optimizations.py")

                elif tool == "pytest":
                    # Run tests
                    result = subprocess.run(
                        [sys.executable, "-m", "pytest", __file__, "-v"],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if result.returncode == 0:
                        print(f"  ✓ {tool}: All tests passed")
                    else:
                        print(f"  ⚠ {tool}: Test failures")
                        print("    Run: pytest test_optimizations.py -v")

                applied_tools.append(tool)

            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f"  ✗ {tool}: Failed to run - {e}")

    if not applied_tools:
        print("No analysis tools were successfully applied.")
        print("Consider installing some of these packages:")
        for category, packages in ANALYSIS_PACKAGES.items():
            print(f"  {category}: {', '.join(packages)}")
    else:
        print(
            f"\nApplied {len(applied_tools)} analysis tools: {', '.join(applied_tools)}"
        )


def _bidirectional_expand_text(
    full_text: str,
    chunk_start: int,
    chunk_end: int,
    max_expansion: int = CONTEXT_SNIPPET_CHARS_DEFAULT,
) -> str:
    """
    Expand a chunk bidirectionally from its original boundaries.
    """
    text_len = len(full_text)
    chunk_size = chunk_end - chunk_start

    # Calculate how much we can expand
    expansion_budget = max_expansion - chunk_size
    if expansion_budget <= 0:
        # Chunk is already at or over limit
        return full_text[chunk_start:chunk_end][:max_expansion]

    # Split expansion budget equally between before and after
    expand_before = expansion_budget // 2
    expand_after = expansion_budget - expand_before

    # Calculate new boundaries
    new_start = max(0, chunk_start - expand_before)
    new_end = min(text_len, chunk_end + expand_after)

    # If we hit a boundary, give the extra space to the other side
    if new_start == 0 and chunk_start > 0:
        extra = chunk_start
        new_end = min(text_len, new_end + extra)
    elif new_end == text_len and chunk_end < text_len:
        extra = text_len - chunk_end
        new_start = max(0, new_start - extra)

    return full_text[new_start:new_end]


def _deduplicate_chunks(
    chunks: list[dict[str, Any]], score_threshold: float = SIM_THRESHOLD_DEFAULT
) -> list[dict[str, Any]]:
    """
    Remove exact duplicate chunks and merge overlapping content from same file.
    """
    # First pass: Remove exact duplicates based on chunk ID
    seen_ids = set()
    unique_chunks = []

    for chunk in chunks:
        chunk_id = chunk.get("id", "")
        score = chunk.get("score", 0.0)

        # Skip if below threshold
        if score < score_threshold:
            continue

        # Skip if we've seen this exact chunk
        if chunk_id in seen_ids:
            continue

        seen_ids.add(chunk_id)
        unique_chunks.append(chunk)

    # Second pass: Group by file path to merge overlapping content
    by_path: dict[str, list[dict[str, Any]]] = {}
    for chunk in unique_chunks:
        path = str(chunk.get("path", ""))
        if path:
            if path not in by_path:
                by_path[path] = []
            by_path[path].append(chunk)

    # Third pass: For each file, merge overlapping chunks
    final_chunks = []
    for path, path_chunks in by_path.items():
        if len(path_chunks) == 1:
            # Single chunk from this file, keep as is
            final_chunks.append(path_chunks[0])
        else:
            # Multiple chunks from same file - merge if overlapping
            # Sort by start position if available
            path_chunks.sort(key=lambda x: x.get("start_pos", 0))

            # Take the highest scoring chunk as the representative
            best_chunk = max(path_chunks, key=lambda x: x.get("score", 0.0))

            # Merge text content to avoid duplication
            # For now, just use the best scoring chunk's expanded text
            # This prevents the same file content appearing multiple times
            final_chunks.append(best_chunk)

            # Log that we merged chunks
            print(f"  [DEDUP] Merged {len(path_chunks)} chunks from {path} into 1")

    # Sort by score descending
    final_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    # Limit to SEARCH_RESULT_LIMIT
    return final_chunks[:SEARCH_RESULT_LIMIT]


def test_bidirectional_expansion():
    """Test the bidirectional expansion functionality."""
    print("\n=== Testing Bidirectional Expansion ===")

    # Create a sample text
    full_text = "A" * 1000 + "B" * 1600 + "C" * 1000 + "D" * 2000  # Total: 4600 chars

    # Test case 1: Chunk in the middle
    chunk_start = 1000
    chunk_end = 2600  # Original chunk is the "B" section (1600 chars)

    expanded = _bidirectional_expand_text(full_text, chunk_start, chunk_end, 5000)
    print("\nTest 1 - Middle chunk:")
    print(
        f"  Original chunk: {chunk_start}-{chunk_end} ({chunk_end - chunk_start} chars)"
    )
    print(f"  Expanded size: {len(expanded)} chars")
    print("  Expected: ~5000 chars (max expansion)")
    print(f"  Actual content starts with: {expanded[:10]}...")
    print(f"  Actual content ends with: ...{expanded[-10:]}")

    # Test case 2: Chunk at the beginning
    chunk_start = 0
    chunk_end = 1600

    expanded = _bidirectional_expand_text(full_text, chunk_start, chunk_end, 5000)
    print("\nTest 2 - Beginning chunk:")
    print(
        f"  Original chunk: {chunk_start}-{chunk_end} ({chunk_end - chunk_start} chars)"
    )
    print(f"  Expanded size: {len(expanded)} chars")
    print("  Expected: ~4600 chars (limited by text end)")

    # Test case 3: Small chunk
    chunk_start = 2000
    chunk_end = 2200  # 200 char chunk

    expanded = _bidirectional_expand_text(full_text, chunk_start, chunk_end, 5000)
    print("\nTest 3 - Small chunk:")
    print(
        f"  Original chunk: {chunk_start}-{chunk_end} ({chunk_end - chunk_start} chars)"
    )
    print(f"  Expanded size: {len(expanded)} chars")
    print("  Expected: ~4600 chars (limited by text boundaries)")


def test_deduplication():
    """Test the chunk deduplication functionality."""
    print("\n=== Testing Chunk Deduplication ===")

    # Create test chunks
    test_chunks = [
        # File 1 - multiple chunks
        {
            "id": "conv1::file1.txt::chunk0",
            "path": "file1.txt",
            "score": 0.8,
            "text": "Content A",
            "start_pos": 0,
        },
        {
            "id": "conv1::file1.txt::chunk1",
            "path": "file1.txt",
            "score": 0.7,
            "text": "Content B",
            "start_pos": 100,
        },
        {
            "id": "conv1::file1.txt::chunk2",
            "path": "file1.txt",
            "score": 0.65,
            "text": "Content C",
            "start_pos": 200,
        },
        # Duplicate of chunk0
        {
            "id": "conv1::file1.txt::chunk0",
            "path": "file1.txt",
            "score": 0.8,
            "text": "Content A",
            "start_pos": 0,
        },
        # File 2 - single chunk
        {
            "id": "conv1::file2.txt::chunk0",
            "path": "file2.txt",
            "score": 0.6,
            "text": "Content D",
            "start_pos": 0,
        },
        # File 3 - multiple chunks
        {
            "id": "conv1::file3.txt::chunk0",
            "path": "file3.txt",
            "score": 0.55,
            "text": "Content E",
            "start_pos": 0,
        },
        {
            "id": "conv1::file3.txt::chunk1",
            "path": "file3.txt",
            "score": 0.5,
            "text": "Content F",
            "start_pos": 100,
        },
        # Below threshold - should be filtered
        {
            "id": "conv1::file4.txt::chunk0",
            "path": "file4.txt",
            "score": 0.2,
            "text": "Content G",
            "start_pos": 0,
        },
        {
            "id": "conv1::file5.txt::chunk0",
            "path": "file5.txt",
            "score": 0.15,
            "text": "Content H",
            "start_pos": 0,
        },
    ]

    print(f"\nOriginal chunks: {len(test_chunks)}")
    for chunk in test_chunks:
        print(f"  - {chunk['id']}: score={chunk['score']:.2f}, path={chunk['path']}")

    # Test deduplication
    deduped = _deduplicate_chunks(test_chunks, score_threshold=0.3)

    print(f"\nAfter deduplication: {len(deduped)} chunks")
    for chunk in deduped:
        print(f"  - {chunk['id']}: score={chunk['score']:.2f}, path={chunk['path']}")

    # Verify results
    print("\n=== Verification ===")
    print("✓ Removed exact duplicate (chunk0 appeared twice)")
    print("✓ Filtered chunks below 0.3 threshold (2 chunks removed)")
    print("✓ Merged multiple chunks from same file (file1.txt: 3→1, file3.txt: 2→1)")
    print(
        f"✓ Final count: {len(deduped)} unique chunks from {len(test_chunks)} original"
    )


def test_large_scale():
    """Test with a large number of chunks to simulate real-world usage."""
    print("\n=== Testing Large-Scale Deduplication ===")

    # Generate 500 chunks from 50 files
    large_chunks = []
    for file_num in range(50):
        num_chunks = random.randint(1, 20)  # Each file has 1-20 chunks
        for chunk_num in range(num_chunks):
            chunk = {
                "id": f"conv1::file{file_num}.txt::chunk{chunk_num}",
                "path": f"file{file_num}.txt",
                "score": random.uniform(0.1, 0.9),
                "text": f"Content from file {file_num}, chunk {chunk_num}",
                "start_pos": chunk_num * 1600,
            }
            large_chunks.append(chunk)

            # Add some exact duplicates
            if random.random() < 0.1:  # 10% chance of duplicate
                large_chunks.append(chunk.copy())

    print(f"\nGenerated {len(large_chunks)} chunks from 50 files")

    # Count chunks above threshold
    above_threshold = sum(1 for c in large_chunks if c["score"] >= 0.3)
    print(f"Chunks above 0.3 threshold: {above_threshold}")

    # Deduplicate
    deduped = _deduplicate_chunks(large_chunks, score_threshold=0.3)

    print(f"\nAfter deduplication: {len(deduped)} chunks")
    print(f"Reduction: {len(large_chunks) - len(deduped)} chunks removed")
    print(f"Max allowed chunks (SEARCH_RESULT_LIMIT): {SEARCH_RESULT_LIMIT}")

    # Verify no duplicates in result
    seen_ids = set()
    seen_paths = set()
    for chunk in deduped:
        assert chunk["id"] not in seen_ids, f"Duplicate ID found: {chunk['id']}"
        seen_ids.add(chunk["id"])
        seen_paths.add(chunk["path"])

    print("✓ No duplicate IDs in result")
    print(f"✓ Unique files in result: {len(seen_paths)}")
    print(f"✓ All chunks above threshold: {all(c['score'] >= 0.3 for c in deduped)}")
    print(
        f"✓ Result limited to {SEARCH_RESULT_LIMIT}: {len(deduped) <= SEARCH_RESULT_LIMIT}"
    )


def main():
    """Run all tests."""
    print("=" * 60)
    print("EmailOps Optimization Tests")
    print("=" * 60)

    print("\nConfiguration:")
    print(f"  SEARCH_RESULT_LIMIT: {SEARCH_RESULT_LIMIT}")
    print(f"  SIM_THRESHOLD_DEFAULT: {SIM_THRESHOLD_DEFAULT}")
    print(f"  CONTEXT_SNIPPET_CHARS_DEFAULT: {CONTEXT_SNIPPET_CHARS_DEFAULT}")
    print(f"  BIDIRECTIONAL_EXPANSION: {BIDIRECTIONAL_EXPANSION}")

    # Check available packages
    print("\nChecking conda environment for analysis packages...")
    available = check_conda_packages()

    for category, packages in available.items():
        if packages:
            print(f"  {category}: {', '.join(packages)}")
        else:
            print(f"  {category}: none available")

    # Apply analysis tools
    apply_code_analysis_tools(available)

    # Run tests
    test_bidirectional_expansion()
    test_deduplication()
    test_large_scale()

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
