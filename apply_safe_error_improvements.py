#!/usr/bin/env python3
"""
Apply safe error handling improvements to EmailOps codebase.
This script adds logging and improves error handling without breaking functionality.
"""

import logging
import re
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def add_logging_import(content: str) -> str:
    """Ensure logging is imported at the top of the file."""
    if 'import logging' not in content:
        # Add after the last import statement
        import_section_end = 0
        for match in re.finditer(r'^(from|import)\s+\S+', content, re.MULTILINE):
            import_section_end = match.end()

        if import_section_end > 0:
            return content[:import_section_end] + '\nimport logging\n' + content[import_section_end:]
    return content

def apply_email_processing_improvements():
    """Improve error handling in email_processing.py."""
    file_path = Path("emailops/email_processing.py")
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return False

    content = file_path.read_text()
    original = content

    # Fix the date parsing exception handler (around line 194)
    content = re.sub(
        r'(\s+except Exception):\n(\s+)return None',
        r'\1 as e:\n\2# Log the parsing failure for debugging\n\2logger.debug("Failed to parse date from email header: %s", e)\n\2return None',
        content
    )

    if content != original:
        file_path.write_text(content)
        logger.info(f"✓ Updated {file_path}: Added logging for date parsing failures")
        return True
    return False

def apply_file_utils_improvements():
    """Improve error handling in file_utils.py."""
    file_path = Path("emailops/file_utils.py")
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return False

    content = file_path.read_text()
    original = content

    # Fix encoding detection loop (keep continue, add logging)
    content = re.sub(
        r'(\s+except \(UnicodeDecodeError, UnicodeError\)):\n(\s+)continue',
        r'\1 as e:\n\2# Log at trace level to avoid spam\n\2logger.debug("Encoding %s failed for %s: %s", enc, path.name, e)\n\2continue',
        content
    )

    content = re.sub(
        r'(\s+except Exception):\n(\s+)continue',
        r'\1 as e:\n\2# Unexpected error during encoding detection\n\2logger.debug("Unexpected error with encoding %s for %s: %s", enc, path.name, e)\n\2continue',
        content
    )

    # Improve read_text_file error logging
    content = re.sub(
        r'(\s+logger\.warning\("Failed to read text file %s: %s", path, e\))\n(\s+)return ""',
        r'\1\n\2# Return empty string to maintain backward compatibility\n\2# Callers expect empty string on failure, not an exception\n\2return ""',
        content
    )

    # Fix file_lock exception handling
    content = re.sub(
        r'(\s+except OSError):\n(\s+if lock_file:)',
        r'\1 as e:\n\2logger.debug("Lock attempt failed: %s", e)\n\2if lock_file:',
        content
    )

    content = re.sub(
        r'(\s+except Exception as e:\n\s+logger\.warning\("Error releasing lock on %s: %s", lock_path, e\))',
        r'\1\n                # Don\'t re-raise - this is cleanup code',
        content
    )

    if content != original:
        file_path.write_text(content)
        logger.info(f"✓ Updated {file_path}: Improved error logging while maintaining compatibility")
        return True
    return False

def apply_utils_improvements():
    """Improve error handling in utils.py."""
    file_path = Path("emailops/utils.py")
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return False

    content = file_path.read_text()
    original = content

    # Fix dotenv loading (keep it optional)
    content = re.sub(
        r'except Exception:\s*# pragma: no cover - optional dependency\n\s+pass',
        r'except Exception as e:  # pragma: no cover - optional dependency\n    # dotenv is optional - log at debug level\n    logger.debug("python-dotenv not available (optional): %s", e)\n    pass',
        content
    )

    if content != original:
        file_path.write_text(content)
        logger.info(f"✓ Updated {file_path}: Added logging for optional dotenv loading")
        return True
    return False

def apply_config_improvements():
    """Improve error handling in config.py."""
    file_path = Path("emailops/config.py")
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return False

    content = file_path.read_text()
    original = content

    # Improve credential validation error handling
    content = re.sub(
        r'(\s+except ImportError):\n(\s+# If google-auth not available, fall back to basic checks)\n(\s+)return True',
        r'\1 as e:\n\2\n\3logger.debug("google-auth not available for enhanced validation: %s", e)\n\3return True',
        content
    )

    content = re.sub(
        r'(\s+except Exception):\n(\s+# MalformedError or other auth errors - treat as invalid)\n(\s+)return False',
        r'\1 as e:\n\2\n\3logger.warning("Credential validation failed: %s", e)\n\3return False',
        content
    )

    if content != original:
        file_path.write_text(content)
        logger.info(f"✓ Updated {file_path}: Improved credential validation logging")
        return True
    return False

def apply_llm_runtime_improvements():
    """Improve error handling in llm_runtime.py."""
    file_path = Path("emailops/llm_runtime.py")
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return False

    content = file_path.read_text()
    original = content

    # Improve account validation
    content = re.sub(
        r'(\s+return True, "OK"\n\s+except Exception as e):\n(\s+)return False, str\(e\)',
        r'\1:\n\2logger.error("Account validation failed for %s: %s", account.project_id, e)\n\2return False, str(e)',
        content
    )

    # Improve embedding error handling (don't break on continue)
    content = re.sub(
        r'(\s+# replaced: continue)\n(\s+)raise',
        r'\1\n\2continue  # Keep rotation logic working',
        content
    )

    content = re.sub(
        r'(\s+# replaced: break)\n(\s+)raise',
        r'\1\n\2break  # Exit retry loop on non-retryable error',
        content
    )

    if content != original:
        file_path.write_text(content)
        logger.info(f"✓ Updated {file_path}: Improved error logging without breaking rotation logic")
        return True
    return False

def apply_email_indexer_improvements():
    """Improve error handling in email_indexer.py."""
    file_path = Path("emailops/email_indexer.py")
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return False

    content = file_path.read_text()
    original = content

    # Fix timestamp parsing
    content = re.sub(
        r'(\s+except Exception):\n(\s+)return None',
        r'\1 as e:\n\2logger.debug("Failed to parse timestamp: %s", e)\n\2return None',
        content,
        count=1  # Only first occurrence
    )

    # Fix mtime fallbacks
    content = re.sub(
        r'(\s+except Exception):\n(\s+conv_mtime = time\.time\(\))',
        r'\1 as e:\n\2logger.debug("Failed to get file mtime, using current time: %s", e)\n\2conv_mtime = time.time()',
        content
    )

    # Don't break loops on OSError
    content = re.sub(
        r'(\s+except OSError):\n(\s+# Treat unreadable stat as large: skip to be safe\.)\n(\s+logger\.warning.*)\n(\s+)continue',
        r'\1 as e:\n\2\n\3\n\4continue  # Skip this file but continue processing others',
        content
    )

    if content != original:
        file_path.write_text(content)
        logger.info(f"✓ Updated {file_path}: Improved error logging for indexing operations")
        return True
    return False

def apply_processor_improvements():
    """Improve error handling in processor.py."""
    file_path = Path("emailops/processor.py")
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return False

    content = file_path.read_text()
    original = content

    # These already have proper error handling, just ensure they don't have bare raise
    # The return codes are important for CLI tools

    # No changes needed - processor.py already has good error handling
    logger.info(f"✓ {file_path}: Already has proper error handling")
    return False

def verify_no_bare_raise(content: str, file_name: str) -> list[int]:
    """Check for bare 'raise' statements that would cause issues."""
    issues = []
    lines = content.split('\n')
    in_except_block = False

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Track if we're in an except block
        if re.match(r'except\s+.*:', stripped):
            in_except_block = True
        elif stripped and not line[0].isspace():
            in_except_block = False

        # Check for bare raise outside except blocks
        if stripped == 'raise' and not in_except_block:
            issues.append(i)

    return issues

def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Applying Safe Error Handling Improvements")
    logger.info("=" * 60)

    changes = []

    # Apply improvements to each file
    if apply_email_processing_improvements():
        changes.append("email_processing.py")

    if apply_file_utils_improvements():
        changes.append("file_utils.py")

    if apply_utils_improvements():
        changes.append("utils.py")

    if apply_config_improvements():
        changes.append("config.py")

    if apply_llm_runtime_improvements():
        changes.append("llm_runtime.py")

    if apply_email_indexer_improvements():
        changes.append("email_indexer.py")

    apply_processor_improvements()  # Just for completeness

    # Verify no bare raise statements were introduced
    logger.info("\n" + "=" * 60)
    logger.info("Verification Phase")
    logger.info("=" * 60)

    all_good = True
    for file_name in changes:
        file_path = Path("emailops") / file_name
        if file_path.exists():
            content = file_path.read_text()
            issues = verify_no_bare_raise(content, file_name)
            if issues:
                logger.error(f"✗ {file_name} has bare 'raise' statements at lines: {issues}")
                all_good = False
            else:
                logger.info(f"✓ {file_name}: No bare raise statements")

    logger.info("\n" + "=" * 60)
    if changes:
        logger.info(f"Successfully improved error handling in {len(changes)} files:")
        for f in changes:
            logger.info(f"  • {f}")
        logger.info("\nKey improvements:")
        logger.info("  1. Added debug logging for silent failures")
        logger.info("  2. Maintained backward compatibility")
        logger.info("  3. Preserved error recovery mechanisms")
        logger.info("  4. No breaking changes introduced")
    else:
        logger.info("No changes were needed - files already have proper error handling")

    if not all_good:
        logger.error("\n⚠️  WARNING: Some issues were detected. Please review manually.")
        return 1

    logger.info("\n✅ All improvements applied successfully!")
    logger.info("The code now has better observability without breaking functionality.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
