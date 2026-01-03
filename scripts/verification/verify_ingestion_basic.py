import sys
import os
from pathlib import Path

try:
    root_dir = Path(__file__).resolve().parents[2]
    sys.path.append(str(root_dir / "backend" / "src"))
except IndexError:
    print(
        "Error: Could not determine the project root directory from the script's location.",
        file=sys.stderr,
    )
    print(
        "Please ensure the script is located within the project structure, e.g., 'scripts/verification/'.",
        file=sys.stderr,
    )
    sys.exit(1)
import unittest
from unittest.mock import patch

from cortex.config.loader import EmailOpsConfig
from cortex.ingestion.pii import redact_pii
from cortex.ingestion.text_preprocessor import get_text_preprocessor
from cortex.security.validators import (
    validate_file_result,
)


class TestSecurityAndIngestion(unittest.TestCase):
    def setUp(self):
        # Force strict=False for PII to allow regex fallback.
        # Use autospec=True to create a mock that matches the signature of the real
        # get_config function, ensuring type safety.
        self.patcher = patch("cortex.ingestion.pii.get_config", autospec=True)
        self.mock_get_config = self.patcher.start()

        # Configure the mock's return value to simulate the desired PII settings.
        # The return value is a mock of EmailOpsConfig due to autospec.
        self.mock_get_config.return_value.pii.strict = False
        self.mock_get_config.return_value.pii.enabled = True

    def tearDown(self):
        self.patcher.stop()

    def test_security_validators(self):
        # Test 1: Reject parent traversal.
        # The updated validator detects traversal relative to a base directory.
        base_dir = Path("/var/www")
        # Use a path with an allowed extension for the test.
        relative_path_with_traversal = os.path.join("..", "..", "etc", "test.log")

        result = validate_file_result(
            relative_path_with_traversal, base_directory=base_dir, must_exist=False
        )

        self.assertFalse(result.ok, "Should satisfy invariant: reject parent traversal")
        # The Err object holds the error message, which can be accessed by converting it to a string.
        self.assertIn("Path traversal detected", str(result))

    def test_pii_redaction(self):
        text = "Contact me at user@example.com or 555-123-4567."
        redacted = redact_pii(text)
        self.assertIsInstance(redacted, str)

        # In regex fallback mode, we expect <<EMAIL>> and <<PHONE>> (or similar)
        # Note: The codebase logic might default to specific placeholders.
        self.assertNotIn("@", redacted, "Email should be redacted")
        self.assertNotIn("555-123-4567", redacted, "Phone should be redacted")
        self.assertIn("<<EMAIL>>", redacted)
        self.assertIn("<<PHONE>>", redacted)

    def test_text_preprocessor(self):
        processor = get_text_preprocessor()
        raw_text = "Hello   World! \n\n This is a test."
        cleaned, meta = processor.prepare_for_indexing(
            raw_text, text_type="attachment", tenant_id="test-tenant"
        )
        self.assertNotIn("  ", cleaned, "Whitespace should be collapsed for attachments")
        self.assertIsInstance(meta, dict)
        self.assertIn("pre_cleaned", meta)
        self.assertTrue(meta.get("pre_cleaned"))


if __name__ == "__main__":
    unittest.main()
