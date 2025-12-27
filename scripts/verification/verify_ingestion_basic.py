import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir / "backend" / "src"))
import unittest
from pathlib import Path

sys.path.append(str(Path("backend/src").resolve()))

from unittest.mock import patch

from cortex.ingestion.pii import redact_pii
from cortex.ingestion.text_preprocessor import get_text_preprocessor
from cortex.security.validators import (
    validate_file_result,
    validate_project_id,
)


class TestSecurityAndIngestion(unittest.TestCase):
    def setUp(self):
        # Force strict=False for PII to allow regex fallback
        self.patcher = patch("cortex.ingestion.pii.get_config")
        self.mock_config = self.patcher.start()
        self.mock_config.return_value.pii.strict = False

    def tearDown(self):
        self.patcher.stop()

    def test_security_validators(self):
        print("Testing Security Validators...")
        # Test 1: Reject parent traversal
        result = validate_file_result(
            "/var/www/../../etc/passwd", allow_parent_traversal=False, must_exist=False
        )
        assert not result.ok, "Should satisfy invariant: reject parent traversal"
        print("PASS: Parent traversal rejected")

        # Test 2: Validate GCP Project ID
        result = validate_project_id("my-project-123")
        assert result.ok, "Should satisfy invariant: valid project ID"
        print("PASS: Valid project ID accepted")

        result = validate_project_id("Invalid_Project")
        assert not result.ok, "Should satisfy invariant: invalid project ID rejected"
        print("PASS: Invalid project ID rejected")

    def test_pii_redaction(self):
        print("\nTesting PII Redaction details...")
        text = "Contact me at user@example.com or 555-123-4567."
        redacted = redact_pii(text)
        print(f"Original: {text}")
        print(f"Redacted: {redacted}")

        # In regex fallback mode, we expect <<EMAIL>> and <<PHONE>> (or similar)
        # Note: The codebase logic might default to specific placeholders.
        assert "@" not in redacted, "Email should be redacted"
        assert "555-123-4567" not in redacted, "Phone should be redacted"
        print("PASS: PII redacted")

    def test_text_preprocessor(self):
        print("\nTesting Text Preprocessor...")
        processor = get_text_preprocessor()
        raw_text = "Hello   World! \n\n This is a test."
        cleaned, meta = processor.prepare_for_indexing(
            raw_text, text_type="attachment", tenant_id="test-tenant"
        )
        print(f"Cleaned: '{cleaned}'")
        assert "  " not in cleaned, "Whitespace should be collapsed for attachments"
        assert meta["pre_cleaned"] is True
        print("PASS: Text preprocessing")


if __name__ == "__main__":
    unittest.main()
