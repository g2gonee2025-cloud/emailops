import unittest

# Adjust import to point to the actual location
from cortex.ingestion.core_manifest import (
    extract_metadata_lightweight,
    extract_participants_detailed,
)


class TestCoreManifest(unittest.TestCase):
    def test_extract_metadata_lightweight_dates(self):
        """Test date parsing in metadata extraction."""
        # Case 1: ISO Format with Z
        manifest_iso = {
            "messages": [
                {"date": "2023-01-01T10:00:00Z"},
                {"date": "2023-01-02T10:00:00Z"},
            ]
        }
        meta_iso = extract_metadata_lightweight(manifest_iso)
        # Note: The *current* implementation might just pass strings through or fail on some formats.
        # The *improved* implementation (which we are testing for) handles Z correctly.
        # We'll assert on the expected behavior of the improved code.
        self.assertEqual(meta_iso["start_date"], "2023-01-01T10:00:00+00:00")
        self.assertEqual(meta_iso["end_date"], "2023-01-02T10:00:00+00:00")

    def test_extract_metadata_lightweight_timestamp(self):
        """Test timestamp parsing (int/float)."""
        manifest_ts = {
            "messages": [
                {"date": 1672567200},  # 2023-01-01 10:00:00 UTC
            ]
        }
        meta_ts = extract_metadata_lightweight(manifest_ts)
        self.assertEqual(meta_ts["start_date"], "2023-01-01T10:00:00+00:00")

    def test_extract_participants_deduplication(self):
        """Test participant deduplication."""
        manifest = {
            "messages": [
                {
                    "from": {"name": "Alice", "smtp": "alice@example.com"},
                    "to": [{"name": "Bob", "smtp": "bob@example.com"}],
                    "cc": [{"name": "Alice", "smtp": "alice@example.com"}],  # Duplicate
                },
                {
                    "from": {"name": "Bob", "smtp": "bob@example.com"},
                    "to": [{"name": "Charlie", "smtp": "charlie@example.com"}],
                },
            ]
        }
        # Testing the detailed participant extraction which has deduplication logic
        participants = extract_participants_detailed(manifest)
        self.assertEqual(len(participants), 3)
        emails = {p["email"] for p in participants}
        self.assertSetEqual(
            emails, {"alice@example.com", "bob@example.com", "charlie@example.com"}
        )


if __name__ == "__main__":
    unittest.main()
