import threading
import unittest
from unittest.mock import patch

from cortex.ingestion.pii import PIIEngine, get_pii_engine


class TestPIIEngine(unittest.TestCase):
    def test_redact_pii(self):
        """Verify that PII is redacted from text."""
        engine = PIIEngine(strict=False)
        text = "My email is test@example.com and my phone number is 555-123-4567."
        redacted_text = engine.redact(text)
        self.assertIn("<<EMAIL>>", redacted_text)
        self.assertIn("<<PHONE>>", redacted_text)
        self.assertNotIn("test@example.com", redacted_text)
        self.assertNotIn("555-123-4567", redacted_text)

    def test_detect_pii(self):
        """Verify that PII is detected in text."""
        engine = PIIEngine(strict=False)
        text = "My email is test@example.com."
        entities = engine.detect(text)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].entity_type, "EMAIL_ADDRESS")
        self.assertEqual(entities[0].text, "test@example.com")

    def test_thread_safety(self):
        """Verify that the PII engine singleton is thread-safe."""
        engines = set()

        def get_engine():
            engines.add(get_pii_engine())

        threads = [threading.Thread(target=get_engine) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(engines), 1)

    @patch("cortex.ingestion.pii._HAS_PRESIDIO", False)
    def test_regex_fallback(self):
        """Verify that the regex fallback is used when Presidio is not available."""
        engine = PIIEngine(strict=False)
        text = "My email is test@example.com."
        redacted_text = engine.redact(text)
        self.assertIn("<<EMAIL>>", redacted_text)
        self.assertNotIn("test@example.com", redacted_text)

    @patch("cortex.ingestion.pii._OperatorConfig")
    @patch("cortex.ingestion.pii._AnonymizerEngine")
    @patch("cortex.ingestion.pii._AnalyzerEngine")
    @patch("cortex.ingestion.pii._HAS_PRESIDIO", True)
    def test_result_merging(self, mock_analyzer_cls, mock_anonymizer_cls, mock_operator_cfg_cls):
        """Verify that Presidio and regex results are merged correctly."""
        # This email will be detected by both Presidio and the regex fallback.
        text = "Contact me at test@example.com"

        # Setup mocks
        mock_analyzer_instance = mock_analyzer_cls.return_value
        # Mock Presidio to return one result.
        mock_analyzer_instance.analyze.return_value = [
            unittest.mock.Mock(
                entity_type="EMAIL_ADDRESS", start=14, end=30, score=0.9
            )
        ]

        engine = PIIEngine(strict=False)
        # The regex fallback will also detect the email.
        entities = engine.detect(text)

        # The results should be de-duplicated, resulting in only one entity.
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].entity_type, "EMAIL_ADDRESS")
        self.assertEqual(entities[0].score, 0.9)


if __name__ == "__main__":
    unittest.main()
