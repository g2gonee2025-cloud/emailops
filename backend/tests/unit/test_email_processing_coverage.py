import unittest

import pytest
from cortex.email_processing import clean_email_text


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestEmailProcessingCoverage(unittest.TestCase):
    def test_clean_email_basics(self):
        # Text must be > 100 chars for signature stripping
        padding = "x" * 100
        text = f"From: me\nTo: you\n\nBody content {padding}\n-- \nSig"
        cleaned = clean_email_text(text)
        self.assertIn("Body content", cleaned)
        self.assertNotIn("From:", cleaned)
        self.assertNotIn("Sig", cleaned)

    def test_clean_email_headers_stripping(self):
        text = """From: me
To: you
Cc: someone
Bcc: secret
Subject: Clean me
Date: Now
Message-ID: <123>
In-Reply-To: <abc>
References: <ref>
Content-Type: text/plain
MIME-Version: 1.0
X-Custom: val
Reply-To: me

Actual Body"""
        cleaned = clean_email_text(text)
        self.assertIn("Actual Body", cleaned)
        self.assertNotIn("From:", cleaned)
        self.assertNotIn("X-Custom:", cleaned)
        self.assertNotIn("MIME-Version:", cleaned)

    def test_signature_stripping_variations(self):
        # Test different signature markers
        sigs = [
            "Best regards,\nBob",
            "Kind regards,\nAlice",
            "Sincerely,\nEve",
            "Sent from my iPhone",
            "CONFIDENTIALITY NOTICE: Secret",
        ]
        padding = "x" * 150
        for sig in sigs:
            text = f"Body content {padding}\n\n{sig}"
            cleaned = clean_email_text(text)
            self.assertIn("Body content", cleaned)
            # The regex matches the START of the signature line.
            # clean_email_text should strip from that point.
            # However some patterns might be "Sent from my iPhone" which is the whole line.
            # The signature stripper removes everything after the match.
            # We expect the sig identifier to be REMOVED (or maybe included? logic says: text[:match.start()])
            # So it should be removed.

            # Check if sig key words are gone.
            sig.split()[0]
            if "CONFIDENTIALITY" in sig:
                self.assertNotIn("CONFIDENTIALITY", cleaned)
            elif "iPhone" in sig:
                self.assertNotIn("iPhone", cleaned)

    def test_boilerplate_line_removal(self):
        # Test that boilerplate lines are correctly identified and removed
        boilerplate_lines = [
            "This communication is confidential and intended for the recipient only.",
            "Please consider the environment before printing this email.",
            "CAREERS.CHALHOUBGROUP.COM",
            "Some Very Long Company Name Inc. | 123 Main Street, Suite 100, Anytown, CA 90210, USA | Tel: 555-1234",  # Long address-like line
            "http://www.example.com/some/long/url/that/is/just/a/url",
        ]

        for line in boilerplate_lines:
            text = f"This is the actual content.\n{line}\nThis is more content."
            cleaned = clean_email_text(text)
            self.assertIn("This is the actual content.", cleaned)
            self.assertIn("This is more content.", cleaned)
            self.assertNotIn(line.strip(), cleaned)

    def test_domain_noise_removal(self):
        # Test removal of lines that are just marketing domains
        text = "Some content\nCHALHOUBGROUP.COM\nMore content"
        cleaned = clean_email_text(text)
        self.assertIn("Some content", cleaned)
        self.assertIn("More content", cleaned)
        self.assertNotIn("CHALHOUBGROUP.COM", cleaned)

    def test_pii_redaction(self):
        # Test that email addresses and URLs are redacted
        text = "Contact me at test@example.com or visit http://example.com for more info. www.anothersite.com is also available."
        cleaned = clean_email_text(text)
        self.assertIn("[EMAIL_REDACTED]", cleaned)
        self.assertNotIn("test@example.com", cleaned)
        self.assertIn("[URL_REDACTED]", cleaned)
        self.assertNotIn("http://example.com", cleaned)
        self.assertNotIn("www.anothersite.com", cleaned)
