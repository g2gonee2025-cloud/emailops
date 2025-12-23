import unittest

import pytest
from cortex.archive.email_processing_legacy import (
    extract_email_metadata,
    split_email_thread,
)
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

    def test_extract_metadata(self):
        text = "From: bob@example.com\nTo: alice@example.com, eve@example.com\nSubject: Hi\nCc: charlie@example.com\nBcc: ghost@example.com\nDate: Mon, 1 Jan 2024 10:00:00 -0000\nSome-Header: val\n\nBody"
        meta = extract_email_metadata(text)
        self.assertEqual(meta["sender"], "bob@example.com")
        self.assertEqual(meta["subject"], "Hi")
        self.assertEqual(len(meta["recipients"]), 2)
        self.assertIn("charlie@example.com", meta["cc"])
        self.assertIn("ghost@example.com", meta["bcc"])
        self.assertIn("Mon, 1 Jan 2024", meta["date"])

    def test_extract_metadata_alt_headers(self):
        # Test Sent vs Date, and multiline headers handling if implemented (regex is ^Header:.*$)
        text = "From: me\nTo: you\nSent: Today\nSubject: Re: Hi\n\nMsg"
        meta = extract_email_metadata(text)
        self.assertEqual(meta["date"], "Today")

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

    def test_split_thread(self):
        text = "Msg1\n-----Original Message-----\nMsg2"
        msgs = split_email_thread(text)
        self.assertTrue(len(msgs) >= 2)

        text2 = "Msg1\nOn 01/01/2024, bob wrote:\n> Quoted"
        msgs2 = split_email_thread(text2)
        self.assertTrue(len(msgs2) >= 2)
