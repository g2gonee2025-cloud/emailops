import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from cortex.text_extraction import (
    _extract_eml,
    _extract_excel,
    _extract_msg,
    _extract_pdf,
    _extract_powerpoint,
    _extract_rtf,
    _extract_word_document,
    _extraction_cache,
    extract_text,
)


class TestTextExtraction(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_file = self.test_dir / "test.txt"
        self.test_file.write_text("Hello World")

        # Clear cache before each test
        _extraction_cache.clear()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_caching_behavior(self):
        """Test that extraction results are cached."""
        # First call
        text1 = extract_text(self.test_file)
        self.assertEqual(text1, "Hello World")

        # Verify it's in cache
        cache_key = (self.test_file.resolve(), None)
        self.assertIn(cache_key, _extraction_cache)

        # Modify file
        time.sleep(1.1)
        self.test_file.write_text("Modified Content")

        # Second call should get new content because mtime changed
        text2 = extract_text(self.test_file)
        self.assertEqual(text2, "Modified Content")

    @patch("cortex.text_extraction.TEXT_EXTENSIONS", {".custom"})
    def test_extract_text_file_path(self):
        f = self.test_dir / "test.custom"
        f.write_text("Simple Text")
        self.assertEqual(extract_text(f), "Simple Text")

    @patch("cortex.text_extraction.EMAIL_EXTENSIONS", {".eml"})
    def test_extract_eml_flow(self):
        f = self.test_dir / "test.eml"
        f.write_bytes(b"From: me@test.com\r\n\r\nBodyContent")

        # We assume the _extract_eml logic works (tested separately below),
        # just verifying the integration
        txt = extract_text(f)
        self.assertIn("BodyContent", txt)

    def test_extract_eml_content(self):
        f = self.test_dir / "sample.eml"
        # Create a multipart EML
        content = (
            b'Content-Type: multipart/alternative; boundary="boundary"\r\n'
            b"From: sender@example.com\r\n"
            b"Subject: Test Subject\r\n\r\n"
            b"--boundary\r\n"
            b'Content-Type: text/plain; charset="utf-8"\r\n\r\n'
            b"Plain text body.\r\n"
            b"--boundary--"
        )
        f.write_bytes(content)
        result = _extract_eml(f)
        self.assertIn("Test Subject", result)
        self.assertIn("Plain text body", result)

    def test_extract_msg_mock(self):
        """Mock external library extract_msg"""
        mock_msg_lib = MagicMock()
        mock_obj = MagicMock()
        mock_obj.body = "Msg Body"
        mock_obj.subject = "Msg Subject"
        mock_msg_lib.Message.return_value = mock_obj

        f = self.test_dir / "test.msg"
        f.touch()

        with patch.dict("sys.modules", {"extract_msg": mock_msg_lib}):
            result = _extract_msg(f)
            self.assertIn("Msg Body", result)

    def test_extract_pdf_mock_pypdf(self):
        """Mock pypdf"""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PDF Page 1"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader.is_encrypted = False

        mock_pypdf = MagicMock()
        mock_pypdf.PdfReader.return_value = mock_reader

        f = self.test_dir / "test.pdf"
        f.write_bytes(b"%PDF-1.4")

        with patch.dict("sys.modules", {"pypdf": mock_pypdf}):
            result = _extract_pdf(f, max_chars=None)
            self.assertEqual(result, "PDF Page 1")

    def test_extract_excel_mock(self):
        """Mock pandas for excel"""
        mock_df = MagicMock()
        mock_df.to_csv.return_value = "col1,col2\nval1,val2"
        mock_df.shape = (2, 2)
        mock_df.size = 4

        mock_excel_file = MagicMock()
        mock_excel_file.sheet_names = ["Sheet1"]
        mock_excel_file.__enter__.return_value = mock_excel_file
        mock_excel_file.__exit__.return_value = None

        mock_pd = MagicMock()
        mock_pd.ExcelFile.return_value = mock_excel_file
        mock_pd.read_excel.return_value = mock_df

        f = self.test_dir / "test.xlsx"
        f.touch()

        with patch.dict("sys.modules", {"pandas": mock_pd}):
            result = _extract_excel(f, ".xlsx", None)
            self.assertIn("val1,val2", result)

    def test_extract_rtf_mock(self):
        mock_rtf_mod = MagicMock()
        mock_rtf_mod.rtf_to_text.return_value = "RTF Content"
        f = self.test_dir / "test.rtf"
        f.write_bytes(b"rtf data")

        with patch.dict(
            "sys.modules", {"striprtf": MagicMock(), "striprtf.striprtf": mock_rtf_mod}
        ):
            result = _extract_rtf(f, None)
            self.assertEqual(result, "RTF Content")

    def test_extract_pptx_mock(self):
        mock_shape = MagicMock()
        mock_shape.text = "Slide Text"

        mock_slide = MagicMock()
        mock_slide.shapes = [mock_shape]

        mock_prs = MagicMock()
        mock_prs.slides = [mock_slide]

        mock_pptx = MagicMock()
        mock_pptx.Presentation.return_value = mock_prs

        f = self.test_dir / "test.pptx"
        f.touch()

        # Mock python-pptx
        with patch.dict("sys.modules", {"pptx": mock_pptx}):
            result = _extract_powerpoint(f, None)
            self.assertEqual(result, "Slide Text")

    def test_extract_docx_mock(self):
        mock_doc = MagicMock()
        mock_para = MagicMock()
        mock_para.text = "Doc Paragraph"
        mock_doc.paragraphs = [mock_para]
        mock_doc.tables = []

        mock_docx = MagicMock()
        mock_docx.Document.return_value = mock_doc

        f = self.test_dir / "test.docx"
        f.touch()

        with patch.dict("sys.modules", {"docx": mock_docx}):
            result = _extract_word_document(f, ".docx", None)
            self.assertEqual(result, "Doc Paragraph")


if __name__ == "__main__":
    unittest.main()
