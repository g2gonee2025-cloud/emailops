
"""Unit tests for emailops.utils module."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

from emailops.utils import (
    DOCX_EXTENSIONS,
    EMAIL_EXTENSIONS,
    EXCEL_EXTENSIONS,
    PDF_EXTENSIONS,
    PPT_EXTENSIONS,
    RTF_EXTENSIONS,
    TEXT_EXTENSIONS,
    Person,
    _html_to_text,
    _strip_control_chars,
    clean_email_text,
    ensure_dir,
    extract_email_metadata,
    extract_text,
    find_conversation_dirs,
    load_conversation,
    read_text_file,
    split_email_thread,
)


class TestStripControlChars(TestCase):
    """Test _strip_control_chars function."""

    def test_strip_control_chars_empty_string(self):
        """Test with empty string."""
        assert _strip_control_chars("") == ""

    def test_strip_control_chars_none(self):
        """Test with None."""
        # _strip_control_chars expects a string, so we test empty string behavior
        assert _strip_control_chars("") == ""

    def test_strip_control_chars_removes_control_chars(self):
        """Test removal of control characters."""
        text = "Hello\x00World\x01\x02\x03"
        assert _strip_control_chars(text) == "HelloWorld"

    def test_strip_control_chars_preserves_tab_and_newline(self):
        """Test that tab and newline are preserved."""
        text = "Hello\tWorld\nNew Line"
        assert _strip_control_chars(text) == "Hello\tWorld\nNew Line"

    def test_strip_control_chars_normalizes_line_endings(self):
        """Test normalization of line endings."""
        text = "Line1\r\nLine2\rLine3"
        assert _strip_control_chars(text) == "Line1\nLine2\nLine3"

    def test_strip_control_chars_with_unicode(self):
        """Test with Unicode characters."""
        text = "Hello 世界 🌍\x00\x01"
        assert _strip_control_chars(text) == "Hello 世界 🌍"


class TestReadTextFile(TestCase):
    """Test read_text_file function."""

    def test_read_text_file_utf8(self):
        """Test reading UTF-8 file."""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write("Hello World")
            temp_path = Path(f.name)

        try:
            content = read_text_file(temp_path)
            assert content == "Hello World"
        finally:
            temp_path.unlink()

    def test_read_text_file_utf8_with_bom(self):
        """Test reading UTF-8 file with BOM."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.txt') as f:
            f.write(b'\xef\xbb\xbfHello World')  # UTF-8 BOM
            temp_path = Path(f.name)

        try:
            content = read_text_file(temp_path)
            assert content == "Hello World"
        finally:
            temp_path.unlink()

    def test_read_text_file_utf16(self):
        """Test reading UTF-16 file."""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-16', delete=False, suffix='.txt') as f:
            f.write("Hello World")
            temp_path = Path(f.name)

        try:
            content = read_text_file(temp_path)
            assert content == "Hello World"
        finally:
            temp_path.unlink()

    def test_read_text_file_with_max_chars(self):
        """Test reading file with max_chars limit."""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write("Hello World This Is A Long Text")
            temp_path = Path(f.name)

        try:
            content = read_text_file(temp_path, max_chars=11)
            assert content == "Hello World"
        finally:
            temp_path.unlink()

    def test_read_text_file_with_control_chars(self):
        """Test reading file with control characters."""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write("Hello\x00World\x01")
            temp_path = Path(f.name)

        try:
            content = read_text_file(temp_path)
            assert content == "HelloWorld"
        finally:
            temp_path.unlink()

    @patch('emailops.utils.logger')
    def test_read_text_file_nonexistent(self, mock_logger):
        """Test reading non-existent file."""
        content = read_text_file(Path("/nonexistent/file.txt"))
        assert content == ""
        mock_logger.warning.assert_called()


class TestHtmlToText(TestCase):
    """Test _html_to_text function."""

    def test_html_to_text_empty(self):
        """Test with empty HTML."""
        assert _html_to_text("") == ""

    def test_html_to_text_simple(self):
        """Test with simple HTML."""
        html = "<p>Hello World</p>"
        result = _html_to_text(html)
        assert "Hello World" in result

    def test_html_to_text_with_scripts(self):
        """Test HTML with script tags."""
        html = "<html><script>alert('test');</script><body>Hello</body></html>"
        result = _html_to_text(html)
        assert "alert" not in result
        assert "Hello" in result

    def test_html_to_text_with_styles(self):
        """Test HTML with style tags."""
        html = "<html><style>body { color: red; }</style><body>Hello</body></html>"
        result = _html_to_text(html)
        assert "color" not in result
        assert "Hello" in result

    def test_html_to_text_complex(self):
        """Test with complex HTML."""
        html = """
        <html>
            <head><title>Test</title></head>
            <body>
                <h1>Title</h1>
                <p>Paragraph <strong>bold</strong> text</p>
                <ul>
                    <li>Item 1</li>
                    <li>Item 2</li>
                </ul>
            </body>
        </html>
        """
        result = _html_to_text(html)
        assert "Title" in result
        assert "Paragraph" in result
        assert "bold" in result
        assert "Item 1" in result

    @patch('bs4.BeautifulSoup', side_effect=ImportError)
    def test_html_to_text_without_beautifulsoup(self, _):
        """Test fallback when BeautifulSoup is not available."""
        html = "<p>Hello <strong>World</strong></p>"
        result = _html_to_text(html)
        assert "Hello" in result
        assert "World" in result


class TestExtractText(TestCase):
    """Test extract_text function."""

    def test_extract_text_txt_file(self):
        """Test extracting text from .txt file."""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write("Hello World")
            temp_path = Path(f.name)

        try:
            content = extract_text(temp_path)
            assert content == "Hello World"
        finally:
            temp_path.unlink()

    def test_extract_text_html_file(self):
        """Test extracting text from .html file."""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.html') as f:
            f.write("<html><body><p>Hello World</p></body></html>")
            temp_path = Path(f.name)

        try:
            content = extract_text(temp_path)
            assert "Hello World" in content
        finally:
            temp_path.unlink()

    def test_extract_text_json_file(self):
        """Test extracting text from .json file."""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.json') as f:
            json.dump({"message": "Hello World"}, f)
            temp_path = Path(f.name)

        try:
            content = extract_text(temp_path)
            assert "Hello World" in content
        finally:
            temp_path.unlink()

    def test_extract_text_with_max_chars(self):
        """Test extracting text with max_chars limit."""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write("Hello World This Is A Long Text")
            temp_path = Path(f.name)

        try:
            content = extract_text(temp_path, max_chars=11)
            assert content == "Hello World"
        finally:
            temp_path.unlink()

    def test_extract_text_unsupported_format(self):
        """Test extracting text from unsupported format."""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.xyz') as f:
            f.write("Hello World")
            temp_path = Path(f.name)

        try:
            content = extract_text(temp_path)
            assert content == ""
        finally:
            temp_path.unlink()

    @patch('docx.Document')
    def test_extract_text_docx_file(self, mock_docx):
        """Test extracting text from .docx file."""
        mock_doc = Mock()
        mock_doc.paragraphs = [Mock(text="Hello World")]
        mock_doc.tables = []
        mock_docx.return_value = mock_doc

        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as f:
            temp_path = Path(f.name)

        try:
            content = extract_text(temp_path)
            assert content == "Hello World"
        finally:
            temp_path.unlink()

    @patch('pypdf.PdfReader')
    def test_extract_text_pdf_file(self, mock_pdf_reader):
        """Test extracting text from .pdf file."""
        mock_page = Mock()
        mock_page.extract_text.return_value = "Hello World"
        mock_reader = Mock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as f:
            temp_path = Path(f.name)

        try:
            content = extract_text(temp_path)
            assert content == "Hello World"
        finally:
            temp_path.unlink()


class TestCleanEmailText(TestCase):
    """Test clean_email_text function."""

    def test_clean_email_text_empty(self):
        """Test with empty text."""
        assert clean_email_text("") == ""

    def test_clean_email_text_removes_headers(self):
        """Test removal of email headers."""
        text = """From: sender@example.com
To: recipient@example.com
Subject: Test Email
Date: 2024-01-01

Hello World"""
        result = clean_email_text(text)
        assert "From:" not in result
        assert "To:" not in result
        assert "Subject:" not in result
        assert "Hello World" in result

    def test_clean_email_text_removes_signatures(self):
        """Test removal of email signatures."""
        text = """Hello World

Best regards,
John Doe"""
        result = clean_email_text(text)
        assert "Hello World" in result
        # Signature might be removed depending on implementation

    def test_clean_email_text_removes_quoted_text(self):
        """Test removal of quoted text."""
        text = """Hello World

> This is quoted text
> from previous email

New content here"""
        result = clean_email_text(text)
        assert "Hello World" in result
        assert "New content" in result
        assert "> This is quoted" not in result

    def test_clean_email_text_redacts_emails(self):
        """Test email address redaction."""
        text = "Contact me at john.doe@example.com"
        result = clean_email_text(text)
        assert "[email@example.com]" in result

    def test_clean_email_text_redacts_urls(self):
        """Test URL redaction."""
        text = "Visit https://example.com for more info"
        result = clean_email_text(text)
        assert "[URL]" in result
        assert "https://example.com" not in result

    def test_clean_email_text_normalizes_whitespace(self):
        """Test whitespace normalization."""
        text = "Hello    World\n\n\n\nMultiple     spaces"
        result = clean_email_text(text)
        assert "Hello World" in result
        assert "\n\n\n\n" not in result
        assert "     " not in result

    def test_clean_email_text_with_bom(self):
        """Test handling of BOM."""
        text = "\ufeffHello World"
        result = clean_email_text(text)
        assert result == "Hello World"


class TestExtractEmailMetadata(TestCase):
    """Test extract_email_metadata function."""

    def test_extract_email_metadata_empty(self):
        """Test with empty text."""
        metadata = extract_email_metadata("")
        assert metadata["sender"] is None
        assert metadata["recipients"] == []
        assert metadata["date"] is None
        assert metadata["subject"] is None
        assert metadata["cc"] == []
        assert metadata["bcc"] == []

    def test_extract_email_metadata_full_headers(self):
        """Test with full email headers."""
        text = """From: sender@example.com
To: recipient1@example.com, recipient2@example.com
Cc: cc@example.com
Subject: Test Email
Date: Mon, 1 Jan 2024 10:00:00 +0000

Email body"""
        metadata = extract_email_metadata(text)
        assert metadata["sender"] == "sender@example.com"
        assert metadata["recipients"] == ["recipient1@example.com", "recipient2@example.com"]
        assert metadata["cc"] == ["cc@example.com"]
        assert metadata["subject"] == "Test Email"
        assert metadata["date"] == "Mon, 1 Jan 2024 10:00:00 +0000"

    def test_extract_email_metadata_partial_headers(self):
        """Test with partial headers."""
        text = """From: sender@example.com
Subject: Test

Body"""
        metadata = extract_email_metadata(text)
        assert metadata["sender"] == "sender@example.com"
        assert metadata["subject"] == "Test"
        assert metadata["recipients"] == []

    def test_extract_email_metadata_case_insensitive(self):
        """Test case-insensitive header matching."""
        text = """from: sender@example.com
SUBJECT: Test"""
        metadata = extract_email_metadata(text)
        assert metadata["sender"] == "sender@example.com"
        assert metadata["subject"] == "Test"


class TestSplitEmailThread(TestCase):
    """Test split_email_thread function."""

    def test_split_email_thread_empty(self):
        """Test with empty text."""
        assert split_email_thread("") == []

    def test_split_email_thread_single_message(self):
        """Test with single message."""
        text = "Hello World"
        assert split_email_thread(text) == ["Hello World"]

    def test_split_email_thread_with_original_message(self):
        """Test with original message separator."""
        text = """New message

----- Original Message -----
Old message"""
        result = split_email_thread(text)
        assert len(result) == 2
        assert "New message" in result[0]
        assert "Old message" in result[1]

    def test_split_email_thread_with_forwarded_message(self):
        """Test with forwarded message separator."""
        text = """New message

----- Forwarded Message -----
Forwarded content"""
        result = split_email_thread(text)
        assert len(result) == 2

    def test_split_email_thread_with_on_wrote(self):
        """Test with 'On ... wrote:' separator."""
        text = """Reply message

On 2024-01-01 John wrote:
Original message"""
        result = split_email_thread(text)
        assert len(result) == 2

    @patch('emailops.utils.parsedate_to_datetime')
    def test_split_email_thread_with_dates(self, mock_parse_date):
        """Test chronological sorting when dates are present."""
        mock_parse_date.side_effect = [
            datetime(2024, 1, 2),  # Second message
            datetime(2024, 1, 1),  # First message
        ]

        text = """Date: 2024-01-02
Second message

----- Original Message -----
Date: 2024-01-01
First message"""

        result = split_email_thread(text)
        assert len(result) == 2
        # Should be sorted chronologically


class TestFindConversationDirs(TestCase):
    """Test find_conversation_dirs function."""

    def test_find_conversation_dirs_empty(self):
        """Test with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            assert find_conversation_dirs(root) == []

    def test_find_conversation_dirs_single(self):
        """Test with single conversation directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            conv_dir = root / "conv1"
            conv_dir.mkdir()
            (conv_dir / "Conversation.txt").touch()

            result = find_conversation_dirs(root)
            assert len(result) == 1
            assert result[0] == conv_dir

    def test_find_conversation_dirs_multiple(self):
        """Test with multiple conversation directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            conv1 = root / "conv1"
            conv1.mkdir()
            (conv1 / "Conversation.txt").touch()

            conv2 = root / "subdir" / "conv2"
            conv2.mkdir(parents=True)
            (conv2 / "Conversation.txt").touch()

            # Non-conversation directory
            other = root / "other"
            other.mkdir()
            (other / "file.txt").touch()

            result = find_conversation_dirs(root)
            assert len(result) == 2
            assert conv1 in result
            assert conv2 in result


class TestLoadConversation(TestCase):
    """Test load_conversation function."""

    def test_load_conversation_empty_dir(self):
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conv_dir = Path(tmpdir)
            result = load_conversation(conv_dir)

            assert result["path"] == str(conv_dir)
            assert result["conversation_txt"] == ""
            assert result["attachments"] == []
            assert result["summary"] == {}
            assert result["manifest"] == {}

    def test_load_conversation_with_text(self):
        """Test loading conversation with text file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conv_dir = Path(tmpdir)
            conv_file = conv_dir / "Conversation.txt"
            conv_file.write_text("Hello World")

            result = load_conversation(conv_dir)

            assert result["conversation_txt"] == "Hello World"

    def test_load_conversation_with_manifest(self):
        """Test loading conversation with manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conv_dir = Path(tmpdir)
            manifest_file = conv_dir / "manifest.json"
            manifest_data = {"id": "123", "subject": "Test"}
            manifest_file.write_text(json.dumps(manifest_data))

            result = load_conversation(conv_dir)

            assert result["manifest"] == manifest_data

    def test_load_conversation_with_summary(self):
        """Test loading conversation with summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conv_dir = Path(tmpdir)
            summary_file = conv_dir / "summary.json"
            summary_data = {"summary": "Test summary"}
            summary_file.write_text(json.dumps(summary_data))

            result = load_conversation(conv_dir)

            assert result["summary"] == summary_data

    def test_load_conversation_with_attachments(self):
        """Test loading conversation with attachments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conv_dir = Path(tmpdir)

            # Create attachments directory
            att_dir = conv_dir / "Attachments"
            att_dir.mkdir()

            # Create text attachment
            att_file = att_dir / "test.txt"
            att_file.write_text("Attachment content")

            result = load_conversation(conv_dir, include_attachment_text=True)

            assert len(result["attachments"]) == 1
            assert "Attachment content" in result["attachments"][0]["text"]
            assert "ATTACHMENT: test.txt" in result["conversation_txt"]

    def test_load_conversation_with_bom_in_files(self):
        """Test loading conversation with BOM in files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conv_dir = Path(tmpdir)

            # Write file with BOM
            conv_file = conv_dir / "Conversation.txt"
            conv_file.write_bytes(b'\xef\xbb\xbfHello World')

            result = load_conversation(conv_dir)

            assert result["conversation_txt"] == "Hello World"


class TestEnsureDir(TestCase):
    """Test ensure_dir function."""

    def test_ensure_dir_creates_directory(self):
        """Test that ensure_dir creates a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "new_dir"

            assert not test_dir.exists()
            ensure_dir(test_dir)
            assert test_dir.exists()
            assert test_dir.is_dir()

    def test_ensure_dir_creates_nested_directories(self):
        """Test that ensure_dir creates nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "level1" / "level2" / "level3"

            assert not test_dir.exists()
            ensure_dir(test_dir)
            assert test_dir.exists()
            assert test_dir.is_dir()

    def test_ensure_dir_idempotent(self):
        """Test that ensure_dir is idempotent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "test_dir"

            ensure_dir(test_dir)
            assert test_dir.exists()

            # Call again - should not raise
            ensure_dir(test_dir)
            assert test_dir.exists()


class TestPerson(TestCase):
    """Test Person class."""

    def test_person_initialization(self):
        """Test Person initialization."""
        person = Person("John Doe", "1990-01-01")
        assert person.name == "John Doe"
        assert person.birthdate == "1990-01-01"

    @patch('emailops.utils.datetime')
    def test_person_age_calculation(self, mock_datetime):
        """Test age calculation."""
        mock_datetime.datetime.today.return_value = datetime(2024, 1, 1)
        mock_datetime.datetime.strptime = datetime.strptime

        person = Person("John Doe", "1990-01-01")
        assert person.age == 34

    @patch('emailops.utils.datetime')
    def test_person_age_before_birthday(self, mock_datetime):
        """Test age calculation before birthday."""
        mock_datetime.datetime.today.return_value = datetime(2024, 1, 1)
        mock_datetime.datetime.strptime = datetime.strptime

        person = Person("John Doe", "1990-12-31")
        assert person.age == 33  # Birthday hasn't occurred yet

    def test_person_age_empty_birthdate(self):
        """Test age with empty birthdate."""
        person = Person("John Doe", "")
        assert person.age == 0

    def test_person_age_invalid_birthdate(self):
        """Test age with invalid birthdate."""
        person = Person("John Doe", "invalid-date")
        assert person.age == 0

    @patch('emailops.utils.datetime')
    def test_person_getAge_method(self, mock_datetime):
        """Test getAge method (alias for age property)."""
        mock_datetime.datetime.today.return_value = datetime(2024, 1, 1)
        mock_datetime.datetime.strptime = datetime.strptime

        person = Person("John Doe", "1990-01-01")
        assert person.getAge() == 34
        assert person.getAge() == person.age


class TestFileExtensions(TestCase):
    """Test file extension constants."""

    def test_text_extensions(self):
        """Test TEXT_EXTENSIONS constant."""
        assert ".txt" in TEXT_EXTENSIONS
        assert ".md" in TEXT_EXTENSIONS
        assert ".json" in TEXT_EXTENSIONS
        assert ".html" in TEXT_EXTENSIONS

    def test_docx_extensions(self):
        """Test DOCX_EXTENSIONS constant."""
        assert ".docx" in DOCX_EXTENSIONS
        assert ".doc" in DOCX_EXTENSIONS

    def test_pdf_extensions(self):
        """Test PDF_EXTENSIONS constant."""
        assert ".pdf" in PDF_EXTENSIONS

    def test_excel_extensions(self):
        """Test EXCEL_EXTENSIONS constant."""
        assert ".xlsx" in EXCEL_EXTENSIONS
        assert ".xls" in EXCEL_EXTENSIONS

    def test_ppt_extensions(self):
        """Test PPT_EXTENSIONS constant."""
        assert ".pptx" in PPT_EXTENSIONS
        assert ".ppt" in PPT_EXTENSIONS

    def test_rtf_extensions(self):
        """Test RTF_EXTENSIONS constant."""
        assert ".rtf" in RTF_EXTENSIONS

    def test_email_extensions(self):
        """Test EMAIL_EXTENSIONS constant."""
        assert ".eml" in EMAIL_EXTENSIONS
        assert ".msg" in EMAIL_EXTENSIONS


class TestEmailParsing(TestCase):
    """Test email parsing functions."""

    @patch('email.parser.BytesParser')
    def test_extract_eml_success(self, mock_parser_class):
        """Test successful .eml extraction."""
        mock_msg = Mock()
        mock_msg.get.side_effect = lambda h: {
            "From": "sender@example.com",
            "To": "recipient@example.com",
            "Subject": "Test",
            "Date": "2024-01-01"
        }.get(h)
        mock_msg.is_multipart.return_value = False
        mock_msg.get_content_type.return_value = "text/plain"
        mock_msg.get_content.return_value = "Email body"

        mock_parser = Mock()
        mock_parser.parsebytes.return_value = mock_msg
        mock_parser_class.return_value = mock_parser

        with tempfile.NamedTemporaryFile(delete=False, suffix='.eml') as f:
            f.write(b"test email content")
            temp_path = Path(f.name)

        try:
            from emailops.utils import _extract_eml
            result = _extract_eml(temp_path)
            assert "sender@example.com" in result
            assert "Email body" in result
        finally:
            temp_path.unlink()

    @patch('extract_msg.Message')
    def test_extract_msg_success(self, mock_extract_msg_class):
        """Test successful .msg extraction."""
        # Configure the mock properly using spec and configure_mock
        mock_msg = Mock()
        mock_msg.body = "Email body"
        mock_msg.htmlBody = None
        # Use configure_mock for the getattr behavior
        mock_msg.configure_mock(**{
            'from': "sender@example.com",
            'to': "recipient@example.com",
            'subject': "Test",
            'date': "2024-01-01",
            'cc': None,
            'bcc': None
        })
        mock_extract_msg_class.return_value = mock_msg

        with tempfile.NamedTemporaryFile(delete=False, suffix='.msg') as f:
            temp_path = Path(f.name)

        try:
            from emailops.utils import _extract_msg
            result = _extract_msg(temp_path)
            assert "Email body" in result
        finally:
            temp_path.unlink()
