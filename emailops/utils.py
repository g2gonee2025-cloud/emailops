from __future__ import annotations

import datetime
import json
import logging
import os
import re
from pathlib import Path

"""
Utilities for reading conversation exports and attachments.

Notes:
- Optional dependencies are imported lazily and failures are handled gracefully.
- Text extraction is conservative: we only parse a handful of common formats.
- All extracted text is sanitized to remove control characters and normalize
  newlines so downstream JSON serialization and indexing are reliable.
"""

# Best‑effort env loading (safe if python-dotenv isn't installed)
try:  # pragma: no cover
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - optional dependency
    pass

# Library-safe logging: no basicConfig at module level
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File type support
# ---------------------------------------------------------------------------
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".log",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".xml",
    ".html",
    ".htm",
}
DOCX_EXTENSIONS = {".docx", ".doc"}
PDF_EXTENSIONS = {".pdf"}
EXCEL_EXTENSIONS = {".xlsx", ".xls"}
PPT_EXTENSIONS = {".pptx", ".ppt"}
RTF_EXTENSIONS = {".rtf"}
EMAIL_EXTENSIONS = {".eml", ".msg"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Control chars (except TAB, LF) frequently break JSON & indexing
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")


def _strip_control_chars(s: str) -> str:
    """Remove non-printable control characters and normalize newlines."""
    if not s:
        return ""
    # Normalize CRLF/CR -> LF and strip control characters
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL_CHARS.sub("", s)


def read_text_file(path: Path, *, max_chars: int | None = None) -> str:
    """
    Read a text file with multiple fallbacks and basic sanitation.

    Args:
        path: Path to the text file
        max_chars: Optional limit to prevent excessive memory usage

    Returns:
        Decoded and sanitized string (may be truncated if max_chars is set)
    """
    # Try a few common encodings; fall back to latin-1 with ignore
    # Try utf-8-sig first to handle BOM properly
    for enc in ("utf-8-sig", "utf-8", "utf-16"):
        try:
            data = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
        except Exception:
            logger.warning("Failed reading %s with %s", path, enc)
            return ""
    else:
        try:
            data = path.read_text(encoding="latin-1", errors="ignore")
        except Exception as e:
            logger.warning("Failed to read text file %s: %s", path, e)
            return ""

    if max_chars is not None and len(data) > max_chars:
        data = data[:max_chars]
    return _strip_control_chars(data)


def _html_to_text(html: str) -> str:
    """Best-effort conversion of HTML to text; falls back to regex strip."""
    if not html:
        return ""
    # Try BeautifulSoup if available for better results
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        # Remove script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return re.sub(r"\s+", " ", text)
    except Exception:
        # Regex fallback: strip tags and collapse whitespace
        text = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text)


def _extract_text_from_doc_win32(path: Path) -> str:
    """Use pywin32/Word to extract text from legacy .doc files on Windows."""
    try:
        import win32com.client  # type: ignore

        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        doc = None
        try:
            doc = word.Documents.Open(str(path.resolve()))
            return _strip_control_chars(doc.Content.Text or "")
        finally:
            if doc:
                doc.Close(False)
            word.Quit()  # type: ignore[attr-defined]
    except ImportError:
        logger.warning("pywin32 not installed; cannot process .doc files on Windows.")
        return ""
    except Exception as e:
        logger.error("Error processing .doc file %s with win32com: %s", path, e)
        return ""


def _extract_eml(path: Path) -> str:
    """Parse .eml messages using the stdlib 'email' package."""
    try:
        from email import policy
        from email.parser import BytesParser
        from email.utils import (
            parsedate_to_datetime,  # noqa: F401  (imported for potential future use)
        )

        msg = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    except Exception as e:
        logger.warning("Failed to parse EML %s: %s", path, e)
        return ""

    parts: list[str] = []
    # Include a minimal header block for context
    for hdr in ("From", "To", "Cc", "Bcc", "Subject", "Date"):
        if msg.get(hdr):
            parts.append(f"{hdr}: {msg.get(hdr)}")

    # Prefer text/plain; fall back to text/html
    bodies: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                try:
                    bodies.append(part.get_content())
                except Exception:
                    pass
        if not bodies:
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    try:
                        bodies.append(_html_to_text(part.get_content()))
                    except Exception:
                        pass
    else:
        ctype = msg.get_content_type()
        try:
            if ctype == "text/html":
                bodies.append(_html_to_text(msg.get_content()))
            else:
                bodies.append(msg.get_content())
        except Exception:
            pass

    parts.append("")  # blank line between headers and body
    parts.append("\n\n".join(bodies))
    return _strip_control_chars("\n".join(parts)).strip()


def _extract_msg(path: Path) -> str:
    """Parse Outlook .msg files if extract_msg is available."""
    try:
        import extract_msg  # type: ignore
    except Exception:
        logger.info("extract_msg not installed; skipping .msg file: %s", path)
        return ""
    try:
        m = extract_msg.Message(str(path))
        # Prefer HTML if available
        body = ""
        try:
            html = getattr(m, "htmlBody", None)
            if html:
                body = _html_to_text(html)
        except Exception:
            body = ""

        if not body:
            body = m.body or ""
        headers = []
        for k in ("from", "to", "cc", "bcc", "subject", "date"):
            val = getattr(m, k, None)
            if val:
                headers.append(f"{k.capitalize()}: {val}")

        text = "\n".join(headers + ["", body])
        return _strip_control_chars(text)
    except Exception as e:
        logger.warning("Failed to parse MSG %s: %s", path, e)
        return ""


def extract_text(path: Path, *, max_chars: int | None = None) -> str:
    """
    Extract text from supported file types with robust error handling.
    Unknown formats return an empty string.

    Args:
        path: Path to the file
        max_chars: Optional hard cap on returned text size

    Returns:
        Extracted (and sanitized) text, possibly truncated.
    """
    suffix = path.suffix.lower()

    # Text-like files (handle HTML/XML below)
    if suffix in TEXT_EXTENSIONS:
        text = read_text_file(path, max_chars=max_chars)
        if suffix in {".html", ".htm", ".xml"}:
            text = _html_to_text(text)
        return text

    # Word documents
    if suffix in DOCX_EXTENSIONS:
        try:
            if suffix == ".docx":
                import docx  # type: ignore

                doc = docx.Document(str(path))
                parts: list[str] = [
                    p.text for p in doc.paragraphs if p.text and p.text.strip()
                ]
                # Include table cells
                for table in getattr(doc, "tables", []):
                    for row in table.rows:
                        for cell in row.cells:
                            if cell.text and cell.text.strip():
                                parts.append(cell.text)
                text = "\n".join(parts)
                if max_chars is not None and len(text) > max_chars:
                    text = text[:max_chars]
                return _strip_control_chars(text)
            else:  # .doc
                if os.name == "nt":
                    win_text = _extract_text_from_doc_win32(path)
                    if win_text:
                        return win_text[:max_chars] if max_chars else win_text
                # Cross‑platform best-effort: try textract if installed; otherwise skip
                try:
                    import textract  # type: ignore

                    raw = textract.process(str(path))  # bytes
                    txt = raw.decode("utf-8", errors="ignore")
                    return _strip_control_chars(txt[:max_chars] if max_chars else txt)
                except Exception:
                    logger.info(
                        "No supported reader for legacy .doc file on this platform: %s",
                        path,
                    )
                    return ""
        except ImportError:
            logger.info(
                "python-docx/textract not installed, skipping Word file: %s", path
            )
            return ""
        except Exception as e:
            logger.warning("Failed to read Word document %s: %s", path, e)
            return ""

    # PowerPoint
    if suffix in PPT_EXTENSIONS:
        try:
            from pptx import Presentation  # type: ignore

            prs = Presentation(str(path))
            parts: list[str] = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    # Many shapes expose .text; ignore drawing objects without text
                    if hasattr(shape, "text"):
                        t = getattr(shape, "text", "") or ""
                        if t.strip():
                            parts.append(t)
            text = "\n".join(parts)
            return _strip_control_chars(text[:max_chars] if max_chars else text)
        except ImportError:
            logger.info("python-pptx not installed, skipping PowerPoint file: %s", path)
            return ""
        except Exception as e:
            logger.warning("Failed to read PowerPoint file %s: %s", path, e)
            return ""

    # RTF
    if suffix in RTF_EXTENSIONS:
        try:
            from striprtf.striprtf import rtf_to_text  # type: ignore

            data = path.read_bytes().decode(
                "latin-1", errors="ignore"
            )  # RTF is ASCII/latin-1 compatible
            text = rtf_to_text(data)
            return _strip_control_chars(text[:max_chars] if max_chars else text)
        except ImportError:
            logger.info("striprtf not installed, skipping RTF file: %s", path)
            return ""
        except Exception as e:
            logger.warning("Failed to read RTF file %s: %s", path, e)
            return ""

    # PDFs
    if suffix in PDF_EXTENSIONS:
        try:
            from pypdf import PdfReader  # type: ignore

            try:
                pdf = PdfReader(str(path))
                # Try empty-password decryption when possible
                if getattr(pdf, "is_encrypted", False):
                    try:
                        pdf.decrypt("")  # type: ignore[attr-defined]
                    except Exception:
                        logger.warning(
                            "Skipping encrypted PDF (unable to decrypt): %s", path
                        )
                        return ""
                parts: list[str] = []
                for i, page in enumerate(getattr(pdf, "pages", [])):
                    try:
                        t = page.extract_text() or ""
                        if t:
                            parts.append(t)
                        # Respect max_chars to prevent huge strings
                        if max_chars and sum(len(p) for p in parts) >= max_chars:
                            break
                    except Exception as e:
                        logger.warning(
                            "Failed to extract text from page %d of %s: %s",
                            i + 1,
                            path,
                            e,
                        )
                text = "\n".join(parts)
                return _strip_control_chars(text[:max_chars] if max_chars else text)
            except Exception as e:
                # Handle corruption or unexpected errors
                logger.warning("Failed to read PDF %s: %s. Skipping.", path, e)
                return ""
        except ImportError:
            logger.info("pypdf not installed, skipping PDF file: %s", path)
            return ""
        except Exception as e:
            logger.warning("Unexpected error while reading PDF %s: %s", path, e)
            return ""

    # Excel
    if suffix in EXCEL_EXTENSIONS:
        try:
            import pandas as pd  # type: ignore

            # Prefer explicit engines; fall back to pandas auto-detection
            engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
            try:
                xl = pd.ExcelFile(str(path), engine=engine)
            except Exception:
                xl = pd.ExcelFile(str(path))  # let pandas pick an engine
            text_parts: list[str] = []
            # Guardrails for very large spreadsheets
            max_cells = int(os.getenv("EXCEL_MAX_CELLS", "200000"))
            for sheet_name in xl.sheet_names:
                try:
                    df = pd.read_excel(xl, sheet_name=sheet_name, dtype=str)
                    # Truncate if too big
                    if df.size > max_cells and df.shape[1] > 0:
                        max_rows = max_cells // max(1, df.shape[1])
                        df = df.head(max_rows)
                    text_parts.append(
                        f"[Sheet: {sheet_name}]\n{df.to_string(index=False)}"
                    )
                    if max_chars and sum(len(p) for p in text_parts) >= max_chars:
                        break
                except Exception as e:
                    logger.warning(
                        "Failed reading sheet '%s' in %s: %s", sheet_name, path, e
                    )
            text = "\n".join(text_parts)
            return _strip_control_chars(text[:max_chars] if max_chars else text)
        except ImportError:
            logger.info(
                "pandas/openpyxl/xlrd not installed, skipping Excel file: %s", path
            )
            return ""
        except Exception as e:
            logger.warning("Failed to read Excel file %s: %s", path, e)
            return ""

    # Raw email files
    if suffix in EMAIL_EXTENSIONS:
        return _extract_eml(path) if suffix == ".eml" else _extract_msg(path)

    logger.debug("Unsupported file format for text extraction: %s", suffix)
    return ""


# ---------------------------------------------------------------------------
# Email cleaning / parsing helpers
# ---------------------------------------------------------------------------
_HEADER_PATTERNS = [
    re.compile(p)
    for p in [
        r"(?mi)^(From|Sent|To|Subject|Cc|Bcc|Date|Reply-To|Message-ID|In-Reply-To|References):.*$",
        r"(?mi)^(Importance|X-Priority|X-Mailer|Content-Type|MIME-Version):.*$",
        r"(?mi)^(Thread-Topic|Thread-Index|Accept-Language|Content-Language):.*$",
    ]
]
_SIGNATURE_PATTERNS = [
    re.compile(p, re.MULTILINE)
    for p in [
        r"(?si)^--\s*\n.*",  # traditional signature delimiter
        r"(?si)^\s*best regards.*?$",
        r"(?si)^\s*kind regards.*?$",
        r"(?si)^\s*sincerely.*?$",
        r"(?si)^\s*thanks.*?$",
        r"(?si)^sent from my.*?$",
        r"(?si)\*{3,}.*?confidential.*?\*{3,}",
        r"(?si)this email.*?intended recipient.*?$",
    ]
]
_FORWARDING_PATTERNS = [
    re.compile(p)
    for p in [
        r"(?m)^-{3,}\s*Original Message\s*-{3,}.*?$",
        r"(?m)^-{3,}\s*Forwarded Message\s*-{3,}.*?$",
        r"(?m)^_{10,}.*?$",
    ]
]


def clean_email_text(text: str) -> str:
    """
    Clean email body text for indexing and retrieval.

    The function is intentionally conservative to avoid removing
    substantive content. It primarily:
    - removes common header lines
    - strips simple signatures / legal footers
    - removes quoted reply markers and repeated delimiters
    - redacts email addresses and URLs
    """
    if not text:
        return ""

    # Handle BOM
    if text.startswith("\ufeff"):
        text = text[1:]

    # Remove obvious header lines (keep body content intact)
    for pattern in _HEADER_PATTERNS:
        text = pattern.sub("", text)

    # Remove simple signatures/footers ONLY from the trailing portion
    tail = text[-2000:]  # examine last ~2k chars
    for pattern in _SIGNATURE_PATTERNS:
        tail = pattern.sub("", tail)
    text = text[:-2000] + tail if len(text) > 2000 else tail

    # Remove forwarding separators
    for pattern in _FORWARDING_PATTERNS:
        text = pattern.sub("", text)

    # Remove '>' quoting markers and normalize noise
    text = re.sub(r"(?m)^\s*>+\s?", "", text)
    text = re.sub(
        r"[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", r"[email@\1]", text
    )
    text = re.sub(r"https?://[^\s]+", "[URL]", text)
    text = re.sub(r"[=\-_*]{10,}", "", text)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"\!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^\s*$", "", text)  # drop blank-only lines
    return _strip_control_chars(text).strip()


def extract_email_metadata(text: str) -> dict[str, object]:
    """
    Extract structured metadata from raw RFC-822 style headers in text.
    Returns a dict with fields: sender, recipients, date, subject, cc, bcc.

    This is a best-effort heuristic and does not parse multi-line headers.
    """
    md: dict[str, object] = {
        "sender": None,
        "recipients": [],
        "date": None,
        "subject": None,
        "cc": [],
        "bcc": [],
    }
    # Simple one-line header extraction
    m = re.search(r"(?mi)^From:\s*(.+?)$", text)
    if m:
        md["sender"] = m.group(1).strip()

    m = re.search(r"(?mi)^To:\s*(.+?)$", text)
    if m:
        recipients = [r.strip() for r in m.group(1).split(",") if r.strip()]
        md["recipients"] = recipients

    m = re.search(r"(?mi)^Cc:\s*(.+?)$", text)
    if m:
        md["cc"] = [c.strip() for c in m.group(1).split(",") if c.strip()]

    m = re.search(r"(?mi)^(?:Date|Sent):\s*(.+?)$", text)
    if m:
        md["date"] = m.group(1).strip()

    m = re.search(r"(?mi)^Subject:\s*(.+?)$", text)
    if m:
        md["subject"] = m.group(1).strip()

    return md


def split_email_thread(text: str) -> list[str]:
    """
    Split an email thread into individual messages.

    Heuristics:
    - Use common "Original/Forwarded Message" separators and "On ... wrote:" lines
    - As a tie-breaker, if multiple message blocks contain a 'Date:' header,
      sort the blocks chronologically; otherwise preserve input order.

    Returns:
        List of message bodies in chronological order (oldest -> newest) when possible.
    """
    if not text or not text.strip():
        return []

    # Common separators
    sep = re.compile(
        r"(?mi)^(?:-{3,}\s*Original Message\s*-{3,}|"
        r"-{3,}\s*Forwarded Message\s*-{3,}|"
        r"On .+? wrote:|"
        r"_{10,})\s*$"
    )

    parts = [p.strip() for p in sep.split(text) if p and p.strip()]
    if len(parts) <= 1:
        return [text.strip()]

    # If multiple parts have recognizable Date headers, sort them
    from email.utils import parsedate_to_datetime

    def _parse_date(s: str):
        m = re.search(r"(?mi)^Date:\s*(.+?)$", s)
        if not m:
            return None
        try:
            return parsedate_to_datetime(m.group(1))
        except Exception:
            return None

    dated: list[tuple] = []
    undated: list[str] = []
    for p in parts:
        d = _parse_date(p)
        if d:
            dated.append((d, p))
        else:
            undated.append(p)
    if len(dated) >= 2:
        dated.sort(key=lambda x: x[0])
        ordered = [
            p for _, p in dated
        ] + undated  # put undated at the end in original order
    else:
        ordered = parts

    return ordered


def find_conversation_dirs(root: Path) -> list[Path]:
    """
    Heuristic: a conversation directory contains a 'Conversation.txt' file.
    """
    return sorted(p.parent for p in root.rglob("Conversation.txt"))


def load_conversation(
    convo_dir: Path,
    include_attachment_text: bool = False,
    max_total_attachment_text: int = 10000,
    *,
    max_attachment_text_chars: int = int(
        os.getenv("MAX_ATTACHMENT_TEXT_CHARS", "500000")
    ),
    skip_if_attachment_over_mb: float | None = float(
        os.getenv("SKIP_ATTACHMENT_OVER_MB", "0")
    ),
) -> dict:
    """
    Load conversation content, manifest/summary JSON, and attachment texts.

    Args:
        convo_dir: Conversation folder
        include_attachment_text: If True, append attachment text (truncated) to conversation_txt
        max_total_attachment_text: Total characters from attachments to append when include_attachment_text is True
        max_attachment_text_chars: Hard cap per-attachment text to store in memory (default from env MAX_ATTACHMENT_TEXT_CHARS)
        skip_if_attachment_over_mb: If > 0, skip extracting text from files larger than this size in MB

    Returns:
        dict with keys: path, conversation_txt, attachments, summary, manifest
    """
    convo_txt_path = convo_dir / "Conversation.txt"
    summary_json = convo_dir / "summary.json"
    manifest_json = convo_dir / "manifest.json"

    # Read conversation text with BOM handling and sanitation
    conversation_text = ""
    if convo_txt_path.exists():
        try:
            conversation_text = read_text_file(convo_txt_path)
        except Exception as e:
            logger.warning("Failed to read Conversation.txt at %s: %s", convo_dir, e)
            conversation_text = ""

    conv = {
        "path": str(convo_dir),
        "conversation_txt": conversation_text,
        "attachments": [],
        "summary": {},
        "manifest": {},
    }

    # Load manifest.json (with UTF-8 BOM handling and control character sanitization)
    if manifest_json.exists():
        sanitized_text = ""
        try:
            manifest_text = manifest_json.read_text(encoding="utf-8-sig")
            sanitized_text = _CONTROL_CHARS.sub("", manifest_text)
            # Fix stray single backslashes that break JSON
            sanitized_text = re.sub(
                r'(?<!\\)\\(?!["\\/bfnrtu])', r"\\\\", sanitized_text
            )
            conv["manifest"] = json.loads(sanitized_text)
        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to load manifest from %s due to JSON error: %s. Attempting HJSON.",
                convo_dir,
                e,
            )
            try:
                import hjson  # type: ignore

                conv["manifest"] = hjson.loads(sanitized_text)
                logger.info(
                    "Successfully parsed manifest for %s using hjson.", convo_dir
                )
            except Exception as hjson_e:
                logger.error(
                    "hjson also failed to parse manifest for %s: %s. Skipping.",
                    convo_dir,
                    hjson_e,
                )
                conv["manifest"] = {}
        except Exception as e:
            logger.warning(
                "Unexpected error while loading manifest from %s: %s. Skipping.",
                convo_dir,
                e,
            )
            conv["manifest"] = {}

    # Build attachment file list (avoid duplicates)
    attachment_files: list[Path] = []
    attachments_dir = convo_dir / "Attachments"
    if attachments_dir.exists() and attachments_dir.is_dir():
        attachment_files.extend([p for p in attachments_dir.rglob("*") if p.is_file()])

    excluded = {"Conversation.txt", "manifest.json", "summary.json"}
    try:
        for child in convo_dir.iterdir():
            if child.is_file() and child.name not in excluded:
                attachment_files.append(child)
    except Exception as e:
        logger.warning("Failed to iterate conversation dir %s: %s", convo_dir, e)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_files: list[Path] = []
    for f in attachment_files:
        try:
            s = str(f.resolve())
        except Exception:
            s = str(f)
        if s not in seen:
            seen.add(s)
            unique_files.append(f)

    total_appended = 0
    for att_file in unique_files:
        try:
            if skip_if_attachment_over_mb and skip_if_attachment_over_mb > 0:
                try:
                    mb = att_file.stat().st_size / (1024 * 1024)
                    if mb > skip_if_attachment_over_mb:
                        logger.info(
                            "Skipping large attachment (%.2f MB > %.2f MB): %s",
                            mb,
                            skip_if_attachment_over_mb,
                            att_file,
                        )
                        continue
                except Exception:
                    pass

            txt = extract_text(att_file, max_chars=max_attachment_text_chars)
            if txt.strip():
                att_rec = {"path": str(att_file), "text": txt}
                conv["attachments"].append(att_rec)

                # Optionally append a truncated view of the attachment into conversation_txt
                if (
                    include_attachment_text
                    and total_appended < max_total_attachment_text
                ):
                    remaining = max_total_attachment_text - total_appended
                    header = f"\n\n--- ATTACHMENT: {att_file.name} ---\n\n"
                    snippet = txt[: max(0, remaining - len(header))]
                    conv["conversation_txt"] += header + snippet
                    total_appended += len(header) + len(snippet)
        except Exception as e:
            logger.warning("Failed to process attachment %s: %s", att_file, e)

    # Load summary.json (BOM-safe)
    if summary_json.exists():
        try:
            conv["summary"] = json.loads(summary_json.read_text(encoding="utf-8-sig"))
        except Exception:
            conv["summary"] = {}

    return conv


def ensure_dir(p: Path) -> None:
    """Create directory and parents if needed (idempotent)."""
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Person class with age property
# ---------------------------------------------------------------------------
class Person:
    def __init__(self, name: str, birthdate: str):
        """
        Initialize a Person object.

        Args:
            name: Full name of the person
            birthdate: Birthdate in ISO format (YYYY-MM-DD)
        """
        self.name = name
        self.birthdate = birthdate

    @property
    def age(self) -> int:
        """Calculate age based on the birthdate."""
        if not self.birthdate:
            return 0
        try:
            birth_date = datetime.datetime.strptime(self.birthdate, "%Y-%m-%d")
            today = datetime.datetime.today()
            return today.year - birth_date.year - (
                (today.month, today.day) < (birth_date.month, birth_date.day)
            )
        except Exception:
            return 0

    def getAge(self) -> int:
        """Alias for the age property."""
        return self.age
