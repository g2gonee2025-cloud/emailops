"""
Text Extraction.

Wraps Unstructured, Tesseract OCR, pdfplumber, etc.
Implements §6.5 of the Canonical Blueprint.
"""
from __future__ import annotations

import contextlib
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, cast

from cortex.utils import strip_control_chars

logger = logging.getLogger(__name__)

# File type support
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
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
MBOX_EXTENSIONS = {".mbox"}

# Global cache for expensive text extraction operations
# (cached_at, mtime, size, text)
_extraction_cache: dict[tuple[Path, int | None], tuple[float, float, int, str]] = {}
_extraction_cache_lock = threading.Lock()
_CACHE_TTL = int(os.getenv("EXTRACTION_CACHE_TTL", "3600"))  # 1 hour TTL


def _is_cache_valid(
    cache_entry: tuple[float, float, int, str],
    *,
    path: Path,
) -> bool:
    """
    Check if a cached extraction result is still valid.
    Validity requires:
      - TTL not expired, and
      - file mtime/size unchanged since cache was written.
    """
    try:
        cached_at, cached_mtime, cached_size, _txt = cache_entry
    except Exception:
        return False

    # TTL check
    if (time.time() - float(cached_at)) > _CACHE_TTL:
        return False

    try:
        st = path.stat()
        cur_mtime = float(st.st_mtime)
        cur_size = int(st.st_size)
    except Exception:
        return False

    # Use small epsilon for float comparison safety
    return (abs(cur_mtime - float(cached_mtime)) < 1e-6) and (
        cur_size == int(cached_size)
    )


def _finalize_text(text: str, max_chars: int | None) -> str:
    """Normalize, truncate, and sanitize extracted text."""
    text = text or ""
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars]
    return strip_control_chars(text)


def _extract_with_tika(path: Path, max_chars: int | None) -> str:
    """Extract text using Apache Tika if available."""
    try:
        from tika import parser  # type: ignore

        tika_server_url = os.getenv("TIKA_SERVER_URL")
        if tika_server_url:
            os.environ.setdefault("TIKA_SERVER_ENDPOINT", tika_server_url)
    except Exception:
        logger.info("Apache Tika not available; skipping %s", path)
        return ""

    with contextlib.suppress(Exception):
        parsed_raw = parser.from_file(str(path))  # type: ignore[attr-defined]
        if isinstance(parsed_raw, dict):
            parsed_dict = cast(dict[str, Any], parsed_raw)
            content = cast(str | None, parsed_dict.get("content"))
            if content:
                return _finalize_text(content, max_chars)
    logger.debug("Apache Tika produced no content for %s", path)
    return ""


def _extract_with_unstructured(path: Path, suffix: str, max_chars: int | None) -> str:
    """Extract text using Unstructured's partitioners when installed."""
    try:
        from unstructured.partition.auto import partition  # type: ignore
    except Exception:
        logger.info("unstructured not installed; skipping %s", path)
        return ""

    kwargs: dict[str, object] = {
        "filename": str(path),
        "include_page_breaks": True,
        "metadata_filename": True,
        "process_attachments": suffix in EMAIL_EXTENSIONS,
    }

    try:
        elements = cast(Iterable[Any], partition(**kwargs))
    except Exception as exc:
        logger.warning("unstructured partition failed for %s: %s", path, exc)
        return ""

    parts: list[str] = []
    for element in elements:
        text = cast(str | None, getattr(element, "text", None))
        if text and text.strip():
            parts.append(text.strip())

    if not parts:
        return ""

    return _finalize_text("\n\n".join(parts), max_chars)


def _extract_image_ocr(path: Path, max_chars: int | None) -> str:
    """Attempt OCR extraction for image-based attachments."""
    try:
        from PIL import Image  # type: ignore
    except Exception:
        logger.info("Pillow not installed; skipping image OCR for %s", path)
        return ""

    try:
        import pytesseract  # type: ignore
    except Exception:
        logger.info("pytesseract not installed; skipping OCR for %s", path)
        return ""

    try:
        with Image.open(path) as image:
            pytesseract_module = cast(Any, pytesseract)
            text = cast(str, pytesseract_module.image_to_string(image))
    except Exception as exc:
        logger.warning("OCR failed for %s: %s", path, exc)
        return ""

    return _finalize_text(text, max_chars)


def _extract_pdf_with_ocr(path: Path, max_chars: int | None) -> str:
    """Fallback OCR for PDFs that contain no embedded text."""
    try:
        from pdf2image import convert_from_path  # type: ignore
    except Exception:
        logger.info("pdf2image not installed; skipping PDF OCR for %s", path)
        return ""

    try:
        convert = cast(Callable[..., Sequence[Any]], convert_from_path)
        images = convert(str(path))
    except Exception as exc:
        logger.warning("Unable to rasterize PDF %s for OCR: %s", path, exc)
        return ""

    aggregated: list[str] = []
    for image in images:
        with contextlib.suppress(Exception):
            # Reuse image OCR helper with in-memory image object
            try:
                import pytesseract  # type: ignore
            except Exception:
                return ""
            pytesseract_module = cast(Any, pytesseract)
            text = cast(str, pytesseract_module.image_to_string(image))
            if text and text.strip():
                aggregated.append(text.strip())

    if not aggregated:
        return ""

    return _finalize_text("\n".join(aggregated), max_chars)


def _extract_pdf_tables(path: Path, max_chars: int | None) -> str:
    """Extract tables from PDFs using Camelot or pdfplumber when possible."""
    tables_output: list[str] = []

    try:
        import camelot  # type: ignore

        camelot_module = cast(Any, camelot)
        camelot_tables = cast(
            Sequence[Any], camelot_module.read_pdf(str(path), pages="all")
        )
        for index, table in enumerate(camelot_tables):
            df = cast(Any, getattr(table, "df", None))
            csv_data = df.to_csv(index=False, header=True) if df is not None else ""
            tables_output.append(f"[Camelot Table {index + 1}]\n{csv_data}")
    except ImportError:
        logger.info("Camelot not installed; falling back to pdfplumber for %s", path)
    except Exception as exc:
        logger.warning("Camelot table extraction failed for %s: %s", path, exc)

    if not tables_output:
        try:
            import pdfplumber  # type: ignore
        except Exception:
            logger.info(
                "pdfplumber not installed; skipping table extraction for %s", path
            )
        else:
            with contextlib.suppress(Exception):
                pdfplumber_module = cast(Any, pdfplumber)
                with pdfplumber_module.open(str(path)) as pdf:
                    pages = cast(Sequence[Any], getattr(pdf, "pages", []))
                    for page_num, page in enumerate(pages, start=1):
                        table = cast(
                            Sequence[Sequence[str | None]] | None, page.extract_table()
                        )
                        if not table:
                            continue
                        csv_lines: list[str] = []
                        for row in table:
                            if not row or not any(row):
                                continue
                            normalized_row = [cell or "" for cell in row]
                            csv_lines.append(",".join(normalized_row))
                        if csv_lines:
                            tables_output.append(
                                f"[pdfplumber Table {page_num}]\n"
                                + "\n".join(csv_lines)
                            )

    if not tables_output:
        return ""

    return _finalize_text("\n\n".join(tables_output), max_chars)


def _html_to_text(html: str) -> str:
    """Best-effort conversion of HTML to text; falls back to regex strip."""
    if not html:
        return ""
    # Try BeautifulSoup if available for better results
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = cast(Any, BeautifulSoup(html, "html.parser"))
        # Remove script/style
        for tag in cast(Iterable[Any], soup(["script", "style", "noscript"])):
            tag.decompose()
        text = cast(str, soup.get_text(separator=" ", strip=True))
        return re.sub(r"\s+", " ", text)
    except Exception:
        # Regex fallback: strip tags and collapse whitespace
        text = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text)


def _extract_text_from_doc_win32(path: Path) -> str:
    """Use pywin32/Word to extract text from legacy .doc files on Windows."""
    try:
        import win32com.client

        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        doc = None
        try:
            doc = word.Documents.Open(str(path.resolve()))
            return strip_control_chars(doc.Content.Text or "")
        finally:
            if doc:
                doc.Close(False)
            word.Quit()
    except ImportError:
        logger.info("pywin32 not installed; cannot process .doc files on Windows.")
        return ""
    except Exception as e:
        logger.error("Error processing .doc file %s with win32com: %s", path, e)
        return ""


def _extract_eml(path: Path) -> str:
    """Parse .eml messages using the stdlib 'email' package."""
    try:
        from email import policy
        from email.parser import BytesParser

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
                with contextlib.suppress(Exception):
                    bodies.append(part.get_content())
        if not bodies:
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    with contextlib.suppress(Exception):
                        bodies.append(_html_to_text(part.get_content()))
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
    return strip_control_chars("\n".join(parts)).strip()


def _extract_msg(path: Path) -> str:
    """Parse Outlook .msg files if extract_msg is available."""
    try:
        import extract_msg  # type: ignore
    except Exception:
        logger.info("extract_msg not installed; skipping .msg file: %s", path)
        return ""

    extract_msg_module = cast(Any, extract_msg)
    m: Any | None = None
    text_result = ""

    try:
        m = extract_msg_module.Message(str(path))
        # Prefer HTML if available
        body = ""
        try:
            html = cast(str | None, getattr(m, "htmlBody", None))
            if html:
                body = _html_to_text(html)
        except Exception:
            body = ""

        if not body:
            body = cast(str | None, getattr(m, "body", "")) or ""
        headers: list[str] = []
        for k in ("from", "to", "cc", "bcc", "subject", "date"):
            val = cast(str | None, getattr(m, k, None))
            if val:
                headers.append(f"{k.capitalize()}: {val}")

        text = "\n".join([*headers, "", body])
        text_result = strip_control_chars(text)
    except Exception as e:
        logger.warning("Failed to parse MSG %s: %s", path, e)
        text_result = ""
    finally:
        if m is not None:
            try:
                close_fn: Callable[..., object] | None = None
                exit_fn: Callable[..., object] | None = None

                try:
                    close_candidate = cast(Any, m).close  # type: ignore[attr-defined]
                    if callable(close_candidate):
                        close_fn = close_candidate
                except AttributeError:
                    close_fn = None

                if close_fn is None:
                    try:
                        exit_candidate = cast(Any, m).__exit__  # type: ignore[attr-defined]
                        if callable(exit_candidate):
                            exit_fn = exit_candidate
                    except AttributeError:
                        exit_fn = None

                if close_fn is not None:
                    close_fn()
                elif exit_fn is not None:
                    exit_fn(None, None, None)
            except Exception as close_error:
                logger.debug("Failed to close MSG handle for %s: %s", path, close_error)

    return text_result


def _extract_word_document(path: Path, suffix: str, max_chars: int | None) -> str:
    """Extract text from Word documents (.docx, .doc)."""
    try:
        if suffix == ".docx":
            import docx  # type: ignore

            docx_module = cast(Any, docx)
            document = docx_module.Document(str(path))
            parts: list[str] = []
            for paragraph in cast(Iterable[Any], getattr(document, "paragraphs", [])):
                paragraph_text = cast(str | None, getattr(paragraph, "text", None))
                if paragraph_text and paragraph_text.strip():
                    parts.append(paragraph_text)
            for table in cast(Iterable[Any], getattr(document, "tables", [])):
                for row in cast(Iterable[Any], getattr(table, "rows", [])):
                    for cell in cast(Iterable[Any], getattr(row, "cells", [])):
                        cell_text = cast(str | None, getattr(cell, "text", None))
                        if cell_text and cell_text.strip():
                            parts.append(cell_text)
            text = "\n".join(parts)
            if max_chars is not None and len(text) > max_chars:
                text = text[:max_chars]
            return strip_control_chars(text)
        else:  # .doc
            if os.name == "nt":
                win_text = _extract_text_from_doc_win32(path)
                if win_text:
                    return win_text[:max_chars] if max_chars else win_text
            # Cross-platform best-effort: try textract if installed
            try:
                import textract  # type: ignore

                textract_module = cast(Any, textract)
                raw = cast(bytes, textract_module.process(str(path)))
                txt = raw.decode("utf-8", errors="ignore")
                return strip_control_chars(txt[:max_chars] if max_chars else txt)
            except Exception:
                logger.info(
                    "No supported reader for legacy .doc file on this platform: %s",
                    path,
                )
                return ""
    except ImportError:
        logger.info("python-docx/textract not installed, skipping Word file: %s", path)
        return ""
    except Exception as e:
        logger.warning("Failed to read Word document %s: %s", path, e)
        return ""


def _extract_powerpoint(path: Path, max_chars: int | None) -> str:
    """Extract text from PowerPoint files."""
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
        return strip_control_chars(text[:max_chars] if max_chars else text)
    except ImportError:
        logger.info("python-pptx not installed, skipping PowerPoint file: %s", path)
        return ""
    except Exception as e:
        logger.warning("Failed to read PowerPoint file %s: %s", path, e)
        return ""


def _extract_rtf(path: Path, max_chars: int | None) -> str:
    """Extract text from RTF files."""
    try:
        from striprtf.striprtf import rtf_to_text  # type: ignore

        data = path.read_bytes().decode("latin-1", errors="ignore")
        text = rtf_to_text(data)
        return strip_control_chars(text[:max_chars] if max_chars else text)
    except ImportError:
        logger.info("striprtf not installed, skipping RTF file: %s", path)
        return ""
    except Exception as e:
        logger.warning("Failed to read RTF file %s: %s", path, e)
        return ""


def _extract_pdf(path: Path, max_chars: int | None) -> str:
    """Extract text from PDF files."""
    try:
        from pypdf import PdfReader  # type: ignore

        try:
            with Path.open(path, "rb") as fh:
                pdf = PdfReader(fh)
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
                acc = 0
                budget = max_chars if max_chars is not None else None
                for i, page in enumerate(getattr(pdf, "pages", [])):
                    try:
                        t = page.extract_text() or ""
                        if not t:
                            continue
                        remain = None if budget is None else (budget - acc)
                        if remain is not None and remain <= 0:
                            break
                        if remain is not None and len(t) > remain:
                            t = t[:remain]
                        parts.append(t)
                        acc += len(t)
                    except Exception as e:
                        logger.warning(
                            "Failed to extract text from page %d of %s: %s",
                            i + 1,
                            path,
                            e,
                        )
            text = "\n".join(parts)
            return strip_control_chars(text)
        except Exception as e:
            logger.warning("Failed to read PDF %s: %s. Skipping.", path, e)
            return ""
    except ImportError:
        logger.info("pypdf not installed, skipping PDF file: %s", path)
        return ""
    except Exception as e:
        logger.warning("Unexpected error while reading PDF %s: %s", path, e)
        return ""


def _extract_excel(path: Path, suffix: str, max_chars: int | None) -> str:
    """Extract text from Excel files."""
    try:
        from typing import Literal

        import pandas as pd  # type: ignore

        engine: Literal["openpyxl", "xlrd"] = (
            "openpyxl" if suffix == ".xlsx" else "xlrd"
        )

        # Try explicit engine first, then fall back to pandas auto-detection
        def _iter_sheets(_engine: Literal["openpyxl", "xlrd"] | None):
            if _engine:
                return pd.ExcelFile(str(path), engine=_engine)
            return pd.ExcelFile(str(path))

        xl = None
        try:
            xl = _iter_sheets(engine)
        except Exception:
            xl = _iter_sheets(None)

        parts: list[str] = []
        acc = 0
        budget = max_chars if max_chars is not None else None
        max_cells = 200000  # Default max cells limit

        # Ensure the handle is closed even if exceptions occur
        with xl:
            for sheet_name in xl.sheet_names:
                try:
                    df = pd.read_excel(xl, sheet_name=sheet_name, dtype=str)
                    if df.size > max_cells and len(df.shape) > 1:
                        max_rows = max_cells // df.shape[1]
                        df = df.head(max_rows)
                    chunk = f"[Sheet: {sheet_name}]\n{df.to_csv(index=False)}"
                    remain = None if budget is None else (budget - acc)
                    if remain is not None and remain <= 0:
                        break
                    if remain is not None and len(chunk) > remain:
                        chunk = chunk[:remain]
                    parts.append(chunk)
                    acc += len(chunk)
                except Exception as e:
                    logger.warning(
                        "Failed reading sheet '%s' in %s: %s", sheet_name, path, e
                    )
        text = "\n".join(parts)
        return strip_control_chars(text)
    except ImportError:
        logger.info("pandas/openpyxl/xlrd not installed, skipping Excel file: %s", path)
        return ""
    except Exception as e:
        logger.warning("Failed to read Excel file %s: %s", path, e)
        return ""


def extract_text(path: Path, *, max_chars: int | None = None) -> str:
    """
    Extract text from supported file types with robust error handling.

    Supports: .txt, .pdf, .docx, .doc, .xlsx, .xls, .pptx, .ppt,
    .rtf, .eml, .msg, .html, .xml, .md, .json, .yaml, .csv
    Unknown/binary formats return empty string.

    Args:
        path: Path to the file (must exist and be readable)
        max_chars: Optional hard cap on returned text size

    Returns:
        Extracted and sanitized text, possibly truncated.
        Empty string on errors or unsupported formats.
    """
    global _extraction_cache

    # Check cache first
    try:
        cache_key = (path.resolve(), max_chars)
        with _extraction_cache_lock:
            if cache_key in _extraction_cache:
                entry = _extraction_cache[cache_key]
                if _is_cache_valid(entry, path=path):
                    logger.debug("Using cached text extraction for %s", path.name)
                    return entry[3]
    except Exception as e:
        logger.debug("Cache lookup failed for %s: %s", path, e)
        # Proceed with extraction on cache error

    # Basic path validation
    try:
        if not path.exists():
            logger.debug("Path does not exist: %s", path)
            return ""
        if not path.is_file():
            logger.debug("Path is not a file: %s", path)
            return ""
        path = path.resolve()
    except (ValueError, OSError) as e:
        logger.warning("Invalid path: %s - %s", path, e)
        return ""

    suffix = path.suffix.lower()
    segments: list[str] = []
    seen_hashes: set[int] = set()

    def add_segment(segment: str) -> None:
        if not segment:
            return
        normalized = strip_control_chars(segment).strip()
        if not normalized:
            return
        signature = hash(normalized)
        if signature in seen_hashes:
            return
        seen_hashes.add(signature)
        segments.append(normalized)

    if suffix in TEXT_EXTENSIONS:
        from cortex.utils import read_text_file

        text = read_text_file(path, max_chars=max_chars)
        if suffix in {".html", ".htm", ".xml"}:
            text = _html_to_text(text)
        add_segment(text)
    else:
        pipeline: list[tuple[Callable[[], str], bool]] = []

        if suffix in EMAIL_EXTENSIONS:
            if suffix == ".eml":
                pipeline.append((lambda: _extract_eml(path), False))
            else:
                pipeline.append((lambda: _extract_msg(path), False))
            pipeline.append(
                (lambda: _extract_with_unstructured(path, suffix, max_chars), True)
            )
            pipeline.append((lambda: _extract_with_tika(path, max_chars), True))
        elif suffix in DOCX_EXTENSIONS:
            pipeline.append(
                (lambda: _extract_with_unstructured(path, suffix, max_chars), False)
            )
            pipeline.append(
                (lambda: _extract_word_document(path, suffix, max_chars), True)
            )
            pipeline.append((lambda: _extract_with_tika(path, max_chars), False))
        elif suffix in PDF_EXTENSIONS:
            pipeline.append(
                (lambda: _extract_with_unstructured(path, suffix, max_chars), False)
            )
            pipeline.append((lambda: _extract_with_tika(path, max_chars), True))
            pipeline.append((lambda: _extract_pdf(path, max_chars), True))
        elif suffix in EXCEL_EXTENSIONS:
            pipeline.append(
                (lambda: _extract_with_unstructured(path, suffix, max_chars), False)
            )
            pipeline.append((lambda: _extract_excel(path, suffix, max_chars), True))
            pipeline.append((lambda: _extract_with_tika(path, max_chars), False))
        elif suffix in PPT_EXTENSIONS:
            pipeline.append(
                (lambda: _extract_with_unstructured(path, suffix, max_chars), False)
            )
            pipeline.append((lambda: _extract_powerpoint(path, max_chars), True))
            pipeline.append((lambda: _extract_with_tika(path, max_chars), False))
        elif suffix in RTF_EXTENSIONS:
            pipeline.append(
                (lambda: _extract_with_unstructured(path, suffix, max_chars), False)
            )
            pipeline.append((lambda: _extract_rtf(path, max_chars), True))
            pipeline.append((lambda: _extract_with_tika(path, max_chars), False))
        elif suffix in IMAGE_EXTENSIONS:
            pipeline.append(
                (lambda: _extract_with_unstructured(path, suffix, max_chars), False)
            )
            pipeline.append((lambda: _extract_with_tika(path, max_chars), False))
            pipeline.append((lambda: _extract_image_ocr(path, max_chars), True))
        elif suffix in MBOX_EXTENSIONS:
            pipeline.append((lambda: _extract_with_tika(path, max_chars), False))
            pipeline.append(
                (lambda: _extract_with_unstructured(path, suffix, max_chars), True)
            )
        else:
            pipeline.append(
                (lambda: _extract_with_unstructured(path, suffix, max_chars), False)
            )
            pipeline.append((lambda: _extract_with_tika(path, max_chars), False))

        for extractor, always_run in pipeline:
            if segments and not always_run:
                continue
            segment = extractor()
            if segment:
                add_segment(segment)

        if suffix in PDF_EXTENSIONS:
            if not segments:
                ocr_text = _extract_pdf_with_ocr(path, max_chars)
                if ocr_text:
                    add_segment(ocr_text)
            table_text = _extract_pdf_tables(path, max_chars)
            if table_text:
                add_segment(table_text)
        elif suffix in IMAGE_EXTENSIONS and not segments:
            ocr_text = _extract_image_ocr(path, max_chars)
            if ocr_text:
                add_segment(ocr_text)

    combined = "\n\n".join(segments)
    result = _finalize_text(combined, max_chars) if combined else ""

    # Update cache
    if result:
        try:
            cache_key = (path.resolve(), max_chars)
            with _extraction_cache_lock:
                try:
                    st = path.stat()
                    _extraction_cache[cache_key] = (
                        time.time(),
                        float(st.st_mtime),
                        int(st.st_size),
                        result,
                    )
                except Exception:
                    _extraction_cache[cache_key] = (time.time(), 0.0, 0, result)

                # Clean old cache entries periodically
                if len(_extraction_cache) > 100:
                    _extraction_cache = {
                        k: v
                        for k, v in _extraction_cache.items()
                        if (time.time() - float(v[0])) <= _CACHE_TTL
                    }
        except Exception as e:
            logger.debug("Failed to update cache for %s: %s", path, e)

    return result
