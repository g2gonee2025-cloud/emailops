from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

# Recipient types (Outlook constants)
OL_TO = 1
OL_CC = 2
OL_BCC = 3

# Mail item class
OL_MAIL = 43

# DownloadState (per Outlook OlDownloadState)
OL_HEADERONLY = 0  # olHeaderOnly
OL_FULLITEM = 1    # olFullItem

# BodyFormat (Outlook OlBodyFormat)
OL_FORMAT_HTML = 2

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str) -> str:
    """Windows-safe filename: remove forbidden chars, collapse whitespace,
    avoid reserved device names, and trim trailing dots/spaces."""
    if not name:
        return "_"
    # Remove illegal characters (including control chars)
    s = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # Collapse consecutive underscores
    s = re.sub(r"_+", "_", s).rstrip(" .")
    if not s:
        s = "unnamed"
    # Reserved DOS device names (case-insensitive)
    reserved = {
        "CON","PRN","AUX","NUL",
        "COM1","COM2","COM3","COM4","COM5","COM6","COM7","COM8","COM9",
        "LPT1","LPT2","LPT3","LPT4","LPT5","LPT6","LPT7","LPT8","LPT9",
    }
    base = s.split(".")[0].upper()
    if base in reserved or re.match(r"^(COM[1-9]|LPT[1-9])($|\.)", s, re.IGNORECASE):
        s = "_" + s
    # Conservative length limit for safety (we also guard full path elsewhere)
    return s[:240] if len(s) > 240 else s

def local_tzinfo():
    try:
        return datetime.now().astimezone().tzinfo
    except Exception:
        return None

def to_iso_z(dt: datetime | None) -> str:
    """Return UTC ISO8601 with Z. Safely handles naive datetimes by assuming local then converting to UTC."""
    if dt is None:
        return ""
    tz = local_tzinfo()
    try:
        aware = dt if getattr(dt, "tzinfo", None) else (dt.replace(tzinfo=tz) if tz else dt.astimezone())
        return aware.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ""

def to_iso_local(dt: datetime | None) -> str:
    """Return a friendly local timestamp string (MM/DD/YYYY HH:MM) for display."""
    if dt is None:
        return ""
    try:
        tz = local_tzinfo()
        if tz is None:
            # Fallback to naive datetime formatting if timezone detection fails
            return dt.strftime("%m/%d/%Y %H:%M")
        local_dt = dt.astimezone(tz) if dt.tzinfo else dt.replace(tzinfo=tz)
        return local_dt.strftime("%m/%d/%Y %H:%M")
    except Exception:
        return dt.strftime("%m/%d/%Y %H:%M")

def outlook_restrict_datetime(dt: datetime | None) -> str:
    """Format for Outlook Items.Restrict on ReceivedTime.
    Use 12-hour clock with AM/PM to be robust across locales: MM/DD/YYYY HH:MM AM/PM (local).
    """
    if dt is None:
        return ""
    try:
        tz = local_tzinfo()
        if tz is None:
            # Fallback to naive datetime formatting if timezone detection fails
            return dt.strftime("%m/%d/%Y %I:%M %p")
        local_dt = dt.astimezone(tz) if dt.tzinfo else dt.replace(tzinfo=tz)
        return local_dt.strftime("%m/%d/%Y %I:%M %p")
    except Exception:
        return dt.strftime("%m/%d/%Y %I:%M %p")

def short_hex(s: str, length: int = 6) -> str:
    import zlib
    data: bytes
    if not isinstance(s, (bytes, bytearray)):
        data = str(s).encode("utf-8", errors="ignore")
    else:
        data = bytes(s)
    val = zlib.adler32(data) & 0xFFFFFFFF
    return f"{val:08X}"[:length]

def fit_leaf_for_max_path(base: Path, leaf: str, max_total_len: int = 240) -> str:
    """Trim the leaf name so that base/leaf stays under max_total_len characters."""
    allowed = max_total_len - len(str(base)) - 1  # +1 for os.sep
    if allowed < 16:
        allowed = 16
    return leaf[:allowed]

# ---------------- HTML → text and body extraction ----------------

def html_to_text(html: str) -> str:
    """Convert common HTML to readable plain text."""
    import html as _html
    import re as _re
    if not html:
        return ""
    text = html
    # Drop scripts/styles
    text = _re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", "", text)
    # Convert common line breaks
    text = _re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = _re.sub(r"(?is)</p\s*>", "\n\n", text)
    # Strip tags
    text = _re.sub(r"(?s)<[^>]+>", "", text)
    # Unescape
    text = _html.unescape(text)
    # Normalize whitespace
    text = _re.sub(r"[ \t]+", " ", text)
    text = _re.sub(r"\n[ \t]+", "\n", text)
    text = _re.sub(r"[ \t]+\n", "\n", text)
    text = _re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def get_body_text(item) -> str:
    """Return message body as plain text using Body/HTMLBody appropriately."""
    try:
        if getattr(item, "DownloadState", None) != OL_FULLITEM:
            return ""
        fmt = int(getattr(item, "BodyFormat", 0) or 0)
        if fmt == OL_FORMAT_HTML:
            html = getattr(item, "HTMLBody", "") or ""
            if html:
                return html_to_text(html)
        # Otherwise prefer Body
        body = getattr(item, "Body", "") or ""
        if body and re.search(r"(?i)<\s*html|<\s*body", body):
            return html_to_text(body)
        return body.strip()
    except Exception:
        return ""
