"""
Email parsing & threading.

Implements §6.2 of the Canonical Blueprint.

Features:
* Parse .eml / .mbox files (RFC-aware)
* Extract Message-ID, In-Reply-To, References, Subject, Date, participants
* Threading rules:
  1. First use References header chain
  2. Then In-Reply-To
  3. Else cluster by (normalized_subject, participants, time_window)
* Ambiguity handling: log thread_ambiguity in job problems
"""
from __future__ import annotations

import hashlib
import logging
import mailbox
import re
import uuid
from datetime import datetime, timedelta, timezone
from email import policy
from email.header import decode_header, make_header
from email.message import EmailMessage, Message
from email.parser import BytesParser
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Subject normalization patterns
_RE_FWD_PATTERN = re.compile(
    r"^(?:re|fwd?|fw|aw|sv|vs|antw|odp|回复|答复|轉寄):\s*", re.IGNORECASE
)
_BRACKET_PATTERN = re.compile(r"\[.*?\]")
_WHITESPACE_PATTERN = re.compile(r"\s+")

# Time window for clustering (48 hours)
THREAD_TIME_WINDOW = timedelta(hours=48)


# =============================================================================
# Data Models
# =============================================================================


class ParsedEmail(BaseModel):
    """Parsed email structure."""

    message_id: str
    in_reply_to: str | None = None
    references: list[str] = Field(default_factory=list)
    subject: str = ""
    subject_normalized: str = ""
    from_addr: str = ""
    from_name: str = ""
    to_addrs: list[str] = Field(default_factory=list)
    cc_addrs: list[str] = Field(default_factory=list)
    bcc_addrs: list[str] = Field(default_factory=list)
    date: str | None = None
    date_parsed: datetime | None = None
    body_plain: str = ""
    body_html: str | None = None
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)


class ThreadInfo(BaseModel):
    """Threading information for an email."""

    thread_id: str
    position_in_thread: int = 0
    is_thread_start: bool = False
    threading_method: str = "unknown"  # references, in_reply_to, clustering, or subject
    ambiguity_score: float = 0.0


class EmailThread(BaseModel):
    """A thread of related emails."""

    thread_id: str
    subject: str
    subject_normalized: str
    participants: set[str] = Field(default_factory=set)
    message_ids: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    ended_at: datetime | None = None


# =============================================================================
# Helper Functions
# =============================================================================


def _decode_header_value(value: str | None) -> str:
    """Decode RFC 2047 encoded header value."""
    if not value:
        return ""
    try:
        decoded_parts = decode_header(value)
        return str(make_header(decoded_parts))
    except Exception:
        return value.strip() if value else ""


def _normalize_subject(subject: str) -> str:
    """
    Normalize subject for threading.

    - Remove Re:, Fwd:, FW:, etc. prefixes (in multiple languages)
    - Remove [bracketed] content like [EXTERNAL]
    - Collapse whitespace
    - Lowercase
    """
    if not subject:
        return ""

    # Remove Re/Fwd prefixes iteratively
    normalized = subject
    while True:
        new_val = _RE_FWD_PATTERN.sub("", normalized).strip()
        if new_val == normalized:
            break
        normalized = new_val

    # Remove bracketed content
    normalized = _BRACKET_PATTERN.sub("", normalized)

    # Collapse whitespace and lowercase
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized).strip().lower()

    return normalized


def normalize_subject(subject: str) -> str:
    """Public helper that normalizes subjects consistently across modules."""
    return _normalize_subject(subject)


def _extract_email_address(addr_str: str) -> str:
    """Extract clean email address from 'Name <email>' format."""
    if not addr_str:
        return ""
    _, email_addr = parseaddr(addr_str)
    return email_addr.lower().strip()


def _parse_address_list(addr_str: str) -> list[str]:
    """Parse comma-separated address list."""
    if not addr_str:
        return []

    addresses = []
    # Handle quoted commas in names
    parts = re.split(r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", addr_str)
    for part in parts:
        addr = _extract_email_address(part.strip())
        if addr:
            addresses.append(addr)
    return addresses


def _generate_message_id() -> str:
    """Generate a unique message ID for emails without one."""
    return f"<generated-{uuid.uuid4()}@cortex.local>"


def _calculate_participants_hash(participants: set[str]) -> str:
    """Calculate hash of participant set for clustering."""
    sorted_parts = sorted(participants)
    return hashlib.md5(",".join(sorted_parts).encode()).hexdigest()[:16]


# =============================================================================
# Email Parsing
# =============================================================================


def parse_eml(content: bytes) -> ParsedEmail:
    """
    Parse raw EML content.

    Extracts:
    * Message-ID, In-Reply-To, References
    * Subject, Date, Participants
    * Body (plain/html)
    * Attachments

    Args:
        content: Raw email bytes

    Returns:
        ParsedEmail with all extracted fields
    """
    try:
        parser = BytesParser(policy=cast(Any, policy.default))
        msg = cast(EmailMessage, parser.parsebytes(content))
    except Exception as e:
        logger.error(f"Failed to parse email: {e}")
        # Return empty ParsedEmail with generated ID
        return ParsedEmail(
            message_id=_generate_message_id(),
            body_plain="[Failed to parse email content]",
        )

    # 1. Extract headers
    message_id = msg.get("Message-ID", "").strip()
    if not message_id:
        message_id = _generate_message_id()

    in_reply_to = msg.get("In-Reply-To", "").strip() or None

    # Parse References header (space or newline separated)
    references_raw = msg.get("References", "")
    references = [r.strip() for r in re.split(r"[\s\n]+", references_raw) if r.strip()]

    # Subject handling
    subject_raw = msg.get("Subject", "")
    subject = _decode_header_value(subject_raw)
    subject_normalized = _normalize_subject(subject)

    # From address
    from_raw = msg.get("From", "")
    from_decoded = _decode_header_value(from_raw)
    from_name, from_email = parseaddr(from_decoded)
    from_addr = from_email.lower().strip()

    # 2. Parse address lists
    to_addrs = _parse_address_list(_decode_header_value(msg.get("To", "")))
    cc_addrs = _parse_address_list(_decode_header_value(msg.get("Cc", "")))
    bcc_addrs = _parse_address_list(_decode_header_value(msg.get("Bcc", "")))

    # Date parsing
    date_str = msg.get("Date", "").strip() or None
    date_parsed = None
    if date_str:
        try:
            date_parsed = parsedate_to_datetime(date_str)
        except Exception:
            pass

    # 3. Extract body parts and attachments
    body_plain = ""
    body_html = None
    attachments = []

    # Store important headers
    headers = {}
    for hdr in [
        "Message-ID",
        "In-Reply-To",
        "References",
        "Subject",
        "From",
        "To",
        "Cc",
        "Date",
    ]:
        val = msg.get(hdr)
        if val:
            headers[hdr] = str(val)

    def _get_charset(part: Message, default: str = "utf-8") -> str:
        charset = part.get_content_charset()
        return charset or default

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition") or "")

            if "attachment" in content_disposition:
                filename = part.get_filename()
                if filename:
                    try:
                        payload = part.get_payload(decode=True)
                        attachments.append(
                            {
                                "filename": _decode_header_value(filename),
                                "content_type": content_type,
                                "size": len(payload) if payload else 0,
                                "payload": payload,
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to extract attachment {filename}: {e}")
                continue

            if content_type == "text/plain" and "attachment" not in content_disposition:
                try:
                    payload_bytes = part.get_payload(decode=True)
                    if isinstance(payload_bytes, bytes):
                        body_plain += payload_bytes.decode(
                            _get_charset(part), errors="replace"
                        )
                    elif isinstance(payload_bytes, str):
                        body_plain += payload_bytes
                except Exception as e:
                    logger.warning(f"Failed to decode text/plain part: {e}")

            elif (
                content_type == "text/html" and "attachment" not in content_disposition
            ):
                try:
                    payload_bytes = part.get_payload(decode=True)
                    if isinstance(payload_bytes, bytes):
                        html_content = payload_bytes.decode(
                            _get_charset(part), errors="replace"
                        )
                        if body_html is None:
                            body_html = html_content
                        else:
                            body_html += html_content
                    elif isinstance(payload_bytes, str):
                        if body_html is None:
                            body_html = payload_bytes
                        else:
                            body_html += payload_bytes
                except Exception as e:
                    logger.warning(f"Failed to decode text/html part: {e}")
    else:
        # Single part message
        content_type = msg.get_content_type()
        try:
            payload_bytes = msg.get_payload(decode=True)
            if isinstance(payload_bytes, bytes):
                payload = payload_bytes.decode(_get_charset(msg), errors="replace")
                if content_type == "text/plain":
                    body_plain = payload
                elif content_type == "text/html":
                    body_html = payload
            elif isinstance(payload_bytes, str):
                if content_type == "text/plain":
                    body_plain = payload_bytes
                elif content_type == "text/html":
                    body_html = payload_bytes
        except Exception as e:
            logger.warning(f"Failed to decode message body: {e}")

    return ParsedEmail(
        message_id=message_id,
        in_reply_to=in_reply_to,
        references=references,
        subject=subject,
        subject_normalized=subject_normalized,
        from_addr=from_addr,
        from_name=from_name,
        to_addrs=to_addrs,
        cc_addrs=cc_addrs,
        bcc_addrs=bcc_addrs,
        date=date_str,
        date_parsed=date_parsed,
        body_plain=body_plain.strip(),
        body_html=body_html.strip() if body_html else None,
        attachments=attachments,
        headers=headers,
    )


def parse_eml_file(path: Path) -> ParsedEmail:
    """Parse an .eml file from disk."""
    content = path.read_bytes()
    return parse_eml(content)


def parse_mbox(path: Path) -> list[ParsedEmail]:
    """
    Parse an mbox file containing multiple emails.

    Args:
        path: Path to .mbox file

    Returns:
        List of ParsedEmail objects
    """
    emails = []
    try:
        mbox = mailbox.mbox(str(path))
        for message in mbox:
            try:
                # Convert mailbox.mboxMessage to bytes
                content = message.as_bytes()
                parsed = parse_eml(content)
                emails.append(parsed)
            except Exception as e:
                logger.warning(f"Failed to parse message in mbox: {e}")
    except Exception as e:
        logger.error(f"Failed to open mbox file {path}: {e}")

    return emails


# =============================================================================
# Threading Logic
# =============================================================================


class EmailThreader:
    """
    Email threading engine implementing Blueprint §6.2 threading rules.

    Threading priority:
    1. References header chain (most reliable)
    2. In-Reply-To header
    3. Clustering by (normalized_subject, participants, time_window)
    """

    def __init__(self, time_window: timedelta = THREAD_TIME_WINDOW):
        self.time_window = time_window
        self.threads: dict[str, EmailThread] = {}
        self.message_to_thread: dict[str, str] = {}
        self.reference_chains: dict[str, str] = {}  # message_id -> thread_id

    def _get_participants(self, email: ParsedEmail) -> set[str]:
        """Get all participants in an email."""
        participants = set()
        if email.from_addr:
            participants.add(email.from_addr)
        participants.update(email.to_addrs)
        participants.update(email.cc_addrs)
        return participants

    def _find_thread_by_references(self, email: ParsedEmail) -> str | None:
        """Try to find thread using References header."""
        # Check each reference, starting from oldest (first)
        for ref in email.references:
            if ref in self.reference_chains:
                return self.reference_chains[ref]
        return None

    def _find_thread_by_in_reply_to(self, email: ParsedEmail) -> str | None:
        """Try to find thread using In-Reply-To header."""
        if email.in_reply_to and email.in_reply_to in self.reference_chains:
            return self.reference_chains[email.in_reply_to]
        return None

    def _find_thread_by_clustering(
        self, email: ParsedEmail
    ) -> tuple[str | None, float]:
        """
        Find thread by clustering on (normalized_subject, participants, time_window).

        Returns:
            Tuple of (thread_id or None, ambiguity_score)
        """
        if not email.subject_normalized:
            return None, 0.0

        participants = self._get_participants(email)
        candidates = []

        for thread_id, thread in self.threads.items():
            # Check subject match
            if thread.subject_normalized != email.subject_normalized:
                continue

            # Check participant overlap (at least 50% shared)
            overlap = len(participants & thread.participants)
            union = len(participants | thread.participants)
            if union == 0 or overlap / union < 0.5:
                continue

            # Check time window
            if email.date_parsed and thread.ended_at:
                time_diff = abs((email.date_parsed - thread.ended_at).total_seconds())
                if time_diff > self.time_window.total_seconds():
                    continue

            candidates.append((thread_id, overlap / union if union > 0 else 0))

        if not candidates:
            return None, 0.0

        # Sort by participant overlap (best match first)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Calculate ambiguity score (high if multiple similar candidates)
        ambiguity = 0.0
        if len(candidates) > 1:
            ambiguity = min(1.0, len(candidates) * 0.2)

        return candidates[0][0], ambiguity

    def _create_thread(self, email: ParsedEmail) -> str:
        """Create a new thread starting with this email."""
        thread_id = str(uuid.uuid4())
        thread = EmailThread(
            thread_id=thread_id,
            subject=email.subject,
            subject_normalized=email.subject_normalized,
            participants=self._get_participants(email),
            message_ids=[email.message_id],
            started_at=email.date_parsed,
            ended_at=email.date_parsed,
        )
        self.threads[thread_id] = thread
        return thread_id

    def _add_to_thread(self, thread_id: str, email: ParsedEmail) -> None:
        """Add an email to an existing thread."""
        thread = self.threads[thread_id]
        thread.message_ids.append(email.message_id)
        thread.participants.update(self._get_participants(email))

        if email.date_parsed:
            if thread.started_at is None or email.date_parsed < thread.started_at:
                thread.started_at = email.date_parsed
            if thread.ended_at is None or email.date_parsed > thread.ended_at:
                thread.ended_at = email.date_parsed

    def thread_email(self, email: ParsedEmail) -> ThreadInfo:
        """
        Determine thread for a single email.

        Args:
            email: Parsed email to thread

        Returns:
            ThreadInfo with thread assignment and metadata
        """
        # Check if already threaded
        if email.message_id in self.message_to_thread:
            cached_thread_id = self.message_to_thread[email.message_id]
            thread = self.threads[cached_thread_id]
            return ThreadInfo(
                thread_id=cached_thread_id,
                position_in_thread=thread.message_ids.index(email.message_id),
                is_thread_start=thread.message_ids[0] == email.message_id,
                threading_method="cached",
            )

        # Try threading methods in priority order
        thread_id: str | None = None
        method = "unknown"
        ambiguity = 0.0

        # 1. Try References header
        thread_id = self._find_thread_by_references(email)
        if thread_id:
            method = "references"

        # 2. Try In-Reply-To header
        if not thread_id:
            thread_id = self._find_thread_by_in_reply_to(email)
            if thread_id:
                method = "in_reply_to"

        # 3. Try clustering
        if not thread_id:
            thread_id, ambiguity = self._find_thread_by_clustering(email)
            if thread_id:
                method = "clustering"

        # 4. Create new thread if no match
        is_thread_start = False
        if not thread_id:
            thread_id = self._create_thread(email)
            method = "new_thread"
            is_thread_start = True
        else:
            self._add_to_thread(thread_id, email)

        assert thread_id is not None
        # Register message in lookups
        self.message_to_thread[email.message_id] = thread_id
        self.reference_chains[email.message_id] = thread_id

        # Also register all references to this thread
        for ref in email.references:
            if ref not in self.reference_chains:
                self.reference_chains[ref] = thread_id

        thread = self.threads[thread_id]
        position = (
            thread.message_ids.index(email.message_id)
            if email.message_id in thread.message_ids
            else 0
        )

        return ThreadInfo(
            thread_id=thread_id,
            position_in_thread=position,
            is_thread_start=is_thread_start,
            threading_method=method,
            ambiguity_score=ambiguity,
        )

    def thread_emails(
        self, emails: list[ParsedEmail]
    ) -> list[tuple[ParsedEmail, ThreadInfo]]:
        """
        Thread a list of emails.

        Args:
            emails: List of parsed emails

        Returns:
            List of (email, thread_info) tuples
        """
        # Sort by date first for better threading
        sorted_emails = sorted(
            emails,
            key=lambda e: e.date_parsed or datetime.min.replace(tzinfo=timezone.utc),
        )

        results = []
        for email in sorted_emails:
            thread_info = self.thread_email(email)
            results.append((email, thread_info))

        return results


# =============================================================================
# Convenience Functions
# =============================================================================


def thread_emails_from_files(paths: list[Path]) -> list[tuple[ParsedEmail, ThreadInfo]]:
    """
    Parse and thread emails from a list of files.

    Supports .eml and .mbox files.

    Args:
        paths: List of file paths

    Returns:
        List of (email, thread_info) tuples
    """
    emails = []

    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".eml":
            emails.append(parse_eml_file(path))
        elif suffix == ".mbox":
            emails.extend(parse_mbox(path))
        else:
            logger.warning(f"Unsupported email format: {path}")

    threader = EmailThreader()
    return threader.thread_emails(emails)


__all__ = [
    "ParsedEmail",
    "ThreadInfo",
    "EmailThread",
    "EmailThreader",
    "parse_eml",
    "parse_eml_file",
    "parse_mbox",
    "thread_emails_from_files",
]
