from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from emailops.common.models import Participant, ParticipantRole
from emailops.util_processing import redact_pii

from .conversation import normalize_subject
from .smtp_resolver import get_recipients_array, get_sender_smtp
from .utils import OL_CC, OL_FULLITEM, OL_TO, to_iso_local, to_iso_z

log = logging.getLogger(__name__)

def looks_like_html(s: str) -> bool:
    import re
    return bool(re.search(r'</?(html|body|div|p|br|table|tr|td|span|a|ul|ol|li)\b', s, re.I))

def html_to_text(s: str) -> str:
    import html as html_lib
    import re
    # Remove script/style
    s = re.sub(r'(?is)<(script|style)[^>]*>.*?</\1>', '', s)
    # Normalize line breaks
    s = re.sub(r'(?is)<br\s*/?>', '\n', s)
    s = re.sub(r'(?is)</p\s*>', '\n\n', s)
    # Strip remaining tags and unescape
    s = re.sub(r'(?s)<[^>]+>', '', s)
    s = html_lib.unescape(s)
    # Tidy whitespace
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n[ \t]+', '\n', s)
    s = re.sub(r'[ \t]+\n', '\n', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()

def _extract_new_message_content(body: str) -> str:
    """Extract new message content and mask thread history and signatures."""
    if not body:
        return ""

    import re

    lines = body.split('\n')
    new_content_lines = []
    quoted_content_lines = []
    in_quoted_section = False

    # Look for very specific multi-line Outlook thread pattern
    for i, line in enumerate(lines):
        if not in_quoted_section and i < len(lines) - 1:
            # Check for "From: " followed by "Sent: " on next line
            current = line.strip()
            next_line = lines[i + 1].strip()

            if (
                current.startswith('From:')
                and len(current) > 6
                and next_line.startswith('Sent:')
                and any(
                    day in next_line
                    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                )
            ):
                in_quoted_section = True
                quoted_content_lines.append(line)
                continue

        if in_quoted_section:
            quoted_content_lines.append(line)
        else:
            new_content_lines.append(line)

    # Join content
    new_content = '\n'.join(new_content_lines).strip()

    # Process main content
    # Remove URL lines
    lines = new_content.split('\n')
    cleaned_lines = []
    for line in lines:
        if '<https://' in line or '<http://' in line:
            url_removed = re.sub(r'<https?://[^>]+>', '', line).strip()
            if len(url_removed) > 10:
                cleaned_lines.append(url_removed)
        else:
            cleaned_lines.append(line)
    new_content = '\n'.join(cleaned_lines)

    # Remove CAUTION disclaimers
    caution_en = "CAUTION: This email originated from outside of the organization. Do not click links or open attachments unless you recognize the sender and know the content is safe."
    caution_ar = "تم إرسال هذا البريد الإلكتروني من خارج مجموعة شلهوب. لا تنقر على الروابط أو تفتح المرفقات إلا إذا كنت تعرف المرسل وتعلم أنه آمن"
    new_content = new_content.replace(caution_en, '')
    new_content = new_content.replace(caution_ar, '')

    # Clean signatures
    new_content = _remove_email_signatures(new_content)

    # Combine new content with masked quoted text
    if quoted_content_lines:
        masked_quoted_text = '\n<quoted_text>\n' + '\n'.join(quoted_content_lines) + '\n</quoted_text>'
        full_content = new_content.strip() + masked_quoted_text
    else:
        full_content = new_content.strip()

    # Enhanced whitespace cleaning
    while '\n\n\n' in full_content:
        full_content = full_content.replace('\n\n\n', '\n\n')
    while '  ' in full_content:
        full_content = full_content.replace('  ', ' ')
    full_content = re.sub(r'\n[ \t]+', '\n', full_content)
    full_content = re.sub(r'[ \t]+\n', '\n', full_content)

    return full_content.strip()

def _remove_email_signatures(text: str) -> str:
    """Remove email signatures and company branding from message content."""
    if not text:
        return ""

    import re

    # Split into lines for analysis
    lines = text.split('\n')

    # Find signature start by looking for common signature patterns
    signature_start = len(lines)  # Default to end if no signature found

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # Strong signature indicators - stop immediately when found
        strong_indicators = [
            # ALL CAPS names after "Regards," or "Best regards,"
            (i > 0 and lines[i-1].strip().lower() in ['regards,', 'best regards,', 'kind regards,'] and
             re.match(r'^[A-Z\s\.]+$', stripped) and len(stripped) > 5),
            # Job titles in ALL CAPS (strong pattern after name)
            re.match(r'^[A-Z\s\.\-]+(MANAGER|ACCOUNTANT|ASSOCIATE|SUPERVISOR|DIRECTOR|CHIEF|HEAD|SPECIALIST)[A-Z\s\.\-]*$', stripped),
            # Company names with LLC/FZE/FZ-LLC
            re.match(r'^[A-Z\s\.\-]+(LLC|FZE|FZ-LLC)$', stripped),
            # Social media URLs (any line containing these)
            'chalhoubgroup.com' in stripped.lower(),
            'facebook.com/chalhoubgroup' in stripped.lower(),
            'twitter.com/chalhoubgroup' in stripped.lower() or 'x.com/chalhoubgroup' in stripped.lower(),
            'instagram.com/chalhoubgroup' in stripped.lower(),
            'linkedin.com/company/chalhoub' in stripped.lower(),
            'glassdoor.com' in stripped.lower(),
            'careers.chalhoubgroup.com' in stripped.lower(),
            # Phone numbers
            re.match(r'^\+971[\s\d]+$', stripped),
        ]

        if any(strong_indicators):
            signature_start = i
            break

    # Also look for signature after natural message endings (be conservative)
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped:
            continue

        # Look for natural message endings - only at end of message
        if (i >= len(lines) - 10 and  # Only in last 10 lines
            (stripped.lower().endswith(('regards,', 'thanks,', 'thank you,', 'best,')) or
             stripped.lower() in ['regards', 'thanks', 'thank you', 'best'])):
            # Cut after this line, but allow for name
            cutoff = i + 1
            # Allow up to 2 more lines for name after regards (but stop at signature content)
            scan_limit = min(len(lines), i + 3)
            for j in range(i + 1, scan_limit):
                if j >= len(lines):
                    break
                next_line = lines[j].strip()
                if next_line and not any(ind in next_line.lower() for ind in ['http', 'chalhoub', 'www.', '.com']):
                    cutoff = j + 1
                else:
                    break
            signature_start = min(signature_start, cutoff)
            break

    # Keep only content before signature
    clean_lines = lines[:signature_start]

    # Remove trailing empty lines
    while clean_lines and not clean_lines[-1].strip():
        clean_lines.pop()

    return '\n'.join(clean_lines)

def _msg_to_dict(item: Any, cached_body: str | None = None, cached_smtp: tuple[str, list[dict[str, str]], list[dict[str, str]]] | None = None) -> dict[str, Any]:
    """Convert Outlook mail item to dictionary.

    Args:
        item: Outlook mail item
        cached_body: Pre-read body text (avoids expensive COM call)
        cached_smtp: Pre-resolved SMTP data as (sender_smtp, to_list, cc_list) tuple
    """
    date_utc_z = to_iso_z(getattr(item, "ReceivedTime", None) or getattr(item, "SentOn", None))

    # Use cached body if provided, otherwise read it (expensive COM call)
    if cached_body is None:
        body = ""
        try:
            download_state = getattr(item, "DownloadState", None)
            if download_state == OL_FULLITEM:
                body = getattr(item, "Body", "") or ""
                if not body:
                    # If Body is empty, try HTMLBody
                    body = getattr(item, "HTMLBody", "") or ""
                if body and looks_like_html(body):
                    body = html_to_text(body)
                # Extract only new message content, removing thread history
                body = _extract_new_message_content(body)
            else:
                body = ""
        except Exception as e:
            log.debug("Failed to get body: %s", e)
    else:
        body = redact_pii(_extract_new_message_content(cached_body or ""))

    # Use cached SMTP data
    if cached_smtp is None:
        sender_smtp = get_sender_smtp(item)
        to_list = get_recipients_array(item, OL_TO)
        cc_list = get_recipients_array(item, OL_CC)
    else:
        sender_smtp, to_list, cc_list = cached_smtp

    sender_name = getattr(item, "SenderName", "") or ""
    try:
        from_participant = Participant(
            name=sender_name if sender_name.strip() else sender_smtp,
            email=sender_smtp,
            role=ParticipantRole.SENDER
        )
    except Exception as e:
        log.warning(f"Could not create sender participant from {sender_name}, {sender_smtp}. Error: {e}")
        from_participant = Participant(name="Unknown", email="unknown@example.com", role=ParticipantRole.SENDER)


    to_participants = []
    for r in to_list:
        if r.get('smtp'):
            try:
                to_participants.append(Participant(
                    name=r.get('name') or r.get('smtp'),
                    email=r.get('smtp'),
                    role=ParticipantRole.RECIPIENT
                ))
            except Exception as e:
                log.warning(f"Could not create TO participant from {r}. Error: {e}")

    cc_participants = []
    for r in cc_list:
        if r.get('smtp'):
            try:
                cc_participants.append(Participant(
                    name=r.get('name') or r.get('smtp'),
                    email=r.get('smtp'),
                    role=ParticipantRole.CC
                ))
            except Exception as e:
                log.warning(f"Could not create CC participant from {r}. Error: {e}")

    return {
        "from": from_participant.model_dump(exclude_none=True),
        "to": [p.model_dump(exclude_none=True) for p in to_participants],
        "cc": [p.model_dump(exclude_none=True) for p in cc_participants],
        "date": date_utc_z,
        "subject": getattr(item, "Subject", "") or "",
        "text": body,
    }

def generate_manifest(mail_items: list[Any], conversation_key: str, output_dir: Path, cached_bodies: list[str] | None = None, cached_smtp_data: list[tuple[str, list[dict[str, str]], list[dict[str, str]]]] | None = None, full_timeline: tuple[datetime, datetime] | None = None, previous_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate manifest.json for a conversation.

    Args:
        mail_items: List of mail items to export (must be pre-sorted if cached data provided)
        conversation_key: Conversation identifier
        output_dir: Output directory path
        cached_bodies: Pre-extracted message bodies (must match mail_items order)
        cached_smtp_data: Pre-resolved SMTP data (must match mail_items order)
        full_timeline: Optional (earliest, latest) datetime across full conversation history
        previous_manifest: Optional previous manifest for incremental merge
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.debug("Unable to ensure output directory %s: %s", output_dir, exc)

    # If cached data is provided, items MUST already be in the same order
    # to maintain alignment. Only sort if no cached data.
    if cached_bodies or cached_smtp_data:
        # Do NOT re-sort - items must stay aligned with cached arrays
        items = list(mail_items)
    else:
        # Safe to sort when no cached data
        def _sort_key(m: Any) -> datetime:
            dt = getattr(m, "ReceivedTime", None) or getattr(m, "SentOn", None)
            return dt if dt is not None else datetime.min
        try:
            items = sorted(mail_items, key=_sort_key)
        except Exception:
            items = list(mail_items)

    # Use cached data if provided to avoid expensive re-reads and Exchange lookups
    if cached_bodies and cached_smtp_data and len(cached_bodies) == len(items) and len(cached_smtp_data) == len(items):
        messages = [_msg_to_dict(it, body, smtp) for it, body, smtp in zip(items, cached_bodies, cached_smtp_data, strict=False)]
    elif cached_bodies and len(cached_bodies) == len(items):
        messages = [_msg_to_dict(it, body) for it, body in zip(items, cached_bodies, strict=False)]
    else:
        messages = [_msg_to_dict(it) for it in items]

    if previous_manifest:
        previous_messages: list[dict[str, Any]] = []
        raw_previous = previous_manifest.get("messages")
        if isinstance(raw_previous, list):
            for msg in cast(list[Any], raw_previous):
                if isinstance(msg, dict):
                    previous_messages.append(cast(dict[str, Any], msg))

        def _message_signature(payload: dict[str, Any]) -> tuple[str, str, str]:
            from_block = payload.get("from")
            email = ""
            if isinstance(from_block, dict):
                email = str(cast(dict[str, Any], from_block).get("email") or "")
            return (
                str(payload.get("date") or ""),
                email,
                str(payload.get("subject") or ""),
            )

        existing_signatures = {_message_signature(msg) for msg in messages}
        for prev in previous_messages:
            signature = _message_signature(prev)
            if signature not in existing_signatures:
                messages.append(prev)
                existing_signatures.add(signature)

        def _message_sort_key(payload: dict[str, Any]) -> tuple[str, str]:
            signature = _message_signature(payload)
            return signature[0], signature[2]

        messages.sort(key=_message_sort_key)

    # Find the chronologically first message for smart_subject
    # Since items might be in any order, find the oldest by date
    oldest_msg_idx = 0
    if len(messages) > 1:
        oldest_date = None
        for idx, msg in enumerate(messages):
            try:
                date_str = msg.get("date", "")
                if date_str:
                    # Parse ISO date
                    if date_str.endswith("Z"):
                        date_str = date_str[:-1] + "+00:00"
                    msg_date = datetime.fromisoformat(date_str)
                    if oldest_date is None or msg_date < oldest_date:
                        oldest_date = msg_date
                        oldest_msg_idx = idx
            except Exception:
                pass

    first_msg = messages[oldest_msg_idx]
    raw_subject = first_msg.get("subject") or ""
    smart_subject = normalize_subject(raw_subject)

    # Find the chronologically last (newest) message for last_message
    last_message_data = None
    if messages:
        newest_msg_idx = 0
        if len(messages) > 1:
            newest_date = None
            for idx, msg in enumerate(messages):
                try:
                    date_str = msg.get("date", "")
                    if date_str:
                        # Parse ISO date
                        if date_str.endswith("Z"):
                            date_str = date_str[:-1] + "+00:00"
                        msg_date = datetime.fromisoformat(date_str)
                        if newest_date is None or msg_date > newest_date:
                            newest_date = msg_date
                            newest_msg_idx = idx
                except Exception:
                    pass

        last_message_data = {
            "from": messages[newest_msg_idx]["from"],
            "to": messages[newest_msg_idx]["to"],
            "cc": messages[newest_msg_idx]["cc"],
            "date": messages[newest_msg_idx]["date"]
        }

    # Use full_timeline if provided, otherwise calculate from items
    if full_timeline:
        timeline_start, timeline_end = full_timeline
        timeline_dict = {
            "start": to_iso_z(timeline_start),
            "end": to_iso_z(timeline_end),
            "start_local": to_iso_local(timeline_start),
            "end_local": to_iso_local(timeline_end),
        }
    else:
        # Find actual min/max dates from items (don't assume order)
        min_date = None
        max_date = None
        for item in items:
            try:
                dt = getattr(item, "ReceivedTime", None) or getattr(item, "SentOn", None)
                if dt:
                    if min_date is None or dt < min_date:
                        min_date = dt
                    if max_date is None or dt > max_date:
                        max_date = dt
            except Exception:
                pass

        timeline_dict = {
            "start": to_iso_z(min_date),
            "end": to_iso_z(max_date),
            "start_local": to_iso_local(min_date),
            "end_local": to_iso_local(max_date),
        }

    if previous_manifest:
        prev_timeline = previous_manifest.get("timeline")
        if isinstance(prev_timeline, dict):
            prev_timeline_dict = cast(dict[str, Any], prev_timeline)
            prev_start = prev_timeline_dict.get("start")
            if not timeline_dict["start"] and isinstance(prev_start, str):
                timeline_dict["start"] = prev_start
            prev_end = prev_timeline_dict.get("end")
            if not timeline_dict["end"] and isinstance(prev_end, str):
                timeline_dict["end"] = prev_end
            prev_start_local = prev_timeline_dict.get("start_local")
            if not timeline_dict["start_local"] and isinstance(prev_start_local, str):
                timeline_dict["start_local"] = prev_start_local
            prev_end_local = prev_timeline_dict.get("end_local")
            if not timeline_dict["end_local"] and isinstance(prev_end_local, str):
                timeline_dict["end_local"] = prev_end_local

    manifest: dict[str, Any] = {
        "conversation_key": conversation_key,
        "smart_subject": smart_subject,
        "messages": messages,  # CRITICAL: Include messages array for incremental export logic
        "last_message": last_message_data,
        "timeline": timeline_dict,
    }
    if previous_manifest and "version" in previous_manifest:
        manifest["previous_manifest_version"] = previous_manifest["version"]
    return manifest

def build_conversation_text(mail_items: list[Any], cached_bodies: list[str] | None = None, cached_smtp_data: list[tuple[str, list[dict[str, str]], list[dict[str, str]]]] | None = None) -> str:
    """Create Conversation.txt content in reverse chronological order (newest first).

    IMPORTANT: mail_items must already be sorted in the desired order, as cached_bodies
    and cached_smtp_data arrays must match the order of mail_items for correct alignment.
    """
    # Do NOT re-sort items here! They are already sorted in _export_conversation
    # and the cached arrays are aligned with that order. Re-sorting would cause
    # mismatched data between items and their cached bodies/SMTP data.
    items = list(mail_items)

    lines: list[str] = []
    for i, it in enumerate(items, start=1):
        from_name = getattr(it, "SenderName", "") or ""
        date_local = to_iso_local(getattr(it, "ReceivedTime", None) or getattr(it, "SentOn", None))
        subject = getattr(it, "Subject", "") or ""

        # Use cached SMTP data if provided, otherwise resolve
        if cached_smtp_data and i <= len(cached_smtp_data):
            sender_smtp, to_list, cc_list = cached_smtp_data[i-1]
        else:
            sender_smtp = get_sender_smtp(it)
            to_list = get_recipients_array(it, OL_TO)
            cc_list = get_recipients_array(it, OL_CC)

        # Cached bodies if provided
        if cached_bodies and i <= len(cached_bodies):
            body = redact_pii(_extract_new_message_content(cached_bodies[i-1] or ""))
        else:
            body = getattr(it, "Body", "") or ""
            if not body:
                body = getattr(it, "HTMLBody", "") or ""
            if body and looks_like_html(body):
                body = html_to_text(body)
            # Extract only new message content, removing thread history
            body = _extract_new_message_content(body)

        lines.append(f"Message {i}")
        lines.append(f"From: {from_name} <{sender_smtp}>".strip())
        lines.append("To: " + ", ".join([f"{r['name']} <{r['smtp']}>".strip() for r in to_list]))
        # Only include CC for Message 1 (newest message in reverse chronological order)
        if cc_list and i == 1:
            lines.append("Cc: " + ", ".join([f"{r['name']} <{r['smtp']}>".strip() for r in cc_list]))
        lines.append(f"Date: {date_local}")
        lines.append(f"Subject: {subject}")
        lines.append("")
        lines.append(body.rstrip())
        lines.append("")
        lines.append("-" * 72)

    # Final cleanup pass on entire conversation text
    conversation_text = "\n".join(lines).rstrip() + "\n"

    # Aggressive newline cleanup - limit to max 2 consecutive
    # Replace 3+ newlines with exactly 2
    while '\n\n\n' in conversation_text:
        conversation_text = conversation_text.replace('\n\n\n', '\n\n')

    return conversation_text
