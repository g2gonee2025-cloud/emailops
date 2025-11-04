from __future__ import annotations

import logging
import re
from typing import Any

from .mapitags import (
    PR_CONVERSATION_ID,
    PR_CONVERSATION_INDEX,
    PR_IN_REPLY_TO_ID,
    PR_REFERENCES,
)

log = logging.getLogger(__name__)

def adler32_seed(text: str | None) -> str:
    """Return 8-char hex Adler32 implemented to match PowerShell Get-Adler32Hex.
    Format is {b:04X}{a:04X} where a and b are the Adler sums mod 65521.
    """
    if text is None:
        text = ""
    data = text.encode("utf-8", errors="ignore")
    a, b = 1, 0
    MOD = 65521
    for byte in data:
        a = (a + byte) % MOD
        b = (b + a) % MOD
    return f"{b:04X}{a:04X}"

def normalize_subject(subject: str) -> str:
    """Normalize subject by removing reply/forward prefixes (case-insensitive).
    Strips repeatedly (e.g., 'Re:Re: FW:Hello' -> 'hello'), accepts prefixes with/without spaces.
    Returns 'no subject' if empty after normalization.
    """
    if not subject:
        return "no subject"
    s = subject.strip()
    rx = re.compile(r"^(re|fw|fwd|sv|aw|wg|antw)\s*:\s*", re.IGNORECASE)
    prev: str | None = None
    while s and s != prev:
        prev = s
        s = rx.sub("", s, count=1).lstrip()
    s = s.lower()
    return s if s else "no subject"

def get_conversation_key(mail_item: Any) -> tuple[str, bool]:
    """Generate conversation key identical to the PowerShell heuristic.
    1) RFC-style threading via PR_REFERENCES or PR_IN_REPLY_TO_ID
    2) ConversationID when available
    3) ConversationIndex (hex string) via PropertyAccessor
    4) Subject + sender + minute-round received time + EntryID-derived seed

    PERFORMANCE: Optimized to fail fast on missing properties.

    Returns:
        tuple[str, bool]: (conversation_key, is_fallback)
        - is_fallback is True when using subject-based heuristic (method 4)
        - is_fallback is False when using standard conversation properties
    """
    # Get PropertyAccessor once and reuse (expensive COM call)
    pa = None
    try:
        pa = getattr(mail_item, "PropertyAccessor", None)
    except Exception:
        pa = None

    if pa is not None:
        # 1) RFC-style threading
        try:
            # PR_REFERENCES is the most reliable
            references = pa.GetProperty(PR_REFERENCES)
            if references:
                # The first reference is often the root of the thread
                root_ref = references.split()
                return (f"REF|{root_ref}", False)
        except Exception:
            pass

        try:
            in_reply_to = pa.GetProperty(PR_IN_REPLY_TO_ID)
            if in_reply_to:
                return (f"REPLY|{in_reply_to}", False)
        except Exception:
            pass

    # 2) ConversationID - try direct property first (fastest)
    try:
        conv_id = getattr(mail_item, "ConversationID", None)
        if conv_id:
            return (f"CID|{conv_id}", False)
    except Exception:
        pass

    if pa is not None:
        # 2b) ConversationID via PropertyAccessor (bytes->hex) if direct property missing
        try:
            conv_id_bytes = pa.GetProperty(PR_CONVERSATION_ID)
            if conv_id_bytes:
                hex_str = ''.join(f'{b:02X}' for b in conv_id_bytes)
                return (f"CID|{hex_str}", False)
        except Exception:
            pass

        # 3) ConversationIndex - only if PropertyAccessor is available
        try:
            conv_idx_bytes = pa.GetProperty(PR_CONVERSATION_INDEX)
            if conv_idx_bytes:
                hex_str = ''.join(f'{b:02X}' for b in conv_idx_bytes)
                return (f"CIX|{hex_str}", False)
        except Exception:
            pass

    # 4) Subject-based fallback (most expensive, but unavoidable)
    log.warning("Using subject-based fallback for conversation key (ConversationID/Index/References unavailable)")
    try:
        topic = getattr(mail_item, "ConversationTopic", None) or getattr(mail_item, "Subject", "") or ""
    except Exception:
        topic = getattr(mail_item, "Subject", "") or ""
    subject_norm = normalize_subject(topic)
    sender = (getattr(mail_item, "SenderEmailAddress", "") or "").lower()
    try:
        rcv = getattr(mail_item, "ReceivedTime", None)
        dt_key = rcv.strftime("%Y%m%d%H%M") if rcv else "000000000000"
    except Exception:
        dt_key = "000000000000"
    # EntryID-based seed (first 4 chars of adler32 hex)
    try:
        eid = getattr(mail_item, "EntryID", "") or ""
    except Exception:
        eid = ""
    seed = adler32_seed(eid)[:4]
    return (f"SBJ|{subject_norm}|SND|{sender}|DT|{dt_key}|SEED|{seed}", True)
