from __future__ import annotations

import logging
from typing import Any

from emailops.core_exceptions import ProcessingError
from emailops.utils import safe_str

OutlookItem = Any

log = logging.getLogger(__name__)

def get_sender_smtp(mail_item: OutlookItem) -> str:
    """Resolve MailItem sender to SMTP address.
    Mirrors PowerShell Get-SenderSmtp.
    Returns empty string if sender has no SMTP address OR if resolution fails.
    Check logs for warnings to distinguish error cases.
    """
    try:
        email_type = getattr(mail_item, "SenderEmailType", None)
        if email_type == "EX":
            try:
                sender = getattr(mail_item, "Sender", None)
                if sender:
                    try:
                        ex_user = sender.GetExchangeUser()
                        if ex_user and getattr(ex_user, "PrimarySmtpAddress", None):
                            return ex_user.PrimarySmtpAddress
                    except Exception as e:
                        log.debug("Exchange user lookup failed for sender: %s", e)
                    try:
                        ex_dl = sender.GetExchangeDistributionList()
                        if ex_dl and getattr(ex_dl, "PrimarySmtpAddress", None):
                            return ex_dl.PrimarySmtpAddress
                    except Exception as e:
                        log.debug("Exchange DL lookup failed for sender: %s", e)
                else:
                    log.warning("Sender object is None for Exchange-type sender")
            except Exception as e:
                log.warning("Failed to access Sender property: %s", e)
        # Fallback to SenderEmailAddress
        fallback = getattr(mail_item, "SenderEmailAddress", "") or ""
        if not fallback and email_type == "EX":
            log.warning("Exchange sender resolution failed, no fallback address available")
        return fallback
    except Exception as exc:
        log.error("Critical failure resolving sender SMTP: %s", exc)
        raise ProcessingError("Failed to resolve sender SMTP address") from exc

def _addr_entry_to_smtp(addr_entry: OutlookItem) -> str:
    if not addr_entry:
        return ""
    try:
        addr_type = getattr(addr_entry, "Type", None)
        if addr_type == "EX":
            try:
                ex_user = addr_entry.GetExchangeUser()
                if ex_user and getattr(ex_user, "PrimarySmtpAddress", None):
                    return ex_user.PrimarySmtpAddress
            except Exception as e:
                log.debug("Exchange user lookup failed for recipient: %s", e)
            try:
                ex_dl = addr_entry.GetExchangeDistributionList()
                if ex_dl and getattr(ex_dl, "PrimarySmtpAddress", None):
                    return ex_dl.PrimarySmtpAddress
            except Exception as e:
                log.debug("Exchange DL lookup failed for recipient: %s", e)
            log.warning("Exchange recipient resolution failed for AddressEntry (no PrimarySmtpAddress)")
        # Fallback to Address property
        if getattr(addr_entry, "Address", None):
            return addr_entry.Address or ""
    except Exception as e:
        log.warning("AddressEntry SMTP resolution failed: %s", e)
    return ""

def get_recipient_smtp(recipient: OutlookItem) -> str:
    ae = getattr(recipient, "AddressEntry", None)
    return _addr_entry_to_smtp(ae)

def get_recipients_array(mail_item: OutlookItem, recipient_type: int) -> list[dict[str, str]]:
    recips = []
    dropped_count = 0
    try:
        recipients = getattr(mail_item, "Recipients", None)
        if recipients is None:
            return recips
        for r in recipients:
            try:
                if getattr(r, "Type", None) == recipient_type:
                    name = safe_str(getattr(r, "Name", ""), 255)
                    smtp = get_recipient_smtp(r)
                    if name or smtp: # Only add if name or smtp is present
                        recips.append({"name": name, "smtp": smtp})
            except Exception as e:
                dropped_count += 1
                log.warning("Failed to process recipient (type=%s): %s", recipient_type, e)
                continue
        if dropped_count > 0:
            log.error("Dropped %d recipients due to processing errors (type=%s)", dropped_count, recipient_type)
    except Exception as e:
        log.error("Failed to enumerate recipients: %s", e)
    return recips
