"""
Policy Enforcer.

Implements §11.2 of the Canonical Blueprint.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Literal, Optional, Set

from cortex.observability import trace_operation
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class PolicyDecision(BaseModel):
    """
    Policy decision result.

    Blueprint §9.1:
    * action: str
    * decision: Literal["allow", "deny", "require_approval"]
    * reason: str
    * risk_level: Literal["low", "medium", "high"]
    * metadata: Dict[str, Any]
    """

    model_config = ConfigDict(extra="forbid")

    action: str
    decision: Literal["allow", "deny", "require_approval"]
    reason: str
    risk_level: Literal["low", "medium", "high"]
    metadata: Dict[str, Any] = Field(default_factory=dict)


# -----------------------------------------------------------------------------
# Policy Configuration
# -----------------------------------------------------------------------------

# Actions by risk level
LOW_RISK_ACTIONS: Set[str] = {
    "search",
    "read_thread",
    "read_message",
    "get_thread_context",
    "summarize_thread",
    "answer_question",
}

MEDIUM_RISK_ACTIONS: Set[str] = {
    "draft_email",
    "create_draft",
    "modify_draft",
    "upload_attachment",
}

HIGH_RISK_ACTIONS: Set[str] = {
    "send_email",
    "delete_message",
    "delete_thread",
    "delete_attachment",
    "admin_action",
    "export_data",
    "bulk_operation",
}

# Patterns that indicate external communication
EXTERNAL_DOMAIN_PATTERN = re.compile(r"@(?!internal\.company\.com$)", re.IGNORECASE)

# Maximum recipients for auto-approval
MAX_RECIPIENTS_AUTO_APPROVE = 10

# Sensitive content patterns
SENSITIVE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(confidential|secret|private|internal only)\b", re.IGNORECASE),
    re.compile(r"\b(password|credential|api.?key|token)\b", re.IGNORECASE),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN pattern
    re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),  # Credit card pattern
]


# -----------------------------------------------------------------------------
# Policy Checks
# -----------------------------------------------------------------------------


def _check_recipient_policy(metadata: Dict[str, Any]) -> Optional[str]:
    """Check recipient-related policies."""
    recipients = metadata.get("recipients", [])

    if not recipients:
        return None

    # Check recipient count
    if len(recipients) > MAX_RECIPIENTS_AUTO_APPROVE:
        return (
            f"Too many recipients ({len(recipients)} > {MAX_RECIPIENTS_AUTO_APPROVE})"
        )

    # Check for external domains
    external_recipients = [r for r in recipients if EXTERNAL_DOMAIN_PATTERN.search(r)]
    if external_recipients and metadata.get("check_external", True):
        return f"External recipients detected: {', '.join(external_recipients[:3])}"

    return None


def _check_content_policy(metadata: Dict[str, Any]) -> Optional[str]:
    """Check content-related policies."""
    content = metadata.get("content", "") or ""
    subject = metadata.get("subject", "") or ""
    full_text = f"{subject} {content}"

    for pattern in SENSITIVE_PATTERNS:
        if pattern.search(full_text):
            return f"Sensitive content detected (pattern: {pattern.pattern[:30]}...)"

    return None


def _check_attachment_policy(metadata: Dict[str, Any]) -> Optional[str]:
    """Check attachment-related policies."""
    attachments = metadata.get("attachments", [])

    if not attachments:
        return None

    # Check for dangerous extensions
    dangerous_extensions = {".exe", ".bat", ".cmd", ".ps1", ".vbs", ".js"}

    for attachment in attachments:
        filename = attachment.get("filename", "").lower()
        for ext in dangerous_extensions:
            if filename.endswith(ext):
                return f"Dangerous attachment type: {ext}"

    # Check total size
    total_size = sum(a.get("size", 0) for a in attachments)
    max_size = metadata.get("max_attachment_size", 25 * 1024 * 1024)  # 25MB default

    if total_size > max_size:
        return f"Attachment size exceeds limit ({total_size} > {max_size})"

    return None


def _determine_risk_level(action: str) -> Literal["low", "medium", "high"]:
    """Determine the risk level for an action."""
    if action in LOW_RISK_ACTIONS:
        return "low"
    elif action in MEDIUM_RISK_ACTIONS:
        return "medium"
    elif action in HIGH_RISK_ACTIONS:
        return "high"
    else:
        # Unknown actions default to medium risk
        logger.warning(f"Unknown action '{action}' defaulting to medium risk")
        return "medium"


# -----------------------------------------------------------------------------
# Main Policy Check
# -----------------------------------------------------------------------------


@trace_operation("check_action")
def check_action(action: str, metadata: Dict[str, Any]) -> PolicyDecision:
    """
    Check if an action is allowed based on policies.

    Blueprint §11.2:
    * Map user + context -> PolicyDecision
    * Low risk: auto-allow
    * Medium risk: allow with logging
    * High risk: require approval or deny

    Args:
        action: The action being performed
        metadata: Context about the action (recipients, content, etc.)

    Returns:
        PolicyDecision with allow/deny/require_approval
    """
    risk_level = _determine_risk_level(action)
    violations: List[str] = []

    # Run policy checks
    if recipient_issue := _check_recipient_policy(metadata):
        violations.append(recipient_issue)

    if content_issue := _check_content_policy(metadata):
        violations.append(content_issue)

    if attachment_issue := _check_attachment_policy(metadata):
        violations.append(attachment_issue)

    # Determine decision based on risk level and violations
    decision: Literal["allow", "deny", "require_approval"]
    reason: str

    if violations:
        # Any violation on high-risk action = deny
        if risk_level == "high":
            decision = "deny"
            reason = f"Policy violation on high-risk action: {'; '.join(violations)}"
        # Violations on medium-risk = require approval
        elif risk_level == "medium":
            decision = "require_approval"
            reason = f"Policy check required: {'; '.join(violations)}"
        # Violations on low-risk = still allow but log
        else:
            decision = "allow"
            reason = f"Allowed with warnings: {'; '.join(violations)}"
            logger.warning(
                f"Low-risk action '{action}' has policy warnings: {violations}"
            )
    else:
        # No violations
        if risk_level == "high":
            # High-risk actions without violations still require approval
            decision = "require_approval"
            reason = "High-risk action requires approval"
        else:
            decision = "allow"
            reason = "No policy violations"

    # Check for explicit deny list
    if metadata.get("force_deny"):
        decision = "deny"
        reason = "Explicitly denied by policy"

    # Check for bypass (e.g., admin override)
    if metadata.get("admin_bypass") and decision != "deny":
        decision = "allow"
        reason = "Admin bypass enabled"

    return PolicyDecision(
        action=action,
        decision=decision,
        reason=reason,
        risk_level=risk_level,
        metadata={
            "violations": violations,
            "original_metadata_keys": list(metadata.keys()),
        },
    )


def require_policy_approval(decision: PolicyDecision) -> bool:
    """Check if a policy decision requires human approval."""
    return decision.decision == "require_approval"


def is_action_allowed(decision: PolicyDecision) -> bool:
    """Check if an action is allowed (either directly or after approval)."""
    return decision.decision in ("allow", "require_approval")
