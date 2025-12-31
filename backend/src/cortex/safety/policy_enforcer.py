"""
Policy Enforcer.

Implements §11.2 of the Canonical Blueprint.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from cortex.observability import trace_operation
from cortex.safety.config import PolicyConfig
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

_POLICY_CONFIG: PolicyConfig | None = None


def _get_policy_config() -> PolicyConfig:
    global _POLICY_CONFIG
    if _POLICY_CONFIG is None:
        try:
            _POLICY_CONFIG = PolicyConfig()
        except Exception as exc:
            logger.exception("Failed to load policy configuration.")
            raise RuntimeError("Policy configuration failed to load.") from exc
    return _POLICY_CONFIG


PolicyViolation = tuple[str, str]


class PolicyDecision(BaseModel):
    """
    Policy decision result.

    Blueprint §9.1:
    * action: str
    * decision: Literal["allow", "deny", "require_approval"]
    * reason: str
    * risk_level: Literal["low", "medium", "high"]
    * metadata: dict[str, Any]
    """

    model_config = ConfigDict(extra="forbid")

    action: str
    decision: Literal["allow", "deny", "require_approval"]
    reason: str
    risk_level: Literal["low", "medium", "high"]
    metadata: dict[str, Any] = Field(default_factory=dict)


# -----------------------------------------------------------------------------
# Policy Checks
# -----------------------------------------------------------------------------


def _check_recipient_policy(metadata: dict[str, Any]) -> PolicyViolation | None:
    """Check recipient-related policies."""
    recipients_value = metadata.get("recipients")

    if recipients_value is None:
        return None

    recipients: list[str]
    if isinstance(recipients_value, str):
        recipients = [recipients_value]
    elif isinstance(recipients_value, (list, tuple, set)):
        recipients = [r for r in recipients_value if isinstance(r, str)]
        if recipients_value and not recipients:
            return ("invalid_recipients", "Invalid recipient metadata")
    else:
        try:
            recipients = [r for r in list(recipients_value) if isinstance(r, str)]
            if recipients_value and not recipients:
                return ("invalid_recipients", "Invalid recipient metadata")
        except TypeError:
            return ("invalid_recipients", "Invalid recipient metadata")

    if not recipients:
        return None

    # Check recipient count
    config = _get_policy_config()
    if len(recipients) > config.max_recipients_auto_approve:
        return (
            "recipient_count",
            f"Too many recipients ({len(recipients)} > {config.max_recipients_auto_approve})",
        )

    # Check for external domains
    external_domain_pattern = config.get_external_domain_pattern()
    external_recipients = [
        r
        for r in recipients
        if isinstance(r, str) and external_domain_pattern.search(r)
    ]
    if external_recipients and metadata.get("check_external", True):
        return (
            "external_recipients",
            f"External recipients detected (count={len(external_recipients)})",
        )

    return None


def _check_content_policy(metadata: dict[str, Any]) -> PolicyViolation | None:
    """Check content-related policies."""
    content = metadata.get("content", "") or ""
    subject = metadata.get("subject", "") or ""
    full_text = f"{subject} {content}"

    sensitive_patterns = _get_policy_config().get_sensitive_patterns()
    for pattern in sensitive_patterns:
        if pattern.search(full_text):
            return ("sensitive_content", "Sensitive content detected")

    return None


def _check_attachment_policy(metadata: dict[str, Any]) -> PolicyViolation | None:
    """Check attachment-related policies."""
    attachments_value = metadata.get("attachments")

    if not attachments_value:
        return None

    if isinstance(attachments_value, dict):
        attachments = [attachments_value]
    elif isinstance(attachments_value, (list, tuple, set)):
        attachments = list(attachments_value)
    else:
        try:
            attachments = list(attachments_value)
        except TypeError:
            return ("invalid_attachment_metadata", "Invalid attachment metadata")

    if not attachments:
        return None

    if any(not isinstance(attachment, dict) for attachment in attachments):
        return ("invalid_attachment_metadata", "Invalid attachment metadata")

    # Check for dangerous extensions
    dangerous_extensions = _get_policy_config().dangerous_extensions
    for attachment in attachments:
        filename = attachment.get("filename", "")
        if not isinstance(filename, str):
            continue
        filename = filename.lower()
        for ext in dangerous_extensions:
            if filename.endswith(ext):
                return ("dangerous_attachment", f"Dangerous attachment type: {ext}")

    # Check total size
    total_size = 0
    invalid_size = False
    for attachment in attachments:
        size = attachment.get("size", 0)
        try:
            size_value = int(size)
        except (TypeError, ValueError):
            invalid_size = True
            size_value = 0
        if size_value < 0:
            invalid_size = True
            size_value = 0
        total_size += size_value

    if invalid_size:
        return ("invalid_attachment_size", "Invalid attachment size metadata")

    max_size_mb = _get_policy_config().max_attachment_size_mb
    max_size_bytes = max_size_mb * 1024 * 1024

    if total_size > max_size_bytes:
        return (
            "attachment_size",
            f"Attachment size exceeds limit ({total_size} > {max_size_bytes})",
        )

    return None


def _determine_risk_level(action: str) -> Literal["low", "medium", "high"]:
    """Determine the risk level for an action."""
    config = _get_policy_config()
    if action in config.low_risk_actions:
        return "low"
    elif action in config.medium_risk_actions:
        return "medium"
    elif action in config.high_risk_actions:
        return "high"
    else:
        # Unknown actions default to medium risk
        logger.warning(f"Unknown action '{action}' defaulting to medium risk")
        return "medium"


# -----------------------------------------------------------------------------
# Main Policy Check
# -----------------------------------------------------------------------------


@trace_operation("check_action")
def check_action(action: str, metadata: dict[str, Any] | None = None) -> PolicyDecision:
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
    violations: list[PolicyViolation] = []

    # Run policy checks (defaults to safe empty structures if keys missing)
    safe_meta = metadata if isinstance(metadata, dict) else {}

    if recipient_issue := _check_recipient_policy(safe_meta):
        violations.append(recipient_issue)

    if content_issue := _check_content_policy(safe_meta):
        violations.append(content_issue)

    if attachment_issue := _check_attachment_policy(safe_meta):
        violations.append(attachment_issue)

    # Determine decision based on risk level and violations
    decision: Literal["allow", "deny", "require_approval"]
    reason: str

    violation_messages = [message for _, message in violations]
    violation_codes = [code for code, _ in violations]

    if violations:
        # Any violation on high-risk action = deny
        if risk_level == "high":
            decision = "deny"
            reason = (
                "Policy violation on high-risk action: "
                f"{'; '.join(violation_messages)}"
            )
        # Violations on medium-risk = require approval
        elif risk_level == "medium":
            decision = "require_approval"
            reason = f"Policy check required: {'; '.join(violation_messages)}"
        # Violations on low-risk = still allow but log
        else:
            decision = "allow"
            reason = f"Allowed with warnings: {'; '.join(violation_messages)}"
            logger.warning(
                "Low-risk action '%s' has policy warnings (%d).",
                action,
                len(violations),
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
    if safe_meta.get("force_deny") is True:
        decision = "deny"
        reason = "Explicitly denied by policy"

    # Check for bypass (e.g., admin override)
    # SECURITY: Check for admin role logic
    # Admin can bypass "require_approval" but NOT "deny".
    if decision != "deny":
        # Check roles (support both singular 'role' and list 'user_roles' for compatibility)
        roles: list[str] = []
        roles_value = safe_meta.get("user_roles")
        if isinstance(roles_value, str):
            roles.append(roles_value)
        elif isinstance(roles_value, (list, tuple, set)):
            roles.extend([role for role in roles_value if isinstance(role, str)])
        elif roles_value is not None:
            try:
                roles.extend(
                    [role for role in list(roles_value) if isinstance(role, str)]
                )
            except TypeError:
                roles = []

        role_value = safe_meta.get("role")
        if isinstance(role_value, str):
            roles.append(role_value)

        if safe_meta.get("roles_verified") is True and "admin" in roles:
            decision = "allow"
            reason = "Admin bypass enabled"

    return PolicyDecision(
        action=action,
        decision=decision,
        reason=reason,
        risk_level=risk_level,
        metadata={
            "violation_codes": violation_codes,
            "violation_count": len(violation_codes),
            "original_metadata_keys": list(safe_meta.keys()),
        },
    )


def require_policy_approval(decision: PolicyDecision) -> bool:
    """Check if a policy decision requires human approval."""
    return decision.decision == "require_approval"


def is_action_allowed(decision: PolicyDecision) -> bool:
    """Check if an action is allowed (either directly or after approval)."""
    return decision.decision == "allow"
