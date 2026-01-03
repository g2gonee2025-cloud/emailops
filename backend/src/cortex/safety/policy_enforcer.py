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

_INVALID_RECIPIENT_METADATA = "Invalid recipient metadata"


def _parse_recipients(
    recipients_value: Any,
) -> tuple[list[str] | None, PolicyViolation | None]:
    """Parse and validate recipient metadata."""
    if recipients_value is None:
        return None, None

    recipients: list[str]
    if isinstance(recipients_value, str):
        recipients = [recipients_value]
    elif isinstance(recipients_value, (list, tuple, set)):
        recipients = [r for r in recipients_value if isinstance(r, str)]
        if recipients_value and not recipients:
            return None, ("invalid_recipients", _INVALID_RECIPIENT_METADATA)
    else:
        try:
            recipients = [r for r in recipients_value if isinstance(r, str)]
            if recipients_value and not recipients:
                return None, ("invalid_recipients", _INVALID_RECIPIENT_METADATA)
        except TypeError:
            return None, ("invalid_recipients", _INVALID_RECIPIENT_METADATA)

    return recipients, None


def _check_recipient_policy(metadata: dict[str, Any]) -> PolicyViolation | None:
    """Check recipient-related policies."""
    recipients, error = _parse_recipients(metadata.get("recipients"))

    if error:
        return error
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


def _parse_attachments(
    attachments_value: Any,
) -> tuple[list[dict] | None, PolicyViolation | None]:
    """Parse and validate attachment metadata."""
    if not attachments_value:
        return None, None

    attachments: list[dict]
    if isinstance(attachments_value, dict):
        attachments = [attachments_value]
    elif isinstance(attachments_value, (list, tuple, set)):
        attachments = list(attachments_value)
    else:
        try:
            attachments = list(attachments_value)
        except TypeError:
            return None, ("invalid_attachment_metadata", "Invalid attachment metadata")

    if not attachments:
        return None, None

    if any(not isinstance(attachment, dict) for attachment in attachments):
        return None, ("invalid_attachment_metadata", "Invalid attachment metadata")

    return attachments, None


def _check_dangerous_extensions(attachments: list[dict]) -> PolicyViolation | None:
    """Check for dangerous file extensions in attachments."""
    dangerous_extensions = _get_policy_config().dangerous_extensions
    for attachment in attachments:
        filename = attachment.get("filename", "")
        if not isinstance(filename, str):
            continue
        filename = filename.lower()
        for ext in dangerous_extensions:
            if filename.endswith(ext):
                return "dangerous_attachment", f"Dangerous attachment type: {ext}"
    return None


def _check_attachment_size(attachments: list[dict]) -> PolicyViolation | None:
    """Check total size of attachments."""
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
        return "invalid_attachment_size", "Invalid attachment size metadata"

    max_size_mb = _get_policy_config().max_attachment_size_mb
    max_size_bytes = max_size_mb * 1024 * 1024

    if total_size > max_size_bytes:
        return (
            "attachment_size",
            f"Attachment size exceeds limit ({total_size} > {max_size_bytes})",
        )
    return None


def _check_attachment_policy(metadata: dict[str, Any]) -> PolicyViolation | None:
    """Check attachment-related policies."""
    attachments, error = _parse_attachments(metadata.get("attachments"))
    if error:
        return error
    if not attachments:
        return None

    if extension_violation := _check_dangerous_extensions(attachments):
        return extension_violation

    if size_violation := _check_attachment_size(attachments):
        return size_violation

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


def _determine_decision_from_violations(
    risk_level: Literal["low", "medium", "high"],
    violations: list[PolicyViolation],
    action: str,
) -> tuple[Literal["allow", "deny", "require_approval"], str]:
    """Determine the policy decision based on risk level and violations."""
    violation_messages = [message for _, message in violations]

    if violations:
        if risk_level == "high":
            return (
                "deny",
                f"Policy violation on high-risk action: {'; '.join(violation_messages)}",
            )
        elif risk_level == "medium":
            return (
                "require_approval",
                f"Policy check required: {'; '.join(violation_messages)}",
            )
        else:
            logger.warning(
                "Low-risk action '%s' has policy warnings (%d).",
                action,
                len(violations),
            )
            return "allow", f"Allowed with warnings: {'; '.join(violation_messages)}"
    else:
        if risk_level == "high":
            return "require_approval", "High-risk action requires approval"
        else:
            return "allow", "No policy violations"


def _parse_user_roles(safe_meta: dict[str, Any]) -> list[str]:
    """Parse user roles from metadata."""
    roles: list[str] = []
    roles_value = safe_meta.get("user_roles")
    if isinstance(roles_value, str):
        roles.append(roles_value)
    elif isinstance(roles_value, (list, tuple, set)):
        roles.extend([role for role in roles_value if isinstance(role, str)])
    elif roles_value is not None:
        try:
            roles.extend([role for role in roles_value if isinstance(role, str)])
        except TypeError:
            pass  # roles remains empty

    role_value = safe_meta.get("role")
    if isinstance(role_value, str):
        roles.append(role_value)
    return roles


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

    safe_meta = metadata if isinstance(metadata, dict) else {}

    if recipient_issue := _check_recipient_policy(safe_meta):
        violations.append(recipient_issue)
    if content_issue := _check_content_policy(safe_meta):
        violations.append(content_issue)
    if attachment_issue := _check_attachment_policy(safe_meta):
        violations.append(attachment_issue)

    decision, reason = _determine_decision_from_violations(
        risk_level, violations, action
    )

    if safe_meta.get("force_deny") is True:
        decision = "deny"
        reason = "Explicitly denied by policy"

    # Admin can bypass "require_approval" but NOT "deny".
    if decision != "deny" and safe_meta.get("roles_verified") is True:
        roles = _parse_user_roles(safe_meta)
        if "admin" in roles:
            decision = "allow"
            reason = "Admin bypass enabled"

    violation_codes = [code for code, _ in violations]

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
