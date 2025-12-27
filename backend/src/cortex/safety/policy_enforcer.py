"""
Policy Enforcer.

Implements §11.2 of the Canonical Blueprint.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional

from cortex.observability import trace_operation
from cortex.safety.config import PolicyConfig
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# Load the policy configuration
POLICY_CONFIG = PolicyConfig()


class DLPProvider(ABC):
    """Abstract base class for a Data Loss Prevention (DLP) provider."""

    @abstractmethod
    async def scan(self, text: str) -> List[str]:
        """Scans the text for sensitive information and returns a list of violations."""
        pass


class MockDLPProvider(DLPProvider):
    """
    A mock DLP provider for local testing.

    This provider simulates an async network call and uses regex to find
    common sensitive patterns. A production implementation would use a more
    robust service.
    """
    PATTERNS = {
        "API_KEY": re.compile(r"(sk-[a-zA-Z0-9]{20,})"),
        "CREDIT_CARD": re.compile(r"\b(\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4})\b"),
        "SSN": re.compile(r"\b(\d{3}-\d{2}-\d{4})\b"),
    }

    async def scan(self, text: str) -> List[str]:
        """Simulates an async DLP scan."""
        await asyncio.sleep(0.01)  # Simulate network latency
        violations = []
        for pii_type, pattern in self.PATTERNS.items():
            if pattern.search(text):
                violations.append(pii_type)
        return violations


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
# Policy Checks
# -----------------------------------------------------------------------------


def _check_recipient_policy(metadata: Dict[str, Any]) -> Optional[str]:
    """
    Check recipient-related policies and handle PII securely.

    PII (email addresses) are hashed for secure logging and auditing.
    """
    recipients = metadata.get("recipients", [])
    if not recipients:
        return None

    # Check recipient count
    max_recipients = POLICY_CONFIG.max_recipients_auto_approve
    if len(recipients) > max_recipients:
        return f"Too many recipients ({len(recipients)} > {max_recipients})"

    # Check for external domains
    external_domain_pattern = POLICY_CONFIG.get_external_domain_pattern()
    external_recipients = [
        r for r in recipients if isinstance(r, str) and external_domain_pattern.search(r)
    ]

    if external_recipients and metadata.get("check_external", True):
        # Hash the external recipient list for secure auditing
        hashed_recipients = [
            hashlib.sha256(r.encode()).hexdigest() for r in external_recipients
        ]
        metadata["hashed_external_recipients"] = hashed_recipients

        # Return a PII-safe violation message
        return f"{len(external_recipients)} external recipients detected."

    return None


async def _check_content_policy(metadata: Dict[str, Any]) -> Optional[str]:
    """
    Check content-related policies using an async, pluggable DLP provider.
    """
    content = metadata.get("content", "") or ""
    subject = metadata.get("subject", "") or ""
    full_text = f"{subject}\n{content}"

    # In a real-world scenario, you would have a more sophisticated DLP
    # provider that could be selected based on configuration.
    # For this example, we'll use a mock provider that simulates an async call.
    dlp_provider = MockDLPProvider()
    violations = await dlp_provider.scan(full_text)

    if violations:
        return f"Sensitive content detected: {', '.join(violations)}"

    return None


def _check_attachment_policy(metadata: Dict[str, Any]) -> Optional[str]:
    """Check attachment-related policies."""
    attachments = metadata.get("attachments", [])

    if not attachments:
        return None

    # Check for dangerous extensions
    dangerous_extensions = POLICY_CONFIG.dangerous_extensions
    for attachment in attachments:
        filename = attachment.get("filename", "").lower()
        for ext in dangerous_extensions:
            if filename.endswith(ext):
                return f"Dangerous attachment type: {ext}"

    # Check total size
    total_size = sum(a.get("size", 0) for a in attachments)
    max_size_mb = POLICY_CONFIG.max_attachment_size_mb
    max_size_bytes = metadata.get("max_attachment_size", max_size_mb * 1024 * 1024)

    if total_size > max_size_bytes:
        return f"Attachment size exceeds limit ({total_size} > {max_size_bytes})"

    return None


def _determine_risk_level(action: str) -> Literal["low", "medium", "high"]:
    """Determine the risk level for an action."""
    if action in POLICY_CONFIG.low_risk_actions:
        return "low"
    elif action in POLICY_CONFIG.medium_risk_actions:
        return "medium"
    elif action in POLICY_CONFIG.high_risk_actions:
        return "high"
    else:
        # Unknown actions default to medium risk
        logger.warning(f"Unknown action '{action}' defaulting to medium risk")
        return "medium"


# -----------------------------------------------------------------------------
# Main Policy Check
# -----------------------------------------------------------------------------


@trace_operation("check_action")
async def check_action(action: str, metadata: Dict[str, Any]) -> PolicyDecision:
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

    # Run policy checks (defaults to safe empty structures if keys missing)
    if recipient_issue := _check_recipient_policy(metadata or {}):
        violations.append(recipient_issue)

    if content_issue := await _check_content_policy(metadata or {}):
        violations.append(content_issue)

    if attachment_issue := _check_attachment_policy(metadata or {}):
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

    safe_meta = metadata or {}

    # Check for explicit deny list
    if safe_meta.get("force_deny"):
        decision = "deny"
        reason = "Explicitly denied by policy"

    # The admin bypass has been removed to enforce a "four-eyes" principle.
    # See `escalate_for_admin_approval` for the new approval workflow.

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


def escalate_for_admin_approval(
    decision: PolicyDecision, admin_user: Dict[str, Any]
) -> PolicyDecision:
    """
    Escalates a decision for admin approval, enforcing a "four-eyes" check.

    An admin can approve an action that `require_approval`, but cannot
    override a `deny` decision. This ensures that high-risk actions receive
    a secondary review.

    Args:
        decision: The original policy decision.
        admin_user: A dictionary representing the authenticated admin user.

    Returns:
        An "allow" decision if the admin's approval is valid, otherwise the
        original decision.
    """
    if decision.decision == "require_approval":
        # Ensure the user has the 'admin' role.
        roles = admin_user.get("roles", [])
        if "admin" in roles:
            logger.info(
                "admin_approval_granted",
                action=decision.action,
                admin_user=admin_user.get("email"),
            )
            decision.decision = "allow"
            decision.reason = f"Admin approval granted by {admin_user.get('email')}"
            decision.metadata["approved_by"] = admin_user.get("email")
    return decision
