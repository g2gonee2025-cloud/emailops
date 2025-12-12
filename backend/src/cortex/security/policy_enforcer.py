"""
Policy Enforcer Module.

Enforces business and safety policies on inputs and outputs under the Cortex Governance model.
"""
from typing import Any, Dict, Optional

import structlog
from cortex.config.models import SecurityConfig

logger = structlog.get_logger(__name__)


class PolicyEnforcer:
    """
    Enforces content safety and business policies.
    """

    def __init__(self, config: Optional[SecurityConfig] = None):
        """
        Initialize policy enforcer.

        Args:
            config: Optional configuration object for safety settings.
        """
        self.config = config or SecurityConfig()
        # Default PII terms to block if not configured
        self.blocked_terms = ["password", "secret key", "api key", "access token"]

    def check_content(self, text: str) -> bool:
        """
        Check if content complies with policies.

        Args:
            text: Content to check.

        Returns:
            True if compliant, False if policy violation derived.
        """
        if not text:
            return True

        text_lower = text.lower()

        # Simple keyword blocking for demonstrating policy enforcement
        for term in self.blocked_terms:
            if term in text_lower:
                logger.warning("policy_violation_detected", term=term)
                return False

        return True

    def enforce_input_policy(self, session_context: Dict[str, Any]) -> bool:
        """
        Validate input session context against policies (e.g. auth presence, tenant boundaries).

        Args:
            session_context: Dictionary containing request metadata.

        Returns:
            True if allowed, False otherwise.
        """
        # Example policy: Tenant ID must be present
        if "tenant_id" not in session_context:
            logger.error("policy_enforcement_failed", reason="missing_tenant_id")
            return False

        return True
