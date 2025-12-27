"""
Safety module for Cortex.

Provides:
- Policy enforcement (§11.2)
- Guardrails for LLM output repair (§9.3)
- Grounding verification (§9.4)
"""

from cortex.safety.grounding import (
    ClaimAnalysis,
    GroundingCheck,
    get_unsupported_claims,
    is_answer_grounded,
    tool_check_grounding,
)
from cortex.safety.policy_enforcer import PolicyDecision, check_action
from cortex.security.injection_defense import strip_injection_patterns


__all__ = [
    # Policy enforcement
    "PolicyDecision",
    "check_action",
    # Grounding
    "GroundingCheck",
    "ClaimAnalysis",
    "tool_check_grounding",
    "is_answer_grounded",
    "get_unsupported_claims",
]
