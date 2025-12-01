"""
Safety module for Cortex.

Provides:
- Injection defense (§11.5)
- Policy enforcement (§11.2)
- Guardrails for LLM output repair (§9.3)
- Grounding verification (§9.4)
"""
from cortex.safety.injection_defense import strip_injection_patterns
from cortex.safety.policy_enforcer import PolicyDecision, check_action
from cortex.safety.grounding import (
    GroundingCheck,
    ClaimAnalysis,
    tool_check_grounding,
    is_answer_grounded,
    get_unsupported_claims,
)

__all__ = [
    # Injection defense
    "strip_injection_patterns",
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