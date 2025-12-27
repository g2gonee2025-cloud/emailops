"""
Safety module for Cortex.

Provides:
- Injection defense (§11.5)
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

try:
    from cortex.security.injection_defense import InjectionDefense
except Exception:
    InjectionDefense = None


def strip_injection_patterns(text, defense=None):
    """
    Backwards-compatible wrapper around InjectionDefense.
    If the underlying implementation is unavailable, returns text unchanged.
    """
    if defense is None:
        if InjectionDefense is None:
            return text
        try:
            defense = InjectionDefense()
        except Exception:
            return text
    for attr in (
        "strip_injection_patterns",
        "strip",
        "sanitize",
        "clean",
        "remove_patterns",
    ):
        method = getattr(defense, attr, None)
        if callable(method):
            try:
                return method(text)
            except Exception:
                break
    return text


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
