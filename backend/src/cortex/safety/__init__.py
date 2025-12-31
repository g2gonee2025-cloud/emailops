"""
Safety module for Cortex.

Provides:
- Policy enforcement (§11.2)
- Guardrails for LLM output repair (§9.3)
- Grounding verification (§9.4)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = [
    # Injection defense
    "contains_injection",
    "validate_for_injection",
    # Policy enforcement
    "PolicyDecision",
    "check_action",
    # Guardrails
    "attempt_llm_repair",
    "get_guardrails_client",
    "validate_with_repair",
    # Grounding
    "GroundingCheck",
    "ClaimAnalysis",
    "tool_check_grounding",
    "is_answer_grounded",
    "get_unsupported_claims",
]

_LAZY_IMPORTS = {
    # Injection defense
    "contains_injection": "cortex.security.injection_defense",
    "validate_for_injection": "cortex.security.injection_defense",
    # Policy enforcement
    "PolicyDecision": "cortex.safety.policy_enforcer",
    "check_action": "cortex.safety.policy_enforcer",
    # Guardrails
    "attempt_llm_repair": "cortex.safety.guardrails_client",
    "get_guardrails_client": "cortex.safety.guardrails_client",
    "validate_with_repair": "cortex.safety.guardrails_client",
    # Grounding
    "ClaimAnalysis": "cortex.safety.grounding",
    "GroundingCheck": "cortex.safety.grounding",
    "get_unsupported_claims": "cortex.safety.grounding",
    "is_answer_grounded": "cortex.safety.grounding",
    "tool_check_grounding": "cortex.safety.grounding",
}

if TYPE_CHECKING:
    from cortex.safety.grounding import (
        ClaimAnalysis,
        GroundingCheck,
        get_unsupported_claims,
        is_answer_grounded,
        tool_check_grounding,
    )
    from cortex.safety.guardrails_client import (
        attempt_llm_repair,
        get_guardrails_client,
        validate_with_repair,
    )
    from cortex.safety.policy_enforcer import PolicyDecision, check_action
    from cortex.security.injection_defense import (
        contains_injection,
        validate_for_injection,
    )


def __getattr__(name: str):
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Optional safety dependency for {name!r} is not available."
        ) from exc
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(__all__))
