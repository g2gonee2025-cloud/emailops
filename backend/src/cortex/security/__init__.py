"""
Security module for Cortex.
"""

from cortex.security.injection_defense import contains_injection, validate_for_injection

__all__ = ["contains_injection", "validate_for_injection"]
