"""
Security module for Cortex.
"""

from cortex.security.injection_defense import strip_injection_patterns

__all__ = ["strip_injection_patterns"]
