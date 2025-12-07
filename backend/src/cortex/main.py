"""
FastAPI Entry Point - Backwards Compatibility Re-export.

DEPRECATED: This module exists for backwards compatibility only.
The canonical location is now backend/src/main.py per §2.2 of the Blueprint.

New code should import from the canonical location:
    from main import app, create_app, APP_VERSION

This module re-exports the app from the canonical location.
"""
from __future__ import annotations

import warnings

# Issue deprecation warning
warnings.warn(
    "cortex.main is deprecated. Import from 'main' instead (backend/src/main.py). "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location
try:
    from main import (
        APP_NAME,
        APP_VERSION,
        CorrelationIdMiddleware,
        StructuredLoggingMiddleware,
        TenantUserMiddleware,
        app,
        correlation_id_ctx,
        cortex_error_handler,
        create_app,
        create_error_response,
        generic_exception_handler,
        lifespan,
        setup_opentelemetry,
        setup_security,
        tenant_id_ctx,
        user_id_ctx,
    )

    __all__ = [
        "app",
        "create_app",
        "APP_NAME",
        "APP_VERSION",
        "CorrelationIdMiddleware",
        "TenantUserMiddleware",
        "StructuredLoggingMiddleware",
        "correlation_id_ctx",
        "tenant_id_ctx",
        "user_id_ctx",
        "cortex_error_handler",
        "generic_exception_handler",
        "create_error_response",
        "lifespan",
        "setup_opentelemetry",
        "setup_security",
    ]
except ImportError:
    # Fallback for when running from old location
    # This maintains compatibility during migration
    import logging

    logging.warning("Could not import from canonical main.py location")
