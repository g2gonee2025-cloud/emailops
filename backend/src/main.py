"""
FastAPI Entry Point.

Implements §2.2, §11, §12 of the Canonical Blueprint.

This is the canonical location for the FastAPI entry point as per §2.2.
Provides:
- Correlation ID middleware
- Tenant/user extraction middleware
- Structured JSON logging middleware
- Global CortexError exception handling
- /health and /version endpoints
- OpenTelemetry instrumentation
- OIDC/JWT security integration (§11.1)
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, Optional

import requests
from cortex.common.exceptions import (
    ConfigurationError,
    CortexError,
    EmbeddingError,
    ProviderError,
    SecurityError,
    ValidationError,
)
from cortex.config.loader import get_config
from cortex.observability import get_trace_context, init_observability
from cortex.security.validators import validate_email_format
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

# Version info - should match pyproject.toml
APP_VERSION = "3.1.0"
APP_NAME = "Outlook Cortex (EmailOps Edition)"

# Context variables for request-scoped data
correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id", default="")
tenant_id_ctx: ContextVar[str] = ContextVar("tenant_id", default="")
user_id_ctx: ContextVar[str] = ContextVar("user_id", default="")
claims_ctx: ContextVar[dict[str, Any]] = ContextVar("claims", default={})

# JWT/JWKS helpers
_jwt_decoder: Optional[callable] = None
_jwks_cache: dict[str, Any] = {}


def _load_jwks(jwks_url: str) -> dict[str, Any]:
    if jwks_url in _jwks_cache:
        return _jwks_cache[jwks_url]
    response = requests.get(jwks_url, timeout=5)
    response.raise_for_status()
    data = response.json()
    _jwks_cache[jwks_url] = data
    return data


def _configure_jwt_decoder(
    *, jwks_url: Optional[str], audience: Optional[str], issuer: Optional[str]
) -> None:
    global _jwt_decoder

    if jwks_url:

        def decode(token: str) -> dict[str, Any]:
            try:
                jwks = _load_jwks(jwks_url)
                headers = jwt.get_unverified_header(token)
                kid = headers.get("kid")
                key = None
                for candidate in jwks.get("keys", []):
                    if candidate.get("kid") == kid:
                        key = candidate
                        break
                if key is None:
                    raise SecurityError(
                        "JWT key id not found in JWKS", threat_type="auth_invalid"
                    )

                return jwt.decode(
                    token,
                    key,
                    audience=audience,
                    issuer=issuer,
                    options={
                        "verify_aud": bool(audience),
                        "verify_iss": bool(issuer),
                    },
                )
            except JWTError as exc:
                raise SecurityError("Invalid JWT", threat_type="auth_invalid") from exc
            except Exception as exc:  # HTTP errors, json, etc.
                raise SecurityError(
                    "Failed to load JWKS", threat_type="auth_invalid"
                ) from exc

        _jwt_decoder = decode
        return

    def decode_unverified(token: str) -> dict[str, Any]:
        try:
            return jwt.get_unverified_claims(token)
        except JWTError as exc:
            raise SecurityError("Invalid JWT", threat_type="auth_invalid") from exc

    _jwt_decoder = decode_unverified


def _extract_identity(request: Request) -> tuple[str, str, dict[str, Any]]:
    """Resolve tenant/user identity from JWT or headers."""
    config = get_config()
    auth_header = request.headers.get("Authorization", "")
    claims: dict[str, Any] = {}

    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        if not token:
            raise SecurityError("Bearer token missing", threat_type="auth_missing")
        if _jwt_decoder is None:
            raise SecurityError(
                "JWT decoder not configured", threat_type="auth_invalid"
            )
        claims = _jwt_decoder(token)
    elif config.core.env == "prod":
        raise SecurityError("Authorization header required", threat_type="auth_missing")

    tenant_id = (
        claims.get("tenant_id")
        or claims.get("tid")
        or request.headers.get("X-Tenant-ID")
        or "default"
    )
    user_id = (
        claims.get("sub")
        or claims.get("email")
        or request.headers.get("X-User-ID")
        or "anonymous"
    )

    if isinstance(user_id, str):
        result = validate_email_format(user_id)
        if not result.ok:
            claims["email_invalid"] = user_id
        else:
            user_id = result.value or user_id

    return str(tenant_id), str(user_id), claims


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Middleware: Correlation ID (§12.1)
# ---------------------------------------------------------------------------
class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware to extract or generate a correlation ID for request tracing.

    Blueprint §12.1:
    - Each request gets a correlation_id for log/trace correlation
    - Accepts X-Correlation-ID header or generates a new UUID
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Extract or generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID")
        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        # Store in context var for access throughout request
        token = correlation_id_ctx.set(correlation_id)

        try:
            response = await call_next(request)
            # Echo correlation ID in response
            response.headers["X-Correlation-ID"] = correlation_id
            return response
        finally:
            correlation_id_ctx.reset(token)


# ---------------------------------------------------------------------------
# Middleware: Tenant/User Extraction (§11.1)
# ---------------------------------------------------------------------------
class TenantUserMiddleware(BaseHTTPMiddleware):
    """
    Middleware to extract tenant_id and user_id from request.

    Blueprint §11.1:
    - Identity via OIDC / JWT (e.g. Keycloak)
    - Each request has tenant_id, user_id, roles/claims
    - Sets Postgres RLS context via cortex.tenant_id

    For now, extracts from headers. Full OIDC/JWT validation
    should be implemented in production.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        tenant_id, user_id, claims = _extract_identity(request)

        tenant_token = tenant_id_ctx.set(tenant_id)
        user_token = user_id_ctx.set(user_id)
        claims_token = claims_ctx.set(claims)

        try:
            response = await call_next(request)
            return response
        finally:
            tenant_id_ctx.reset(tenant_token)
            user_id_ctx.reset(user_token)
            claims_ctx.reset(claims_token)


# ---------------------------------------------------------------------------
# Middleware: Structured JSON Logging (§12.1)
# ---------------------------------------------------------------------------
class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for structured JSON logging of requests.

    Blueprint §12.1:
    - Structured JSON logs with fields:
      timestamp, level, logger, message,
      tenant_id, user_id, correlation_id, trace_id, span_id
    - Never log secrets, full email bodies, or raw PII
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.time()

        # Get context values
        correlation_id = correlation_id_ctx.get()
        tenant_id = tenant_id_ctx.get()
        user_id = user_id_ctx.get()
        trace_ctx = get_trace_context()

        try:
            response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000

            # Log successful request
            log_entry = {
                "event": "request_completed",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "trace_id": trace_ctx.get("trace_id"),
                "span_id": trace_ctx.get("span_id"),
            }

            # Log level based on status code
            if response.status_code >= 500:
                logger.error(json.dumps(log_entry))
            elif response.status_code >= 400:
                logger.warning(json.dumps(log_entry))
            else:
                logger.info(json.dumps(log_entry))

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_entry = {
                "event": "request_failed",
                "method": request.method,
                "path": request.url.path,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "duration_ms": round(duration_ms, 2),
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "trace_id": trace_ctx.get("trace_id"),
                "span_id": trace_ctx.get("span_id"),
            }
            logger.error(json.dumps(log_entry))
            raise


# ---------------------------------------------------------------------------
# Exception Handler: CortexError → HTTP Response (§11.4)
# ---------------------------------------------------------------------------
def create_error_response(
    status_code: int,
    error_type: str,
    message: str,
    error_code: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
) -> JSONResponse:
    """Create a structured error response with correlation ID."""
    correlation_id = correlation_id_ctx.get()

    body = {
        "error": {
            "type": error_type,
            "message": message,
            "error_code": error_code,
            "correlation_id": correlation_id,
        }
    }

    # Only include non-sensitive context
    if context:
        safe_context = {
            k: v
            for k, v in context.items()
            if k not in ("password", "api_key", "secret", "token", "credential")
        }
        if safe_context:
            body["error"]["context"] = safe_context

    return JSONResponse(status_code=status_code, content=body)


async def cortex_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler for CortexError hierarchy.

    Blueprint §11.4:
    - ConfigurationError → 500
    - ValidationError → 400
    - SecurityError → 403
    - ProviderError(retryable=False) → 502/503
    - Always include correlation_id
    """
    if not isinstance(exc, CortexError):
        # Shouldn't happen, but fallback
        return await generic_exception_handler(request, exc)

    status_code = 500

    if isinstance(exc, ValidationError):
        status_code = 400
    elif isinstance(exc, SecurityError):
        status_code = 403
    elif isinstance(exc, ProviderError):
        status_code = 503 if exc.retryable else 502
    elif isinstance(exc, EmbeddingError):
        status_code = 503 if exc.retryable else 502
    elif isinstance(exc, ConfigurationError):
        status_code = 500

    return create_error_response(
        status_code=status_code,
        error_type=type(exc).__name__,
        message=exc.message,
        error_code=exc.error_code,
        context=exc.context,
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handler for unexpected exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return create_error_response(
        status_code=500,
        error_type="InternalServerError",
        message="An unexpected error occurred. Please contact support.",
        error_code="INTERNAL_ERROR",
    )


# ---------------------------------------------------------------------------
# OpenTelemetry FastAPI Instrumentation (§12.2, §12.3)
# ---------------------------------------------------------------------------
def setup_opentelemetry(app: FastAPI) -> None:
    """
    Configure OpenTelemetry instrumentation for FastAPI.

    Blueprint §12.2:
    - OpenTelemetry traces for HTTP requests
    - Prometheus metrics for request counts/latency

    Gracefully degrades if dependencies not available.
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        logger.info("✓ OpenTelemetry FastAPI instrumentation enabled")
    except ImportError:
        logger.warning("OpenTelemetry FastAPI instrumentor not available; skipping")
    except Exception as e:
        logger.warning(f"Failed to setup OpenTelemetry instrumentation: {e}")


# ---------------------------------------------------------------------------
# OIDC/JWT Security Integration Stub (§11.1)
# ---------------------------------------------------------------------------
def setup_security(app: FastAPI) -> None:
    """
    Configure OIDC/JWT security integration.

    Blueprint §11.1:
    - Identity via OIDC / JWT (e.g. Keycloak)
    - Extract tenant_id, user_id, roles/claims from JWT

    This is a stub implementation. Production deployments should:
    1. Configure OIDC provider (Keycloak, Auth0, etc.)
    2. Validate JWT tokens on protected routes
    3. Extract claims and populate context vars
    """
    try:
        config = get_config()
        jwks_url = os.getenv("OUTLOOKCORTEX_OIDC_JWKS_URL") or os.getenv(
            "OIDC_JWKS_URL"
        )
        audience = os.getenv("OUTLOOKCORTEX_OIDC_AUDIENCE") or os.getenv(
            "OIDC_AUDIENCE"
        )
        issuer = os.getenv("OUTLOOKCORTEX_OIDC_ISSUER") or os.getenv("OIDC_ISSUER")

        _configure_jwt_decoder(jwks_url=jwks_url, audience=audience, issuer=issuer)

        if jwks_url:
            logger.info("Security: JWT validation enabled via JWKS %s", jwks_url)
        elif config.core.env == "prod":
            logger.warning(
                "Security: prod environment without JWKS configuration; JWT validation will be header-only"
            )
        else:
            logger.info("Security: dev mode with header-based identity fallback")
    except Exception as e:
        logger.warning(f"Failed to setup security: {e}")


# ---------------------------------------------------------------------------
# Lifespan Manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager.

    Blueprint §12.3:
    - Initialize observability on startup
    - Clean shutdown
    """
    config = get_config()

    # Initialize observability stack
    init_observability(
        service_name="cortex-backend",
        enable_tracing=True,
        enable_metrics=True,
        enable_structured_logging=True,
    )

    logger.info(f"Starting {APP_NAME} v{APP_VERSION} ({config.core.env})")

    yield

    logger.info(f"Shutting down {APP_NAME}")


# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    config = get_config()

    # ---------------------------------------------------------------------------
    # Add Middleware (order matters - executed in reverse order)
    # ---------------------------------------------------------------------------

    # 1. CORS - restrict in production
    allowed_origins = (
        ["*"]
        if config.core.env == "dev"
        else [
            "https://app.emailops.io",
            "https://admin.emailops.io",
        ]
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    # 2. Structured logging middleware
    app.add_middleware(StructuredLoggingMiddleware)

    # 3. Tenant/User extraction middleware
    app.add_middleware(TenantUserMiddleware)

    # 4. Correlation ID middleware (must be outermost to set context first)
    app.add_middleware(CorrelationIdMiddleware)

    # ---------------------------------------------------------------------------
    # Exception Handlers (§11.4)
    # ---------------------------------------------------------------------------
    app.add_exception_handler(CortexError, cortex_error_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # ---------------------------------------------------------------------------
    # OpenTelemetry Instrumentation (§12.2)
    # ---------------------------------------------------------------------------
    setup_opentelemetry(app)

    # ---------------------------------------------------------------------------
    # Security Setup (§11.1)
    # ---------------------------------------------------------------------------
    setup_security(app)

    # ---------------------------------------------------------------------------
    # Include API Routers
    # ---------------------------------------------------------------------------
    from cortex.rag_api import (
        routes_answer,
        routes_draft,
        routes_ingest,
        routes_search,
        routes_summarize,
    )

    app.include_router(routes_search.router, prefix="/api/v1", tags=["search"])
    app.include_router(routes_answer.router, prefix="/api/v1", tags=["answer"])
    app.include_router(routes_draft.router, prefix="/api/v1", tags=["draft"])
    app.include_router(routes_summarize.router, prefix="/api/v1", tags=["summarize"])
    app.include_router(routes_ingest.router, prefix="/api/v1", tags=["ingestion"])

    # ---------------------------------------------------------------------------
    # Core Endpoints: /health and /version
    # ---------------------------------------------------------------------------
    @app.get("/health", tags=["system"])
    async def health_check() -> dict[str, Any]:
        """
        Health check endpoint.

        Returns basic health status. In production, this should check:
        - Database connectivity
        - Redis/queue availability
        - External service health
        """
        return {
            "status": "healthy",
            "version": APP_VERSION,
            "environment": config.core.env,
        }

    @app.get("/version", tags=["system"])
    async def version_info() -> dict[str, Any]:
        """
        Version information endpoint.

        Returns application version and build information.
        """
        return {
            "name": APP_NAME,
            "version": APP_VERSION,
            "api_version": "v1",
            "environment": config.core.env,
        }

    return app


# Create the application instance
app = create_app()


# ---------------------------------------------------------------------------
# For running directly with uvicorn
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
