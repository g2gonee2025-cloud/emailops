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

import inspect
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import jwt
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
from cortex.context import (
    claims_ctx,
    correlation_id_ctx,
    tenant_id_ctx,
    user_id_ctx,
)
from cortex.observability import get_logger, get_trace_context, init_observability
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import FileResponse

# Version info - should match pyproject.toml
APP_VERSION = "3.1.0"
APP_NAME = "Outlook Cortex (EmailOps Edition)"
API_V1_PREFIX = "/api/v1"
INVALID_JWT_ERROR = "Invalid JWT"


# JWT/JWKS helpers
_jwt_decoder: Callable[[str], Awaitable[dict[str, Any]] | dict[str, Any]] | None = None
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
    *, jwks_url: str | None, audience: str | None, issuer: str | None
) -> None:
    global _jwt_decoder

    config = get_config()

    if jwks_url:
        _jwt_decoder = _create_jwks_decoder(jwks_url, audience, issuer)
        return

    # Fallback logic
    if config.core.env == "prod":
        _jwt_decoder = _create_prod_reject_decoder()
        return

    # Dev mode: Allow verified secret
    _jwt_decoder = _create_dev_secret_decoder(config)


def _create_jwks_decoder(
    jwks_url: str, audience: str | None, issuer: str | None
) -> Callable[[str], Awaitable[dict[str, Any]]]:
    """Create a JWKS-based JWT decoder."""
    from fastapi.concurrency import run_in_threadpool

    async def decode(token: str) -> dict[str, Any]:
        try:
            jwks = await run_in_threadpool(_load_jwks, jwks_url)
            key_data = _find_jwks_key(jwks, token)

            from jwt import algorithms

            public_key = algorithms.RSAAlgorithm.from_jwk(key_data)

            return _decode_jwt_with_jwks(token, public_key, audience, issuer)
        except jwt.PyJWTError as exc:
            raise SecurityError(INVALID_JWT_ERROR, threat_type="auth_invalid") from exc
        except SecurityError:
            raise
        except requests.RequestException as exc:
            raise SecurityError(
                "Failed to load JWKS", threat_type="auth_invalid"
            ) from exc
        except Exception as exc:
            raise SecurityError(
                "JWT validation failed", threat_type="auth_invalid"
            ) from exc

    return decode


def _decode_jwt_with_jwks(
    token: str, public_key: Any, audience: str | None, issuer: str | None
) -> dict[str, Any]:
    """Decode a JWT token using the provided JWKS public key and options."""
    decode_options = {
        "require": ["exp"],
        "verify_exp": True,
        "verify_nbf": True,
    }
    if not audience:
        decode_options["verify_aud"] = False
    if not issuer:
        decode_options["verify_iss"] = False

    return jwt.decode(
        token,
        public_key,
        algorithms=["RS256"],
        audience=audience if audience else None,
        issuer=issuer if issuer else None,
        options=decode_options,
    )


def _find_jwks_key(jwks: dict[str, Any], token: str) -> dict[str, Any]:
    """Find the matching key in JWKS for the given token."""
    headers = jwt.get_unverified_header(token)
    kid = headers.get("kid")
    for candidate in jwks.get("keys", []):
        if candidate.get("kid") == kid:
            return candidate
    raise SecurityError("JWT key id not found in JWKS", threat_type="auth_invalid")


def _create_prod_reject_decoder() -> Callable[[str], dict[str, Any]]:
    """Create a decoder that rejects all tokens in production without JWKS."""

    def reject(_: str) -> dict[str, Any]:
        raise SecurityError(
            "JWKS configuration required in production",
            threat_type="auth_configuration",
        )

    return reject


def _create_dev_secret_decoder(
    config: Any,
) -> Callable[[str], Awaitable[dict[str, Any]]]:
    """Create a dev-mode decoder using secret key."""
    secret = getattr(config, "secret_key", None)
    if not secret:
        raise ConfigurationError(
            "Missing SECRET_KEY for dev JWT decoding",
            error_code="AUTH_SECRET_MISSING",
        )

    async def decode_verified_secret(token: str) -> dict[str, Any]:
        try:
            payload = jwt.decode(
                token,
                secret,
                algorithms=["HS256"],
                options={
                    "require": ["exp"],
                    "verify_exp": True,
                    "verify_nbf": True,
                },
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
        except jwt.PyJWTError as exc:
            raise SecurityError(INVALID_JWT_ERROR, threat_type="auth_invalid") from exc

    return decode_verified_secret


async def _process_jwt_token(
    auth_header: str,
) -> tuple[str | None, str | None, dict[str, Any]]:
    """Process the JWT from the Authorization header."""
    if not auth_header.startswith("Bearer "):
        return None, None, {}

    if not _jwt_decoder:
        raise SecurityError(
            "JWT decoder not configured", threat_type="auth_configuration"
        )
    token = auth_header[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    try:
        decoded = _jwt_decoder(token)
        # Handle both sync and async decoders
        if inspect.isawaitable(decoded):
            claims = await decoded
        else:
            claims = decoded if isinstance(decoded, dict) else {}

        # Extract standard claims
        tenant_id = claims.get("tenant_id") or claims.get("tid")
        user_id = claims.get("sub") or claims.get("user_id")
        return tenant_id, user_id, claims
    except (HTTPException, SecurityError):
        raise
    except Exception as exc:
        logger.exception("JWT decode failed")
        raise SecurityError(INVALID_JWT_ERROR, threat_type="auth_invalid") from exc


def _process_fallback_headers(
    request: Request, tenant_id: str, user_id: str
) -> tuple[str, str]:
    """Extract identity from headers in dev mode if not already set."""
    config = get_config()
    is_prod_env = config.core.env in {"prod", "production"}

    if is_prod_env:
        return tenant_id, user_id

    final_tenant_id = tenant_id
    final_user_id = user_id

    if tenant_id == "default":
        header_tenant = request.headers.get("X-Tenant-ID", "").strip()
        if header_tenant:
            final_tenant_id = header_tenant
    if user_id == "anonymous":
        header_user = request.headers.get("X-User-ID", "").strip()
        if header_user:
            final_user_id = header_user
    return final_tenant_id, final_user_id


async def _extract_identity(request: Request) -> tuple[str, str, dict[str, Any]]:
    """
    Extract tenant_id, user_id, and claims from request.

    Priority:
    1. JWT token in Authorization header (if decoder configured)
    2. X-Tenant-ID and X-User-ID headers (dev fallback)
    3. Default values
    """
    tenant_id = "default"
    user_id = "anonymous"
    claims: dict[str, Any] = {}

    auth_header = request.headers.get("Authorization", "")
    auth_attempted = auth_header.startswith("Bearer ")

    if auth_attempted:
        jwt_tenant_id, jwt_user_id, claims = await _process_jwt_token(auth_header)
        tenant_id = jwt_tenant_id or tenant_id
        user_id = jwt_user_id or user_id
    else:
        # Fallback to headers (dev mode)
        tenant_id, user_id = _process_fallback_headers(request, tenant_id, user_id)

    return tenant_id, user_id, claims


logger = get_logger(__name__)
_LOG_LEVELS = {
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def _log_event(level: str, event: str, data: dict[str, Any]) -> None:
    if isinstance(logger, logging.Logger):
        logger.log(_LOG_LEVELS[level], event, extra={"event": event, **data})
    else:
        log_fn = getattr(logger, level)
        log_fn(event, **data)


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
        # Expose on request.state so downstream routes can access without
        # coupling to contextvars.
        request.state.correlation_id = correlation_id

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
        tenant_id, user_id, claims = await _extract_identity(request)

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
                _log_event("error", "request_completed", log_entry)
            elif response.status_code >= 400:
                _log_event("warning", "request_completed", log_entry)
            else:
                _log_event("info", "request_completed", log_entry)

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_entry = {
                "method": request.method,
                "path": request.url.path,
                "error_type": type(e).__name__,
                "duration_ms": round(duration_ms, 2),
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "trace_id": trace_ctx.get("trace_id"),
                "span_id": trace_ctx.get("span_id"),
            }
            _log_event("error", "request_failed", log_entry)
            raise


# ---------------------------------------------------------------------------
# Exception Handler: CortexError → HTTP Response (§11.4)
# ---------------------------------------------------------------------------
def create_error_response(
    status_code: int,
    error_type: str,
    message: str,
    error_code: str | None = None,
    context: dict[str, Any] | None = None,
    correlation_id: str | None = None,
) -> JSONResponse:
    """Create a structured error response with correlation ID."""
    correlation_id = correlation_id or correlation_id_ctx.get("unknown")

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
        sensitive_markers = (
            "password",
            "api_key",
            "secret",
            "token",
            "credential",
            "authorization",
            "refresh_token",
            "access_token",
            "id_token",
        )
        safe_context = {
            k: v
            for k, v in context.items()
            if not (
                isinstance(k, str)
                and any(marker in k.lower() for marker in sensitive_markers)
            )
        }
        if safe_context:
            body["error"]["context"] = safe_context

    return JSONResponse(status_code=status_code, content=body)


def cortex_error_handler(request: Request, exc: Exception) -> JSONResponse:
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
        return generic_exception_handler(request, exc)

    status_code = 500

    if isinstance(exc, ValidationError):
        status_code = 400
    elif isinstance(exc, SecurityError):
        status_code = 403
    elif isinstance(exc, ProviderError) or isinstance(exc, EmbeddingError):
        status_code = 503 if exc.retryable else 502
    elif isinstance(exc, ConfigurationError):
        status_code = 500

    return create_error_response(
        status_code=status_code,
        error_type=type(exc).__name__,
        message=getattr(exc, "message", str(exc)),
        error_code=getattr(exc, "error_code", None),
        context=exc.context,
        correlation_id=getattr(request.state, "correlation_id", None),
    )


def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handler for unexpected exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    correlation_id = getattr(request.state, "correlation_id", None)
    return create_error_response(
        status_code=500,
        error_type="InternalServerError",
        message="An unexpected error occurred. Please contact support.",
        error_code="INTERNAL_ERROR",
        correlation_id=correlation_id,
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
        logger.warning("Failed to setup OpenTelemetry instrumentation: %s", e)


# ---------------------------------------------------------------------------
# OIDC/JWT Security Integration Stub (§11.1)
# ---------------------------------------------------------------------------
def setup_security() -> None:
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
    global _jwt_decoder
    try:
        config = get_config()
        is_prod_env = config.core.env in {"prod", "production"}
        jwks_url = os.getenv("OUTLOOKCORTEX_OIDC_JWKS_URL") or os.getenv(
            "OIDC_JWKS_URL"
        )
        audience = os.getenv("OUTLOOKCORTEX_OIDC_AUDIENCE") or os.getenv(
            "OIDC_AUDIENCE"
        )
        issuer = os.getenv("OUTLOOKCORTEX_OIDC_ISSUER") or os.getenv("OIDC_ISSUER")

        _configure_jwt_decoder(jwks_url=jwks_url, audience=audience, issuer=issuer)

        if jwks_url:
            if jwks_url.startswith("http://"):
                logger.warning(
                    "SECURITY: JWKS URL is using insecure HTTP protocol: %s", jwks_url
                )
            elif not jwks_url.startswith("https://"):
                logger.warning(
                    "SECURITY: JWKS URL does not use https protocol: %s", jwks_url
                )

        if jwks_url:
            logger.info("Security: JWT validation enabled via JWKS %s", jwks_url)
        elif is_prod_env:
            logger.warning(
                "Security: prod environment without JWKS configuration; bearer tokens will be rejected"
            )
        else:
            logger.info("Security: dev mode with header-based identity fallback")
    except Exception:
        logger.warning("Failed to setup security")
        logger.debug("Security setup exception", exc_info=True)
        env = os.getenv("ENV", "production")
        if env in {"prod", "production"}:
            _jwt_decoder = _create_prod_reject_decoder()


# ---------------------------------------------------------------------------
# Lifespan Manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager.

    Blueprint §12.3:
    - Initialize observability on startup
    - Pre-compile LangGraph instances for thread-safety (P0 fix)
    - Clean shutdown
    """
    config = get_config()
    import redis.asyncio as redis

    # On startup: Create Redis client and store in app state
    app.state.redis = redis.from_url(config.redis.url, decode_responses=True)

    # Initialize observability stack
    init_observability(
        service_name="cortex-backend",
        enable_tracing=True,
        enable_metrics=True,
        enable_structured_logging=True,
    )

    logger.info(f"Starting {APP_NAME} v{APP_VERSION} ({config.core.env})")

    # P0 Fix: Pre-compile LangGraph instances for thread-safety
    # Prevents race conditions from concurrent lazy compilation
    try:
        from cortex.orchestration.graphs import (
            build_answer_graph,
            build_draft_graph,
            build_summarize_graph,
        )

        logger.info("Pre-compiling LangGraph instances...")
        compiled_graphs = {
            "answer": build_answer_graph().compile(),
            "draft": build_draft_graph().compile(),
            "summarize": build_summarize_graph().compile(),
        }
        app.state.graphs = compiled_graphs
        logger.info("✓ LangGraph instances compiled and cached")
    except Exception as e:
        logger.warning(f"Failed to pre-compile graphs (will lazy-init): {e}")
        app.state.graphs = {}

    yield

    # On shutdown: Gracefully close the Redis client
    redis_client = app.state.redis
    if hasattr(redis_client, "aclose"):
        await redis_client.aclose()
    else:
        close_result = redis_client.close()
        if inspect.isawaitable(close_result):
            await close_result

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
        [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
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
    setup_security()

    # ---------------------------------------------------------------------------
    # Include API Routers
    # ---------------------------------------------------------------------------
    from cortex import routes_admin, routes_auth
    from cortex.rag_api import (
        routes_answer,
        routes_chat,
        routes_draft,
        routes_ingest,
        routes_search,
        routes_summarize,
    )

    # Conditionally include the mock auth router only in dev environments
    if config.core.env == "dev":
        app.include_router(routes_auth.router, prefix=API_V1_PREFIX)

    app.include_router(routes_admin.router, prefix=API_V1_PREFIX)
    app.include_router(routes_search.router, prefix=API_V1_PREFIX, tags=["search"])
    app.include_router(routes_answer.router, prefix=API_V1_PREFIX, tags=["answer"])
    app.include_router(routes_draft.router, prefix=API_V1_PREFIX, tags=["draft"])
    app.include_router(routes_summarize.router, prefix=API_V1_PREFIX, tags=["summarize"])
    app.include_router(routes_chat.router, prefix=API_V1_PREFIX, tags=["chat"])
    app.include_router(routes_ingest.router, prefix=API_V1_PREFIX, tags=["ingestion"])

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
# Static File Serving (SPA Support)
# ---------------------------------------------------------------------------
# Mount static files *after* creating app but before returning if we were doing it inside factory.
# However, since app is created via factory and we need to mount on the instance,
# we can do it here or inside factory.
# Doing it outside allows us to use 'app' generated by factory.

# Actually, let's move this INSIDE create_app for better encapsulation,
# but we need to verify frontend_dist path.


def _mount_static_files(app: FastAPI):
    import os

    project_root = os.getcwd()
    # Support both flat and nested structures (workspace vs docker)
    potential_paths = [
        os.path.join(project_root, "frontend/dist"),
        os.path.join(project_root, "../frontend/dist"),
    ]

    frontend_dist = next((p for p in potential_paths if os.path.isdir(p)), None)

    if not frontend_dist:
        logger.warning(
            f"Frontend dist not found in {potential_paths}. SPA will not be served."
        )
        return

    logger.info(f"Serving frontend from {frontend_dist}")

    # 1. Mount assets
    assets_path = os.path.join(frontend_dist, "assets")
    if os.path.isdir(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

    # 2. Serve static root files (favicon, etc.)
    # We can handle specific files or just let the catch-all handle them if we are careful.

    favicon_path = os.path.join(frontend_dist, "favicon.ico")
    if os.path.isfile(favicon_path):

        @app.get("/favicon.ico", include_in_schema=False)
        async def favicon():
            return FileResponse(favicon_path)

    # 3. Catch-all for SPA
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        # Allow API routes to bubble up 404 if not matched (handled by FastAPI router order? No, this is last)
        # Actually, FastAPI matches routes in order. The API routers are included earlier.
        # So this catch-all will only fire if no other route matches.

        # We should NOT serve index.html for api requests that 404
        if (
            full_path.startswith("api/")
            or full_path.startswith("docs")
            or full_path.startswith("redoc")
        ):
            raise HTTPException(status_code=404, detail="Not Found")

        # Check if the file exists in root (e.g. vite.svg)
        file_path = os.path.join(frontend_dist, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)

        # Fallback to index.html
        return FileResponse(os.path.join(frontend_dist, "index.html"))


_mount_static_files(app)


# Initialize Prometheus instrumentation
try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app)
    logger.info("✓ Prometheus metrics exposed at /metrics")
except ImportError:
    logger.warning("prometheus-fastapi-instrumentator not found; metrics disabled")

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
