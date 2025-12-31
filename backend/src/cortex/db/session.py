"""
Database session management.

Implements §4.2 and §11.1 of the Canonical Blueprint.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

from cortex.common.exceptions import TransactionError
from cortex.config.loader import get_config
from cortex.observability import trace_operation
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import Session, sessionmaker

# Constants
SLOW_QUERY_THRESHOLD_SECONDS = 1.0
HASH_PREFIX_LEN = 8


logger = logging.getLogger(__name__)

# Security hardening: avoid logging or propagating raw exception messages
# which may reveal sensitive information. We attach a filter to this module's
# logger that redacts exception messages from log output originating here.


class SafeDatabaseError(RuntimeError):
    pass


class _RedactingExceptionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Scrub exception info so that exception messages are not emitted.
        if record.exc_info:
            exc_type, _exc_value, tb = record.exc_info  # type: ignore[assignment]
            # Replace the exception value with a new instance with no args/message.
            try:
                sanitized_exc = exc_type()
                sanitized_type = exc_type
            except Exception:
                sanitized_exc = SafeDatabaseError()
                sanitized_type = SafeDatabaseError
            record.exc_info = (
                sanitized_type,
                sanitized_exc,
                tb,
            )  # type: ignore[assignment]
        # Scrub any exception instances passed as formatting args.
        if record.args:
            try:
                if isinstance(record.args, tuple):
                    record.args = tuple(
                        "<redacted>" if isinstance(a, BaseException) else a
                        for a in record.args
                    )
                else:
                    record.args = (
                        "<redacted>"
                        if isinstance(record.args, BaseException)
                        else record.args
                    )
            except Exception:
                # On any failure, do not block logging.
                pass
        return True


# Ensure the filter is attached only once.
if not any(isinstance(f, _RedactingExceptionFilter) for f in logger.filters):
    logger.addFilter(_RedactingExceptionFilter())


def raise_sanitized(
    error_message: str = "Database operation failed",
    *,
    cause: BaseException | None = None,
) -> None:
    if cause is None:
        raise SafeDatabaseError(error_message)
    raise SafeDatabaseError(error_message) from cause


def _hash_text(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return digest[:HASH_PREFIX_LEN]


def _load_config():
    try:
        return get_config()
    except Exception as exc:
        logger.error("Database configuration unavailable", exc_info=True)
        raise SafeDatabaseError("Database configuration unavailable") from exc


_config = _load_config()
_db_config = getattr(_config, "database", None)
if _db_config is None:
    raise SafeDatabaseError("Database configuration unavailable")

_db_url = getattr(_db_config, "url", None)
if not _db_url:
    raise SafeDatabaseError("Database configuration missing database.url")

_db_url_value = str(_db_url)
engine_args = {
    "pool_pre_ping": True,
}
db_backend = make_url(_db_url_value).get_backend_name()
if db_backend != "sqlite":
    pool_size = getattr(_db_config, "pool_size", None)
    max_overflow = getattr(_db_config, "max_overflow", None)
    if pool_size is not None:
        engine_args["pool_size"] = pool_size
    if max_overflow is not None:
        engine_args["max_overflow"] = max_overflow

engine = create_engine(
    _db_url_value,
    **engine_args,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for getting a database session.
    Yields a SQLAlchemy Session and ensures it's closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session(tenant_id: str | None = None) -> Generator[Session, None, None]:
    """
    Context manager for database sessions with optional RLS tenant context.

    Blueprint §11.1:
    - Sets Postgres RLS context via SET app.current_tenant

    Args:
        tenant_id: Optional tenant ID for RLS isolation

    Yields:
        SQLAlchemy Session

    Raises:
        TransactionError: On commit/rollback failures
    """
    session = SessionLocal()
    try:
        # Set RLS tenant context if provided
        if tenant_id:
            set_session_tenant(session, tenant_id)

        yield session

        # Commit on success
        session.commit()

    except Exception as e:
        # Rollback on any error
        try:
            session.rollback()
        except Exception:
            logger.error("Rollback failed", exc_info=True)

        # Wrap in TransactionError for non-CortexErrors
        if not isinstance(e, TransactionError):
            raise TransactionError(
                message="Database transaction failed",
                error_code="TRANSACTION_FAILED",
                context={"operation": "get_db_session"},
            ) from e
        raise

    finally:
        session.close()


def set_session_tenant(session: Session, tenant_id: str) -> None:
    """
    Set the RLS tenant context for a session.

    Blueprint §11.1:
    - Postgres RLS enforces tenant isolation
    - SET app.current_tenant = :tid

    Args:
        session: SQLAlchemy session
        tenant_id: Tenant ID to set

    Raises:
        TransactionError: If tenant_id is invalid
    """
    if not tenant_id:
        raise TransactionError(
            message="tenant_id is required for RLS",
            error_code="RLS_TENANT_REQUIRED",
        )

    # Validate tenant_id format to prevent SQL injection
    if not re.match(r"^[a-zA-Z0-9_-]+$", tenant_id):
        raise TransactionError(
            message="Invalid tenant_id format",
            error_code="RLS_TENANT_INVALID",
            context={"tenant_hash": _hash_text(tenant_id)},
        )

    try:
        session.execute(
            text("SET app.current_tenant = :tenant_id"), {"tenant_id": tenant_id}
        )
        logger.debug("Set RLS tenant context: %s", _hash_text(tenant_id))
    except Exception as e:
        raise TransactionError(
            message="Failed to set RLS tenant",
            error_code="RLS_SET_FAILED",
            context={"tenant_hash": _hash_text(tenant_id)},
        ) from e


@trace_operation("db_transaction")
def execute_in_transaction(
    session: Session,
    operation: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Execute an operation within a transaction with proper error handling.

    Args:
        session: SQLAlchemy session
        operation: Callable to execute
        *args: Positional arguments for operation
        **kwargs: Keyword arguments for operation

    Returns:
        Result of the operation

    Raises:
        TransactionError: On transaction failure
    """
    try:
        result = operation(session, *args, **kwargs)
        session.commit()
        return result
    except Exception as e:
        try:
            session.rollback()
        except Exception:
            logger.error("Rollback failed", exc_info=True)

        if not isinstance(e, TransactionError):
            raise TransactionError(
                message="Transaction failed",
                error_code="TRANSACTION_FAILED",
                context={
                    "operation": (
                        operation.__name__
                        if hasattr(operation, "__name__")
                        else str(operation)
                    )
                },
            ) from e
        raise


# Event listener to log slow queries (optional, for observability)
@event.listens_for(engine, "before_cursor_execute")
def receive_before_cursor_execute(
    conn, cursor, statement, parameters, context, executemany
):
    """Log queries for debugging (can be extended for metrics)."""
    conn.info.setdefault("query_start_time", []).append(time.perf_counter())


@event.listens_for(engine, "after_cursor_execute")
def receive_after_cursor_execute(
    conn, cursor, statement, parameters, context, executemany
):
    """Log slow queries for observability."""
    start_times = conn.info.get("query_start_time")
    if not start_times:
        return
    start_time = start_times.pop()
    total_time = time.perf_counter() - start_time
    if total_time > SLOW_QUERY_THRESHOLD_SECONDS:
        statement_text = statement or ""
        statement_hash = _hash_text(str(statement_text))
        logger.warning(
            "Slow query (%.2fs) hash=%s length=%s",
            total_time,
            statement_hash,
            len(statement_text),
        )


@event.listens_for(engine, "handle_error")
def receive_handle_error(exception_context):
    """Ensure query start times are cleaned up on DB errors."""
    connection = getattr(exception_context, "connection", None)
    if connection is None:
        return
    start_times = connection.info.get("query_start_time")
    if start_times:
        start_times.pop()
