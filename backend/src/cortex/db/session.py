"""
Database session management.

Implements §4.2 and §11.1 of the Canonical Blueprint.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional

from cortex.common.exceptions import TransactionError
from cortex.config.loader import get_config
from cortex.observability import trace_operation
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

_config = get_config()

engine = create_engine(
    _config.database.url,
    pool_size=_config.database.pool_size,
    max_overflow=_config.database.max_overflow,
    pool_pre_ping=True,
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
def get_db_session(tenant_id: Optional[str] = None) -> Generator[Session, None, None]:
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
        except Exception as rollback_error:
            logger.error(f"Rollback failed: {rollback_error}")

        # Wrap in TransactionError for non-CortexErrors
        if not isinstance(e, TransactionError):
            raise TransactionError(
                message=f"Database transaction failed: {e}",
                error_code="TRANSACTION_FAILED",
                context={"original_error": str(e)},
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
    import re

    if not tenant_id:
        raise TransactionError(
            message="tenant_id is required for RLS",
            error_code="RLS_TENANT_REQUIRED",
        )

    # Validate tenant_id format to prevent SQL injection
    if not re.match(r"^[a-zA-Z0-9_-]+$", tenant_id):
        raise TransactionError(
            message=f"Invalid tenant_id format: {tenant_id}",
            error_code="RLS_TENANT_INVALID",
        )

    try:
        session.execute(text(f"SET app.current_tenant = '{tenant_id}'"))
        logger.debug(f"Set RLS tenant context: {tenant_id}")
    except Exception as e:
        raise TransactionError(
            message=f"Failed to set RLS tenant: {e}",
            error_code="RLS_SET_FAILED",
            context={"tenant_id": tenant_id},
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
        except Exception as rollback_error:
            logger.error(f"Rollback failed: {rollback_error}")

        if not isinstance(e, TransactionError):
            raise TransactionError(
                message=f"Transaction failed: {e}",
                error_code="TRANSACTION_FAILED",
                context={
                    "operation": operation.__name__
                    if hasattr(operation, "__name__")
                    else str(operation)
                },
            ) from e
        raise


# Event listener to log slow queries (optional, for observability)
@event.listens_for(engine, "before_cursor_execute")
def receive_before_cursor_execute(
    conn, cursor, statement, parameters, context, executemany
):
    """Log queries for debugging (can be extended for metrics)."""
    conn.info.setdefault("query_start_time", []).append(__import__("time").time())


@event.listens_for(engine, "after_cursor_execute")
def receive_after_cursor_execute(
    conn, cursor, statement, parameters, context, executemany
):
    """Log slow queries for observability."""
    total_time = __import__("time").time() - conn.info["query_start_time"].pop()
    if total_time > 1.0:  # Log queries taking more than 1 second
        logger.warning(f"Slow query ({total_time:.2f}s): {statement[:200]}...")
