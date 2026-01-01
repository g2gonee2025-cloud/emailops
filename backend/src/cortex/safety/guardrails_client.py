"""
Guardrails Client.

Implements §9.3 of the Canonical Blueprint.
Provides LLM output validation with single repair attempt.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import warnings
from typing import Any, TypeVar

from cortex.common.exceptions import LLMOutputSchemaError, ProviderError
from cortex.llm.client import complete_messages
from cortex.observability import trace_operation
from cortex.prompts import (
    SYSTEM_GUARDRAILS_REPAIR,
    USER_GUARDRAILS_REPAIR,
    construct_prompt_messages,
)
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# -----------------------------------------------------------------------------
# Models (Blueprint §9.3.1)
# -----------------------------------------------------------------------------


class RepairAttempt(BaseModel):
    """Result of a repair attempt."""

    success: bool = Field(..., description="Whether repair was successful")
    repaired_json: dict[str, Any] | None = Field(
        None, description="The repaired JSON if successful"
    )
    error_message: str | None = Field(
        None, description="Error message if repair failed"
    )
    original_errors: list[str] = Field(
        default_factory=list, description="Original validation errors"
    )


# -----------------------------------------------------------------------------
# Repair Functions (Blueprint §9.3.1)
# -----------------------------------------------------------------------------


@trace_operation("attempt_llm_repair")
def attempt_llm_repair(
    invalid_json: dict[str, Any],
    target_model: type[T],
    validation_errors: list[str],
    max_attempts: int = 1,
) -> RepairAttempt:
    """
    Attempt to repair invalid LLM JSON output.

    Uses a single repair attempt with structured error feedback
    as per Blueprint §9.3.

    Args:
        invalid_json: The invalid JSON from LLM
        target_model: The Pydantic model to validate against
        validation_errors: List of validation error messages
        max_attempts: Maximum repair attempts (default 1)

    Returns:
        RepairAttempt with success status and repaired JSON if successful
    """
    schema_json = target_model.model_json_schema()
    current_validation_errors = validation_errors[:]

    for attempt in range(max_attempts):
        errors_text = (
            "\n".join(f"- {err}" for err in current_validation_errors)
            or "- Unknown validation error"
        )
        try:
            messages = construct_prompt_messages(
                system_prompt_template=SYSTEM_GUARDRAILS_REPAIR,
                user_prompt_template=USER_GUARDRAILS_REPAIR,
                error=errors_text,
                invalid_json=json.dumps(invalid_json, indent=2),
                target_schema=json.dumps(schema_json, indent=2),
                validation_errors=errors_text,
            )
            raw_output = _complete_json_messages(messages, schema_json)
            repaired = _parse_json_output(raw_output)

            # Validate repaired output
            validated = target_model.model_validate(repaired)

            return RepairAttempt(
                success=True,
                repaired_json=validated.model_dump(),
                original_errors=validation_errors,
            )

        except ValidationError as e:
            logger.warning(
                "Repair attempt %d failed", attempt + 1, extra={"errors": e.errors()}
            )
            current_validation_errors = [str(err) for err in e.errors()]
        except json.JSONDecodeError as e:
            logger.warning("Repair attempt %d returned invalid JSON", attempt + 1)
            current_validation_errors = [f"Invalid JSON: {e}"]
        except Exception as e:
            logger.warning("Repair attempt %d failed with error: %s", attempt + 1, e)
            current_validation_errors = [str(e)]

    return RepairAttempt(
        success=False,
        error_message="Repair failed after maximum attempts",
        original_errors=validation_errors,
    )


def _complete_json_messages(
    messages: list[dict[str, str]],
    schema: dict[str, Any],
) -> str:
    schema_json = json.dumps(schema, indent=2)
    instructions = (
        "Respond with a single valid JSON object that conforms to this JSON Schema:\n"
        f"{schema_json}\n\n"
        "Do not include markdown. Return ONLY the JSON object."
    )
    payload = [dict(m) for m in messages]
    inserted = False
    for message in payload:
        if message.get("role") == "system" and isinstance(message.get("content"), str):
            message["content"] = f"{message['content']}\n\n{instructions}"
            inserted = True
            break
    if not inserted:
        payload.insert(0, {"role": "system", "content": instructions})

    kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
    }
    try:
        return complete_messages(payload, **kwargs)
    except ProviderError as exc:
        msg = str(exc).lower()
        if "response_format" in msg or "json_object" in msg:
            kwargs.pop("response_format", None)
            return complete_messages(payload, **kwargs)
        raise


def _parse_json_output(raw_output: str) -> dict[str, Any]:
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError as e:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_output)
        if match:
            return json.loads(match.group(1))
        raise e


def validate_with_repair(
    raw_output: str,
    target_model: type[T],
    correlation_id: str,
    max_attempts: int = 1,
) -> T:
    """
    Validate LLM output with single repair attempt.

    Implements the full §9.3 pipeline:
    1. Parse JSON
    2. Validate against Pydantic model
    3. If invalid, attempt single repair
    4. If still invalid, raise with logging

    Args:
        raw_output: Raw LLM output string
        target_model: Pydantic model to validate against
        correlation_id: Request correlation ID for logging

    Returns:
        Validated Pydantic model instance

    Raises:
        LLMOutputSchemaError: If validation fails after repair
    """
    # Step 1: Parse JSON
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as e:
        # Try to extract JSON from markdown code blocks
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_output)
        if match:
            try:
                parsed = json.loads(match.group(1))
            except json.JSONDecodeError:
                raise LLMOutputSchemaError(
                    message=f"Invalid JSON in code block: {e}",
                    correlation_id=correlation_id,
                )
        else:
            raise LLMOutputSchemaError(
                message=f"Invalid JSON: {e}",
                correlation_id=correlation_id,
            )

    # Step 2: Validate against model
    try:
        return target_model.model_validate(parsed)
    except ValidationError as initial_error:
        validation_errors = [str(err) for err in initial_error.errors()]

        # Step 3: Single repair attempt
        repair_result = attempt_llm_repair(
            invalid_json=parsed,
            target_model=target_model,
            validation_errors=validation_errors,
            max_attempts=max_attempts,
        )

        if repair_result.success and repair_result.repaired_json:
            return target_model.model_validate(repair_result.repaired_json)

        # Step 4: Log and raise
        logger.error(
            "LLM_OUTPUT_SCHEMA_ERROR",
            extra={
                "correlation_id": correlation_id,
                "model": target_model.__name__,
                "original_errors": repair_result.original_errors,
            },
        )

        raise LLMOutputSchemaError(
            message=f"Validation failed after repair: {repair_result.error_message}",
            correlation_id=correlation_id,
            details={"errors": repair_result.original_errors},
        )


# -----------------------------------------------------------------------------
# GuardrailsClient Class (Legacy Interface)
# -----------------------------------------------------------------------------


class GuardrailsClient:  # DEPRECATED
    """
    Client for interacting with guardrails.

    Blueprint §9.3:
    * Constrained decoding
    * Validation via Pydantic
    * Repair attempts

    Note: For new code, prefer using validate_with_repair() directly.
    """

    def __init__(self):
        """Initialize the client and warn about its deprecation."""
        warnings.warn(
            "GuardrailsClient is deprecated.", DeprecationWarning, stacklevel=2
        )

    def validate_output(
        self,
        output: Any,
        schema: type[T],
        max_retries: int = 1,
        correlation_id: str | None = None,
    ) -> T | None:
        """
        Validate LLM output against a schema, with repair attempts.

        This method is a wrapper around `validate_with_repair` and is maintained
        for backward compatibility. It catches all exceptions and returns None
        on failure.

        Args:
            output: The output to validate (dict, string, or model instance).
            schema: The Pydantic model to validate against.
            max_retries: Number of repair attempts.
            correlation_id: Optional correlation ID for logging.

        Returns:
            A validated model instance, or None if validation fails.
        """
        if isinstance(output, schema):
            return output

        raw_output = ""
        if isinstance(output, str):
            raw_output = output
        elif isinstance(output, dict):
            raw_output = json.dumps(output)
        else:
            try:
                raw_output = json.dumps(output)
            except (TypeError, ValueError):
                logger.warning(f"Could not serialize output of type {type(output)}")
                return None

        try:
            return validate_with_repair(
                raw_output,
                schema,
                correlation_id=correlation_id or "unknown",
                max_attempts=max_retries,
            )
        except LLMOutputSchemaError as e:
            logger.warning(f"Validation failed after repair: {e.details}")
            return None


# Singleton instance for convenience
_guardrails_client: GuardrailsClient | None = None
_client_lock = threading.Lock()


def get_guardrails_client() -> GuardrailsClient:  # DEPRECATED
    """Get or create the guardrails client singleton."""
    warnings.warn(
        "get_guardrails_client is deprecated.", DeprecationWarning, stacklevel=2
    )
    global _guardrails_client
    if _guardrails_client is None:
        with _client_lock:
            if _guardrails_client is None:
                _guardrails_client = GuardrailsClient()
    return _guardrails_client


__all__ = [
    "GuardrailsClient",
    "RepairAttempt",
    "attempt_llm_repair",
    "get_guardrails_client",
    "validate_with_repair",
]
