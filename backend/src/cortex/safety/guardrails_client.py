"""
Guardrails Client.

Implements §9.3 of the Canonical Blueprint.
Provides LLM output validation with single repair attempt.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError

from cortex.llm.client import complete_json
from cortex.prompts import get_prompt
from cortex.observability import trace_operation

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# -----------------------------------------------------------------------------
# Models (Blueprint §9.3.1)
# -----------------------------------------------------------------------------

class RepairAttempt(BaseModel):
    """Result of a repair attempt."""
    success: bool = Field(..., description="Whether repair was successful")
    repaired_json: Optional[Dict[str, Any]] = Field(
        None, description="The repaired JSON if successful"
    )
    error_message: Optional[str] = Field(
        None, description="Error message if repair failed"
    )
    original_errors: List[str] = Field(
        default_factory=list, description="Original validation errors"
    )


# -----------------------------------------------------------------------------
# Repair Functions (Blueprint §9.3.1)
# -----------------------------------------------------------------------------

@trace_operation("attempt_llm_repair")
def attempt_llm_repair(
    invalid_json: Dict[str, Any],
    target_model: Type[T],
    validation_errors: List[str],
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
    # Format schema for repair prompt
    schema_json = target_model.model_json_schema()
    
    # Format errors for repair prompt
    errors_text = "\n".join(f"- {err}" for err in validation_errors)
    
    prompt = get_prompt("GUARDRAILS_REPAIR").format(
        error=errors_text
    )
    
    full_prompt = f"""{prompt}

TARGET SCHEMA:
{json.dumps(schema_json, indent=2)}

INVALID JSON:
{json.dumps(invalid_json, indent=2)}

VALIDATION ERRORS:
{errors_text}

Please fix the JSON to match the schema. Return ONLY valid JSON."""

    for attempt in range(max_attempts):
        try:
            # Use JSON mode for repair
            repaired = complete_json(
                prompt=full_prompt,
                schema=schema_json,
            )
            
            # Validate repaired output
            validated = target_model.model_validate(repaired)
            
            return RepairAttempt(
                success=True,
                repaired_json=validated.model_dump(),
                original_errors=validation_errors,
            )
            
        except ValidationError as e:
            logger.warning(
                f"Repair attempt {attempt + 1} failed",
                extra={"errors": [str(err) for err in e.errors()]},
            )
            # Update errors for next attempt
            validation_errors = [str(err) for err in e.errors()]
        except Exception as e:
            logger.warning(f"Repair attempt {attempt + 1} failed with error: {e}")
    
    return RepairAttempt(
        success=False,
        error_message="Repair failed after maximum attempts",
        original_errors=validation_errors,
    )


def validate_with_repair(
    raw_output: str,
    target_model: Type[T],
    correlation_id: str,
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
    from cortex.common.exceptions import LLMOutputSchemaError
    
    # Step 1: Parse JSON
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as e:
        # Try to extract JSON from markdown code blocks
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_output)
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
            max_attempts=1,  # Blueprint specifies single attempt
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

class GuardrailsClient:
    """
    Client for interacting with guardrails.
    
    Blueprint §9.3:
    * Constrained decoding
    * Validation via Pydantic
    * Repair attempts
    
    Note: For new code, prefer using validate_with_repair() directly.
    """
    
    def validate_output(
        self, 
        output: Any, 
        schema: Type[T], 
        max_retries: int = 1,
        correlation_id: Optional[str] = None,
    ) -> Optional[T]:
        """
        Validate LLM output against schema with repair attempts.
        
        Args:
            output: The output to validate (dict, string, or model instance)
            schema: The Pydantic model to validate against
            max_retries: Number of repair attempts (default 1)
            correlation_id: Optional correlation ID for logging
            
        Returns:
            Validated model instance or None if validation fails
        """
        try:
            # If already validated instance
            if isinstance(output, schema):
                return output
            
            # If dict, validate directly
            if isinstance(output, dict):
                try:
                    return schema.model_validate(output)
                except ValidationError as e:
                    if max_retries > 0:
                        return self._attempt_repair(output, schema, e, max_retries)
                    return None
            
            # If string, try to parse and validate
            if isinstance(output, str):
                try:
                    return validate_with_repair(
                        output, 
                        schema, 
                        correlation_id or "unknown"
                    )
                except Exception:
                    return None
            
            # Try generic validation
            return schema.model_validate(output)
                
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return None
    
    def _attempt_repair(
        self,
        invalid_data: Dict[str, Any],
        schema: Type[T],
        error: ValidationError,
        retries: int,
    ) -> Optional[T]:
        """Attempt to repair invalid data."""
        validation_errors = [str(err) for err in error.errors()]
        
        repair_result = attempt_llm_repair(
            invalid_json=invalid_data,
            target_model=schema,
            validation_errors=validation_errors,
            max_attempts=retries,
        )
        
        if repair_result.success and repair_result.repaired_json:
            try:
                return schema.model_validate(repair_result.repaired_json)
            except ValidationError:
                return None
        
        return None


# Singleton instance for convenience
_guardrails_client: Optional[GuardrailsClient] = None


def get_guardrails_client() -> GuardrailsClient:
    """Get or create the guardrails client singleton."""
    global _guardrails_client
    if _guardrails_client is None:
        _guardrails_client = GuardrailsClient()
    return _guardrails_client


__all__ = [
    "RepairAttempt",
    "attempt_llm_repair",
    "validate_with_repair",
    "GuardrailsClient",
    "get_guardrails_client",
]