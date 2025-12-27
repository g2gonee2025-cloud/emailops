import pytest
from cortex.common.exceptions import (
    CortexError,
    EmbeddingError,
    ValidationError,
    ProviderError,
    FileOperationError,
    TransactionError,
    SecurityError,
    LLMOutputSchemaError,
    RetrievalError,
    RateLimitError,
    CircuitBreakerOpenError,
    PolicyViolationError,
)

def test_cortex_error_initialization():
    """Test CortexError initialization."""
    err = CortexError("Test message", error_code="TEST_CODE", context={"foo": "bar"}, extra="baz")
    assert err.message == "Test message"
    assert err.error_code == "TEST_CODE"
    assert err.context == {"foo": "bar", "extra": "baz"}
    assert str(err) == "Test message"

def test_cortex_error_to_dict():
    """Test CortexError serialization."""
    err = CortexError("Test message", error_code="TEST_CODE", context={"foo": "bar"})
    expected = {
        "error_type": "CortexError",
        "message": "Test message",
        "error_code": "TEST_CODE",
        "context": {"foo": "bar"},
    }
    assert err.to_dict() == expected

@pytest.mark.parametrize(
    "exc_class,init_args,expected_attrs",
    [
        (EmbeddingError, {"message": "Embedding failed", "retryable": True}, {"retryable": True}),
        (ValidationError, {"message": "Invalid input", "field": "email", "rule": "required"}, {"field": "email", "rule": "required"}),
        (ProviderError, {"message": "Provider issue", "provider": "openai", "retryable": False}, {"provider": "openai", "retryable": False}),
        (FileOperationError, {"message": "File not found", "file_path": "/tmp/test", "operation": "read"}, {"file_path": "/tmp/test", "operation": "read"}),
        (TransactionError, {"message": "Transaction failed", "transaction_id": "123"}, {"transaction_id": "123"}),
        (SecurityError, {"message": "Unauthorized access", "threat_type": "injection"}, {"threat_type": "injection"}),
        (LLMOutputSchemaError, {"message": "Schema mismatch", "schema_name": "user_profile", "raw_output": "...", "repair_attempts": 3}, {"schema_name": "user_profile", "raw_output": "...", "repair_attempts": 3}),
        (RetrievalError, {"message": "Could not retrieve", "query": "test query"}, {"query": "test query"}),
        (RateLimitError, {"message": "Rate limit exceeded", "provider": "anthropic", "retry_after": 60}, {"provider": "anthropic", "retryable": True, "retry_after": 60}),
        (CircuitBreakerOpenError, {"message": "Circuit open", "provider": "test", "reset_at": 12345}, {"provider": "test", "retryable": False, "reset_at": 12345}),
        (PolicyViolationError, {"message": "Policy violation", "action": "delete", "policy_name": "admin_only"}, {"action": "delete", "policy_name": "admin_only", "threat_type": "policy_violation"}),
    ],
)
def test_subclass_initialization(exc_class, init_args, expected_attrs):
    """Test that all CortexError subclasses correctly initialize their specific attributes."""
    err = exc_class(**init_args)
    for attr, value in expected_attrs.items():
        assert getattr(err, attr) == value
    assert err.message == init_args["message"]

def test_subclass_kwargs_override():
    """Test that kwargs can override default values in subclasses."""
    err = EmbeddingError("Test", retryable=True)
    assert err.retryable is True

    err_override = EmbeddingError("Test", retryable=False)
    assert err_override.retryable is False

def test_context_handling():
    """Test that extra kwargs are correctly added to the context dictionary."""
    err = ValidationError("Invalid", field="email", rule="required", extra_info="123")
    assert err.field == "email"
    assert err.rule == "required"
    assert err.context == {"extra_info": "123"}

def test_rate_limit_error_kwargs():
    """Test RateLimitError with kwargs."""
    err = RateLimitError("Rate limit exceeded", **{'provider': 'openai', 'retry_after': 30})
    assert err.provider == 'openai'
    assert err.retry_after == 30
    assert err.retryable is True

def test_circuit_breaker_open_error_kwargs():
    """Test CircuitBreakerOpenError with kwargs."""
    err = CircuitBreakerOpenError("Circuit open", **{'provider': 'test', 'reset_at': 12345})
    assert err.provider == 'test'
    assert err.reset_at == 12345
    assert err.retryable is False

def test_policy_violation_error_kwargs():
    """Test PolicyViolationError with kwargs."""
    err = PolicyViolationError("Policy violation", **{'action': 'delete', 'policy_name': 'admin_only'})
    assert err.action == 'delete'
    assert err.policy_name == 'admin_only'
    assert err.threat_type == 'policy_violation'
