"""
Tests for the Result type and its operations.
"""

import pytest
from cortex.common.types import Err, Ok, Result, ResultUnwrapError, is_err, is_ok


def test_ok_creation_and_value():
    """Test that Ok holds the correct value and is identified as Ok."""
    ok_val = Ok(10)
    assert ok_val.ok
    assert ok_val.is_ok()
    assert not ok_val.is_err()
    assert ok_val.value == 10


def test_err_creation_and_error():
    """Test that Err holds the correct error and is identified as Err."""
    err_val = Err("An error occurred")
    assert not err_val.ok
    assert err_val.is_err()
    assert not err_val.is_ok()
    assert err_val.error == "An error occurred"


def test_unwrap_on_ok():
    """Test that unwrap returns the value for Ok."""
    assert Ok(42).unwrap() == 42


def test_unwrap_on_err():
    """Test that unwrap raises an error for Err."""
    with pytest.raises(ResultUnwrapError, match=r"Called unwrap\(\) on Err: 'error'"):
        Err("error").unwrap()


def test_unwrap_err_on_err():
    """Test that unwrap_err returns the error for Err."""
    assert Err("error").unwrap_err() == "error"


def test_unwrap_err_on_ok():
    """Test that unwrap_err raises an error for Ok."""
    with pytest.raises(ResultUnwrapError, match=r"Called unwrap_err\(\) on Ok: 42"):
        Ok(42).unwrap_err()


def test_unwrap_or_on_ok():
    """Test that unwrap_or returns the value for Ok."""
    assert Ok(10).unwrap_or(0) == 10


def test_unwrap_or_on_err():
    """Test that unwrap_or returns the default value for Err."""
    assert Err("error").unwrap_or(0) == 0


def test_unwrap_or_else_on_ok():
    """Test that unwrap_or_else returns the value for Ok."""
    assert Ok(10).unwrap_or_else(lambda e: len(e)) == 10


def test_unwrap_or_else_on_err():
    """Test that unwrap_or_else computes the default value for Err."""
    assert Err("error").unwrap_or_else(lambda e: len(e)) == 5


def test_expect_on_ok():
    """Test that expect returns the value for Ok."""
    assert Ok("value").expect("This should not fail") == "value"


def test_expect_on_err():
    """Test that expect raises an error with a custom message for Err."""
    with pytest.raises(ResultUnwrapError, match="Custom message: 'error'"):
        Err("error").expect("Custom message")


def test_map_on_ok():
    """Test that map transforms the value of Ok."""
    result = Ok(5).map(lambda x: x * 2)
    assert result.ok
    assert result.unwrap() == 10


def test_map_on_err():
    """Test that map preserves the error of Err."""
    result = Err("error").map(lambda x: x * 2)
    assert not result.ok
    assert result.unwrap_err() == "error"


def test_map_error_on_ok():
    """Test that map_error preserves the value of Ok."""
    result = Ok(5).map_error(lambda e: f"Error: {e}")
    assert result.ok
    assert result.unwrap() == 5


def test_map_error_on_err():
    """Test that map_error transforms the error of Err."""
    result = Err("fail").map_error(lambda e: f"Error: {e}")
    assert not result.ok
    assert result.unwrap_err() == "Error: fail"


def test_and_then_on_ok():
    """Test that and_then chains computations for Ok."""
    result = Ok(5).and_then(lambda x: Ok(x * 2))
    assert result.ok
    assert result.unwrap() == 10


def test_and_then_on_ok_returning_err():
    """Test that and_then can return an Err from an Ok."""
    result = Ok(5).and_then(lambda x: Err(f"Failed with {x}"))
    assert not result.ok
    assert result.unwrap_err() == "Failed with 5"


def test_and_then_on_err():
    """Test that and_then preserves the error of Err."""
    result = Err("error").and_then(lambda x: Ok(x * 2))
    assert not result.ok
    assert result.unwrap_err() == "error"


def test_or_else_on_ok():
    """Test that or_else preserves the value of Ok."""
    result = Ok(5).or_else(lambda e: Ok(0))
    assert result.ok
    assert result.unwrap() == 5


def test_or_else_on_err():
    """Test that or_else provides an alternative for Err."""
    result = Err("error").or_else(lambda e: Ok(len(e)))
    assert result.ok
    assert result.unwrap() == 5


def test_or_else_on_err_returning_err():
    """Test that or_else can return another Err."""
    result = Err("error").or_else(lambda e: Err(f"New error: {e}"))
    assert not result.ok
    assert result.unwrap_err() == "New error: error"


def test_is_ok_type_guard():
    """Test the is_ok type guard."""
    r: Result[int, str] = Ok(1)
    if is_ok(r):
        assert r.value == 1
    else:
        pytest.fail("is_ok should have returned True")


def test_is_err_type_guard():
    """Test the is_err type guard."""
    r: Result[int, str] = Err("error")
    if is_err(r):
        assert r.error == "error"
    else:
        pytest.fail("is_err should have returned True")
