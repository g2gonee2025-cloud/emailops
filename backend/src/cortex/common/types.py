"""
Core type definitions for Cortex error handling.

Implements Result[T, E] pattern for type-safe error handling across the codebase.
This replaces mixed error signaling patterns (tuple, Optional, exceptions).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E")
U = TypeVar("U")


@dataclass(frozen=True)
class Result(Generic[T, E]):
    """
    Type-safe result container for operations that may fail.

    Replaces mixed error patterns (tuple[bool, str], T | None, exceptions)
    with a single, composable, type-safe error handling mechanism.

    Attributes:
        ok: True if operation succeeded, False otherwise
        value: Result value (only present if ok=True)
        error: Error information (only present if ok=False)

    Examples:
        >>> # Success case
        >>> result = Result.success(42)
        >>> if result.ok:
        ...     print(result.value)  # Type-safe: mypy knows this is int
        32

        >>> # Failure case
        >>> result = Result.failure("File not found")
        >>> if not result.ok:
        ...     print(result.error)
        File not found

        >>> # Unwrap with default
        >>> value = result.unwrap_or(0)
        0

        >>> # Transform success values
        >>> result = Result.success(5)
        >>> doubled = result.map(lambda x: x * 2)
        >>> doubled.value
        10
    """

    ok: bool
    value: T | None = None
    error: E | None = None

    def __post_init__(self) -> None:
        """Validate Result invariants."""
        if self.ok and self.value is None:
            # Allow None as a valid value if T is Optional, but typically value is present.
            # However, for void success, value might be None.
            # Let's relax this check or ensure we pass None explicitly.
            pass
        if not self.ok and self.error is None:
            raise ValueError("Failure Result must have an error")
        if self.ok and self.error is not None:
            raise ValueError("Success Result cannot have an error")
        if not self.ok and self.value is not None:
            raise ValueError("Failure Result cannot have a value")

    @classmethod
    def success(cls, value: T) -> Result[T, E]:
        """
        Create a success Result containing a value.

        Args:
            value: The success value

        Returns:
            Result with ok=True and the provided value
        """
        return cls(ok=True, value=value, error=None)

    @classmethod
    def ok(cls, value: T) -> Result[T, E]:
        """Alias for success() - canonical API."""
        return cls.success(value)

    @classmethod
    def failure(cls, error: E) -> Result[T, E]:
        """
        Create a failure Result containing an error.

        Args:
            error: The error information

        Returns:
            Result with ok=False and the provided error
        """
        return cls(ok=False, value=None, error=error)

    @classmethod
    def err(cls, error: E) -> Result[T, E]:
        """Alias for failure() - canonical API."""
        return cls.failure(error)

    def is_ok(self) -> bool:
        """Check if Result is success."""
        return self.ok

    def is_err(self) -> bool:
        """Check if Result is failure."""
        return not self.ok

    def unwrap(self) -> T:
        """
        Get the value or raise if error.

        Returns:
            The success value

        Raises:
            ValueError: If Result is a failure
        """
        if not self.ok:
            raise ValueError(f"Called unwrap() on failure Result: {self.error}")
        # assert self.value is not None  # Help type checker
        return self.value  # type: ignore

    def unwrap_err(self) -> E:
        """
        Get the error or raise if success.

        Returns:
            The error value

        Raises:
            ValueError: If Result is a success
        """
        if self.ok:
            raise ValueError(f"Called unwrap_err() on success Result: {self.value}")
        return self.error  # type: ignore

    def unwrap_or(self, default: T) -> T:
        """
        Get the value or return default if error.

        Args:
            default: Default value to return on failure

        Returns:
            The success value or default
        """
        return self.value if self.ok else default  # type: ignore

    def unwrap_or_else(self, fn: Callable[[E], T]) -> T:
        """
        Get the value or compute from error.

        Args:
            fn: Function to compute default from error

        Returns:
            The success value or result of fn(error)
        """
        if self.ok:
            return self.value  # type: ignore
        assert self.error is not None
        return fn(self.error)

    def expect(self, msg: str) -> T:
        """
        Get the value or raise with custom message.

        Args:
            msg: Error message to use if failure

        Returns:
            The success value

        Raises:
            ValueError: If Result is a failure, with custom message
        """
        if not self.ok:
            raise ValueError(f"{msg}: {self.error}")
        return self.value  # type: ignore

    def map(self, fn: Callable[[T], U]) -> Result[U, E]:
        """
        Transform success value, preserve failure.

        Args:
            fn: Function to transform the value

        Returns:
            Result with transformed value if success, same error if failure
        """
        if self.ok:
            return Result.success(fn(self.value))  # type: ignore
        return Result.failure(self.error)  # type: ignore

    def map_error(self, fn: Callable[[E], U]) -> Result[T, U]:
        """
        Transform error, preserve success.

        Args:
            fn: Function to transform the error

        Returns:
            Same value if success, Result with transformed error if failure
        """
        if not self.ok:
            assert self.error is not None
            return Result.failure(fn(self.error))
        return Result.success(self.value)  # type: ignore

    def and_then(self, fn: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """
        Chain operations that return Result (flatMap/bind).

        Args:
            fn: Function that takes the value and returns a new Result

        Returns:
            Result of fn(value) if success, same error if failure

        Example:
            >>> def divide(x: int) -> Result[int, str]:
            ...     return Result.success(100 // x) if x != 0 else Result.failure("div by zero")
            >>> Result.success(10).and_then(divide)
            Result(ok=True, value=10, error=None)
            >>> Result.success(0).and_then(divide)
            Result(ok=False, value=None, error='div by zero')
        """
        if self.ok:
            return fn(self.value)  # type: ignore
        return Result.failure(self.error)  # type: ignore

    def or_else(self, fn: Callable[[E], Result[T, E]]) -> Result[T, E]:
        """
        Provide alternative Result on failure.

        Args:
            fn: Function that takes the error and returns alternative Result

        Returns:
            Same Result if success, result of fn(error) if failure
        """
        if not self.ok:
            assert self.error is not None
            return fn(self.error)
        return self


# -------------------------
# Result Helper Functions
# -------------------------

def collect_results(results: list[Result[T, E]]) -> Result[list[T], E]:
    """
    Collect multiple Results into single Result[list[T], E].

    Fails fast on first error - if any Result is failure, returns that failure.
    Otherwise returns success with list of all values.

    Args:
        results: List of Result objects to collect

    Returns:
        Result[list[T], E]: Success with all values or first failure

    Example:
        >>> results = [Result.success(1), Result.success(2), Result.success(3)]
        >>> collected = collect_results(results)
        >>> collected.unwrap()
        [1, 2, 3]

        >>> results = [Result.success(1), Result.failure("error"), Result.success(3)]
        >>> collected = collect_results(results)
        >>> collected.ok
        False
    """
    values: list[T] = []
    for r in results:
        if not r.ok:
            return Result.failure(r.error)  # type: ignore
        values.append(r.value)  # type: ignore
    return Result.success(values)


def sequence_results(
    items: list[T], fn: Callable[[T], Result[U, E]]
) -> Result[list[U], E]:
    """
    Apply function returning Result to each item, collect results.

    Fails fast on first error. Equivalent to collect(map(fn, items)).

    Args:
        items: List of items to process
        fn: Function that takes item and returns Result

    Returns:
        Result[list[U], E]: Success with all results or first failure

    Example:
        >>> def validate_positive(x: int) -> Result[int, str]:
        ...     return Result.success(x) if x > 0 else Result.failure(f"{x} not positive")
        >>> sequence_results([1, 2, 3], validate_positive).unwrap()
        [1, 2, 3]
        >>> sequence_results([1, -2, 3], validate_positive).ok
        False
    """
    results = [fn(item) for item in items]
    return collect_results(results)


def traverse_results(
    items: list[T], fn: Callable[[T], Result[U, E]]
) -> Result[list[U], list[E]]:
    """
    Apply function to all items, collecting all errors (not fail-fast).

    Unlike sequence_results which fails on first error, this processes all items
    and returns either success with all values or failure with all errors.

    Args:
        items: List of items to process
        fn: Function that takes item and returns Result

    Returns:
        Result[list[U], list[E]]: Success with all values or list of all errors

    Example:
        >>> def validate(x: int) -> Result[int, str]:
        ...     return Result.success(x) if x > 0 else Result.failure(f"{x} invalid")
        >>> traverse_results([1, -2, 3, -4], validate)
        Result(ok=False, value=None, error=['-2 invalid', '-4 invalid'])
    """
    successes: list[U] = []
    failures: list[E] = []

    for item in items:
        result = fn(item)
        if result.ok:
            successes.append(result.value)  # type: ignore
        else:
            assert result.error is not None
            failures.append(result.error)

    if failures:
        return Result.failure(failures)  # type: ignore
    return Result.success(successes)


__all__ = ["Result", "collect_results", "sequence_results", "traverse_results"]