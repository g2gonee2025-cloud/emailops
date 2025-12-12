"""
Core type definitions for Cortex error handling.

This module implements a typed Result[T, E] pattern using a *discriminated union*
(two variants: Ok and Err) so that static type checkers can correctly narrow types
based on the `.ok` tag.

Why this shape?
- `Ok` has:  ok == True  and a `value: T`
- `Err` has: ok == False and an `error: E`
- `Result[T, E]` is an alias: `Ok[T, E] | Err[T, E]`

With this structure, a type checker can understand:

    r: Result[int, str] = ...
    if r.ok:
        # r is Ok[int, str] here; r.value is int
        print(r.value)
    else:
        # r is Err[int, str] here; r.error is str
        print(r.error)

This avoids the common pitfall of `value: T | None` / `error: E | None` on a
single class where the checker cannot reliably correlate `ok` with the presence
of `value` or `error`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, Literal, TypeAlias, TypeGuard, TypeVar, cast

T = TypeVar("T")
E = TypeVar("E")
U = TypeVar("U")
F = TypeVar("F")


class ResultUnwrapError(RuntimeError):
    """Raised when unwrap/expect operations are used on the wrong variant."""


class _ResultOps(Generic[T, E]):
    """
    Mixin implementing Result operations.

    Both Ok and Err inherit these methods. Users typically interact with the
    discriminated union `Result[T, E] = Ok[T, E] | Err[T, E]`.
    """

    # Subclasses provide a Literal[True]/Literal[False] field named `ok`.
    ok: bool

    def is_ok(self) -> bool:
        """Return True if this is Ok."""
        return self.ok

    def is_err(self) -> bool:
        """Return True if this is Err."""
        return not self.ok

    def unwrap(self) -> T:
        """
        Return the success value or raise ResultUnwrapError if this is Err.
        """
        if self.ok:
            return cast(Ok[T, E], self).value
        err = cast(Err[T, E], self).error
        raise ResultUnwrapError(f"Called unwrap() on Err: {err!r}")

    def unwrap_err(self) -> E:
        """
        Return the error value or raise ResultUnwrapError if this is Ok.
        """
        if not self.ok:
            return cast(Err[T, E], self).error
        val = cast(Ok[T, E], self).value
        raise ResultUnwrapError(f"Called unwrap_err() on Ok: {val!r}")

    def unwrap_or(self, default: T) -> T:
        """
        Return the success value, or `default` if this is Err.
        """
        if self.ok:
            return cast(Ok[T, E], self).value
        return default

    def unwrap_or_else(self, fn: Callable[[E], T]) -> T:
        """
        Return the success value, or compute a fallback from the error.
        """
        if self.ok:
            return cast(Ok[T, E], self).value
        return fn(cast(Err[T, E], self).error)

    def expect(self, msg: str) -> T:
        """
        Return the success value, or raise ResultUnwrapError with a custom message.
        """
        if self.ok:
            return cast(Ok[T, E], self).value
        err = cast(Err[T, E], self).error
        raise ResultUnwrapError(f"{msg}: {err!r}")

    def map(self, fn: Callable[[T], U]) -> "Result[U, E]":
        """
        Transform the Ok value, preserving Err.
        """
        if self.ok:
            return Ok(fn(cast(Ok[T, E], self).value))
        return cast(Err[U, E], self)

    def map_error(self, fn: Callable[[E], F]) -> "Result[T, F]":
        """
        Transform the Err error, preserving Ok.
        """
        if self.ok:
            return cast(Ok[T, F], self)
        return Err(fn(cast(Err[T, E], self).error))

    def and_then(self, fn: Callable[[T], "Result[U, E]"]) -> "Result[U, E]":
        """
        Chain computations that themselves return Result (flatMap / bind).
        """
        if self.ok:
            return fn(cast(Ok[T, E], self).value)
        return cast(Err[U, E], self)

    def or_else(self, fn: Callable[[E], "Result[T, E]"]) -> "Result[T, E]":
        """
        Provide an alternative Result if this is Err.
        """
        if not self.ok:
            return fn(cast(Err[T, E], self).error)
        return cast(Ok[T, E], self)


@dataclass(frozen=True, slots=True)
class Ok(_ResultOps[T, E]):
    """
    Success variant of Result[T, E].

    Attributes:
        value: The success value.
        ok: Literal True tag used for type narrowing.
    """

    value: T
    ok: Literal[True] = True


@dataclass(frozen=True, slots=True)
class Err(_ResultOps[T, E]):
    """
    Failure variant of Result[T, E].

    Attributes:
        error: The error value.
        ok: Literal False tag used for type narrowing.
    """

    error: E
    ok: Literal[False] = False


# The discriminated union users should type against.
Result: TypeAlias = Ok[T, E] | Err[T, E]


def is_ok(r: Result[T, E]) -> TypeGuard[Ok[T, E]]:
    """
    Type guard for Ok.

    Useful when a checker doesn't narrow as expected on `if r.ok:` in a given context.
    """
    return r.ok is True


def is_err(r: Result[T, E]) -> TypeGuard[Err[T, E]]:
    """
    Type guard for Err.

    Useful when a checker doesn't narrow as expected on `if not r.ok:` in a given context.
    """
    return r.ok is False


# -------------------------
# Result Helper Functions
# -------------------------


def collect_results(results: Iterable[Result[T, E]]) -> Result[list[T], E]:
    """
    Collect multiple Results into a single Result[list[T], E].

    Fails fast on the first Err.
    """
    values: list[T] = []
    for r in results:
        if not r.ok:
            return Err(cast(Err[T, E], r).error)
        values.append(cast(Ok[T, E], r).value)
    return Ok(values)


def sequence_results(
    items: Iterable[T], fn: Callable[[T], Result[U, E]]
) -> Result[list[U], E]:
    """
    Apply `fn` to each item and collect results (fail-fast).

    Unlike a list comprehension + collect, this truly stops at the first Err.
    """
    out: list[U] = []
    for item in items:
        r = fn(item)
        if not r.ok:
            return Err(cast(Err[U, E], r).error)
        out.append(cast(Ok[U, E], r).value)
    return Ok(out)


def traverse_results(
    items: Iterable[T], fn: Callable[[T], Result[U, E]]
) -> Result[list[U], list[E]]:
    """
    Apply `fn` to every item, collecting *all* errors (not fail-fast).

    Returns:
        Ok(list_of_values) if there are no errors
        Err(list_of_errors) otherwise
    """
    successes: list[U] = []
    failures: list[E] = []

    for item in items:
        r = fn(item)
        if r.ok:
            successes.append(cast(Ok[U, E], r).value)
        else:
            failures.append(cast(Err[U, E], r).error)

    if failures:
        return Err(failures)
    return Ok(successes)


__all__ = [
    "Result",
    "Ok",
    "Err",
    "ResultUnwrapError",
    "is_ok",
    "is_err",
    "collect_results",
    "sequence_results",
    "traverse_results",
]
