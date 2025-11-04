"""
Resilience Patterns for EmailOps Services

Implements retry logic, circuit breakers, and rate limiting
to prevent cascading failures and improve system reliability.
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout: float = 60.0  # Seconds before trying half-open
    expected_exceptions: tuple = (Exception,)  # Exceptions to count as failures


@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_attempts: int = 3
    initial_delay: float = 1.0  # Seconds
    max_delay: float = 60.0  # Maximum backoff delay
    exponential_base: float = 2.0  # Exponential backoff base
    jitter: bool = True  # Add random jitter to prevent thundering herd


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascading failures by failing fast when a service is down.
    """

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: datetime | None = None
        self._lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return False
        return datetime.now() >= self.last_failure_time + timedelta(seconds=self.config.timeout)

    async def call_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute async function with circuit breaker protection"""
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            # Check circuit state
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
                else:
                    raise CircuitOpenError(f"Circuit breaker '{self.name}' is OPEN")

        # Attempt the call
        try:
            result = await func(*args, **kwargs)

            async with self._lock:
                self._on_success()

            return result

        except self.config.expected_exceptions:
            async with self._lock:
                self._on_failure()
            raise

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute synchronous function with circuit breaker protection"""
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
            else:
                raise CircuitOpenError(f"Circuit breaker '{self.name}' is OPEN")

        # Attempt the call
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except self.config.expected_exceptions:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker '{self.name}' CLOSED after recovery")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0  # Reset on success

    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker '{self.name}' OPEN after half-open failure")

        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}' OPEN after {self.failure_count} failures"
                )

    def reset(self):
        """Manually reset the circuit breaker"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info(f"Circuit breaker '{self.name}' manually reset")

    def get_state(self) -> dict[str, Any]:
        """Get current state information"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class RetryManager:
    """
    Retry logic with exponential backoff and jitter.
    """

    def __init__(self, config: RetryConfig | None = None):
        self.config = config or RetryConfig()

    async def retry_async(
        self,
        func: Callable[..., T],
        *args,
        on_retry: Callable[[Exception, int], None] | None = None,
        **kwargs
    ) -> T:
        """
        Retry async function with exponential backoff.

        Args:
            func: Async function to retry
            on_retry: Optional callback(exception, attempt) on each retry

        Returns:
            Function result

        Raises:
            Last exception if all retries fail
        """
        last_exception = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return await func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if attempt == self.config.max_attempts:
                    logger.error(
                        f"All {self.config.max_attempts} retry attempts failed: {e}"
                    )
                    raise

                # Calculate delay with exponential backoff
                delay = min(
                    self.config.initial_delay * (self.config.exponential_base ** (attempt - 1)),
                    self.config.max_delay
                )

                # Add jitter if enabled
                if self.config.jitter:
                    import random
                    delay *= (0.5 + random.random())

                logger.warning(
                    f"Retry attempt {attempt}/{self.config.max_attempts} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                if on_retry:
                    on_retry(e, attempt)

                await asyncio.sleep(delay)

        raise last_exception or Exception("Retry failed")

    def retry(
        self,
        func: Callable[..., T],
        *args,
        on_retry: Callable[[Exception, int], None] | None = None,
        **kwargs
    ) -> T:
        """
        Retry synchronous function with exponential backoff.

        Args:
            func: Function to retry
            on_retry: Optional callback(exception, attempt) on each retry

        Returns:
            Function result

        Raises:
            Last exception if all retries fail
        """
        last_exception = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if attempt == self.config.max_attempts:
                    logger.error(
                        f"All {self.config.max_attempts} retry attempts failed: {e}"
                    )
                    raise

                # Calculate delay with exponential backoff
                delay = min(
                    self.config.initial_delay * (self.config.exponential_base ** (attempt - 1)),
                    self.config.max_delay
                )

                # Add jitter if enabled
                if self.config.jitter:
                    import random
                    delay *= (0.5 + random.random())

                logger.warning(
                    f"Retry attempt {attempt}/{self.config.max_attempts} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                if on_retry:
                    on_retry(e, attempt)

                time.sleep(delay)

        raise last_exception or Exception("Retry failed")


class RateLimiter:
    """
    Token bucket rate limiter to prevent overwhelming services.
    """

    def __init__(self, rate: float, burst: int):
        """
        Initialize rate limiter.

        Args:
            rate: Tokens per second
            burst: Maximum burst size (bucket capacity)
        """
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None

    async def acquire_async(self, tokens: int = 1) -> None:
        """Async acquire tokens, blocking if necessary"""
        if self._lock is None:
            self._lock = asyncio.Lock()

        while True:
            async with self._lock:
                self._refill()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return

                # Calculate wait time
                deficit = tokens - self.tokens
                wait_time = deficit / self.rate

            await asyncio.sleep(wait_time)

    def acquire(self, tokens: int = 1) -> None:
        """Synchronous acquire tokens, blocking if necessary"""
        while True:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return

            # Calculate wait time
            deficit = tokens - self.tokens
            wait_time = deficit / self.rate
            time.sleep(wait_time)

    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.monotonic()
        elapsed = now - self.last_update

        # Add tokens based on rate
        self.tokens = min(
            self.burst,
            self.tokens + elapsed * self.rate
        )

        self.last_update = now

    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without blocking"""
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True

        return False


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


def circuit_breaker(
    failure_threshold: int | None = None,
    timeout: float | None = None,
    error_types: tuple | None = None,
    name: str = "default",
    config: CircuitBreakerConfig | None = None,
    **_kwargs  # Accept other kwargs for compatibility
) -> Callable:
    """
    Decorator to add circuit breaker to a function.

    Can be called with either a CircuitBreakerConfig object or individual parameters.

    Example:
        @circuit_breaker(failure_threshold=5, timeout=60)
        def call_api():
            return requests.get("https://api.example.com")
    """
    # Build config from parameters if provided
    if config is None:
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold or 5,
            timeout=timeout or 60.0,
            expected_exceptions=error_types or (Exception,)
        )

    breaker = CircuitBreaker(name, config)

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await breaker.call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return breaker.call(func, *args, **kwargs)
            return sync_wrapper

    return decorator


# Alias for backward compatibility
with_circuit_breaker = circuit_breaker


def with_retry(
    max_attempts: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    _should_retry: Callable[[Exception], bool] | None = None,
    config: RetryConfig | None = None,
    **_kwargs  # Accept other kwargs for compatibility
) -> Callable:
    """
    Decorator to add retry logic to a function.

    Can be called with either a RetryConfig object or individual parameters.

    Example:
        @with_retry(max_attempts=5, base_delay=1.0)
        def unstable_operation():
            # May fail randomly
            pass
    """
    # Build config from parameters if provided
    if config is None:
        config = RetryConfig(
            max_attempts=max_attempts or 3,
            initial_delay=base_delay or 1.0,
            max_delay=max_delay or 60.0,
        )

    retry_manager = RetryManager(config)

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await retry_manager.retry_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return retry_manager.retry(func, *args, **kwargs)
            return sync_wrapper

    return decorator


# Global circuit breakers for different services
CIRCUIT_BREAKERS = {
    "vertex_api": CircuitBreaker(
        "vertex_api",
        CircuitBreakerConfig(failure_threshold=3, timeout=30)
    ),
    "index_service": CircuitBreaker(
        "index_service",
        CircuitBreakerConfig(failure_threshold=5, timeout=60)
    ),
    "file_service": CircuitBreaker(
        "file_service",
        CircuitBreakerConfig(failure_threshold=10, timeout=10)
    ),
}


def get_circuit_breaker(service_name: str) -> CircuitBreaker:
    """Get or create circuit breaker for a service"""
    if service_name not in CIRCUIT_BREAKERS:
        CIRCUIT_BREAKERS[service_name] = CircuitBreaker(service_name)
    return CIRCUIT_BREAKERS[service_name]


def reset_all_breakers():
    """Reset all circuit breakers (useful for testing)"""
    for breaker in CIRCUIT_BREAKERS.values():
        breaker.reset()


def get_all_breaker_states() -> dict[str, dict]:
    """Get state of all circuit breakers"""
    return {
        name: breaker.get_state()
        for name, breaker in CIRCUIT_BREAKERS.items()
    }
