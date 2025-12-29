import time
import unittest

from cortex.llm.runtime import ResilienceManager, _try_load_json


class TestLLMRuntime(unittest.TestCase):
    def test_extract_json_simple(self):
        """Test simple JSON object extraction."""
        text = '{"key": "value"}'
        self.assertEqual(_try_load_json(text), {"key": "value"})

    def test_extract_json_markdown(self):
        """Test extraction from markdown code blocks."""
        text = 'Here is the JSON:\n```json\n{"key": "value"}\n```'
        self.assertEqual(_try_load_json(text), {"key": "value"})

    def test_extract_json_markdown_no_lang(self):
        """Test extraction from markdown without language specifier."""
        text = '```\n{"key": "value"}\n```'
        self.assertEqual(_try_load_json(text), {"key": "value"})

    def test_extract_json_with_chatter(self):
        """Test extraction with conversational chatter."""
        text = 'Sure, here is the JSON you requested:\n\n{"key": "value"}\n\nHope that helps!'
        self.assertEqual(_try_load_json(text), {"key": "value"})


class TestResilienceManager(unittest.TestCase):
    """Test ResilienceManager circuit breaker and rate limiter."""

    def test_init(self):
        """Test ResilienceManager initialization."""
        rm = ResilienceManager()
        self.assertEqual(rm.circuit_state, "closed")
        self.assertEqual(rm.failures, 0)
        self.assertGreater(rm.tokens, 0)

    def test_check_circuit_closed(self):
        """Test check_circuit when circuit is closed."""
        rm = ResilienceManager()
        # Should not raise when circuit is closed
        rm.check_circuit()  # No exception

    def test_record_outcome_success_resets_failures(self):
        """Test that recording success resets failure count."""
        rm = ResilienceManager()
        rm.failures = 3
        rm.circuit_state = "half-open"
        rm.record_outcome(success=True)
        self.assertEqual(rm.failures, 0)
        self.assertEqual(rm.circuit_state, "closed")

    def test_record_outcome_failure_increments(self):
        """Test that recording failure increments failure count."""
        rm = ResilienceManager()
        rm.record_outcome(success=False)
        self.assertEqual(rm.failures, 1)

    def test_circuit_opens_after_threshold(self):
        """Test that circuit opens after reaching failure threshold."""
        rm = ResilienceManager()
        # Record enough failures to trip the circuit
        for _ in range(10):  # More than typical threshold
            rm.record_outcome(success=False)
        self.assertEqual(rm.circuit_state, "open")

    def test_check_circuit_open_raises(self):
        """Test that check_circuit raises when circuit is open."""
        from cortex.llm.runtime import CircuitBreakerOpenError

        rm = ResilienceManager()
        rm.circuit_state = "open"
        rm.last_failure_time = time.time()  # Just now

        with self.assertRaises(CircuitBreakerOpenError):
            rm.check_circuit()

    def test_check_circuit_transitions_to_half_open(self):
        """Test that circuit transitions to half-open after reset timeout."""
        rm = ResilienceManager()
        rm.circuit_state = "open"
        rm.last_failure_time = time.time() - 1000.0  # Long time ago

        rm.check_circuit()
        self.assertEqual(rm.circuit_state, "half-open")

    def test_acquire_token_immediate(self):
        """Test acquire_token returns immediately when tokens available."""
        rm = ResilienceManager()
        rm.tokens = 10.0  # Plenty of tokens
        start = time.time()
        rm.acquire_token()
        elapsed = time.time() - start
        # Should return quickly
        self.assertLess(elapsed, 0.1)
        self.assertGreater(rm.tokens, 0)

    def test_acquire_token_drains_tokens(self):
        """Test that acquire_token decrements token count."""
        rm = ResilienceManager()
        initial_tokens = rm.tokens
        if initial_tokens >= 1:
            rm.acquire_token()
            # tokens should decrease by approximately 1
            self.assertLess(rm.tokens, initial_tokens)


if __name__ == "__main__":
    unittest.main()
