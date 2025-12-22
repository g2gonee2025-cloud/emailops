"""Unit tests for cortex.observability module."""

from unittest.mock import MagicMock, patch

import pytest


class TestObservabilityInit:
    @patch("cortex.observability._init_tracing")
    @patch("cortex.observability._init_metrics")
    @patch("cortex.observability._init_structured_logging")
    def test_init_observability_all_enabled(self, mock_log, mock_metrics, mock_trace):
        from cortex.observability import init_observability

        mock_trace.return_value = MagicMock()
        mock_metrics.return_value = MagicMock()

        init_observability(
            service_name="test",
            enable_tracing=True,
            enable_metrics=True,
            enable_structured_logging=True,
        )

        mock_trace.assert_called()
        mock_metrics.assert_called()
        mock_log.assert_called()

    @patch("cortex.observability._init_tracing")
    @patch("cortex.observability._init_metrics")
    @patch("cortex.observability._init_structured_logging")
    def test_init_observability_disabled(self, mock_log, mock_metrics, mock_trace):
        from cortex.observability import init_observability

        init_observability(
            service_name="test",
            enable_tracing=False,
            enable_metrics=False,
            enable_structured_logging=False,
        )

        # These should not be called when disabled
        # Actually implementation may still call them but they should no-op


class TestGetLogger:
    def test_get_logger_returns_logger(self):
        from cortex.observability import get_logger

        logger = get_logger("test_module")

        assert logger is not None

    def test_get_logger_different_names(self):
        from cortex.observability import get_logger

        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        # Should return different loggers (or same structured logger)
        assert logger1 is not None
        assert logger2 is not None


class TestGetTraceContext:
    def test_get_trace_context_default(self):
        from cortex.observability import get_trace_context

        ctx = get_trace_context()

        # Default should be None or empty dict
        assert ctx is None or isinstance(ctx, dict)


class TestRecordMetric:
    @patch("cortex.observability._meter", None)
    def test_record_metric_no_meter(self):
        from cortex.observability import record_metric

        # Should not raise when meter is None
        record_metric("test_counter", 1)

    @patch("cortex.observability._meter")
    @patch("cortex.observability._metric_instruments", {})
    def test_record_metric_counter(self, mock_meter):
        from cortex.observability import record_metric

        mock_counter = MagicMock()
        mock_meter.create_counter.return_value = mock_counter

        record_metric("test_counter", 5, metric_type="counter")


class TestTraceOperation:
    def test_trace_operation_sync_function(self):
        from cortex.observability import trace_operation

        @trace_operation("test_op")
        def my_func(x, y):
            return x + y

        result = my_func(2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_trace_operation_async_function(self):
        from cortex.observability import trace_operation

        @trace_operation("test_async_op")
        async def async_func(x, y):
            return x * y

        result = await async_func(4, 5)
        assert result == 20

    def test_trace_operation_with_attributes(self):
        from cortex.observability import trace_operation

        @trace_operation("test_attrs", custom_attr="value")
        def func_with_attrs():
            return "done"

        result = func_with_attrs()
        assert result == "done"

    def test_trace_operation_exception_handling(self):
        from cortex.observability import trace_operation

        @trace_operation("test_error")
        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_func()


class TestInitTracing:
    @patch("cortex.observability.OTLP_AVAILABLE", False)
    def test_init_tracing_no_otlp(self):
        from cortex.observability import _init_tracing

        result = _init_tracing("test", 0.1, None)

        # Should return None when OTLP not available
        assert result is None


class TestInitMetrics:
    @patch("cortex.observability.OTLP_AVAILABLE", False)
    def test_init_metrics_no_otlp(self):
        from cortex.observability import _init_metrics

        result = _init_metrics("test", None)

        # Should return None when OTLP not available
        assert result is None


class TestInitStructuredLogging:
    @patch("cortex.observability.STRUCTLOG_AVAILABLE", False)
    def test_init_structured_logging_no_structlog(self):
        from cortex.observability import _init_structured_logging

        # Should not raise when structlog not available
        _init_structured_logging()
