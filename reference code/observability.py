"""
P1-3, P2-3, P2-7 FIX: Observability infrastructure with OpenTelemetry.

Provides distributed tracing, metrics export, and structured logging for
production debugging and performance monitoring.

Features:
- Automatic span creation for key operations
- Metrics export (Prometheus, Cloud Monitoring)
- Structured logging with trace correlation
- Performance profiling
- Error tracking

Usage:
    from emailops.observability import trace_operation, record_metric, get_logger

    @trace_operation("index_conversation")
    def index_conversation(conv_dir):
        logger = get_logger(__name__)
        logger.info("indexing_started", conv_id=conv_dir.name)
        # ... work ...
        record_metric("conversations_indexed", 1, {"status": "success"})
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

# Optional OpenTelemetry imports
try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

    OTEL_AVAILABLE = True
except ImportError:
    trace = None  # type: ignore
    metrics = None  # type: ignore
    OTEL_AVAILABLE = False

# Optional structlog
try:
    import structlog

    STRUCTLOG_AVAILABLE = True
except ImportError:
    structlog = None  # type: ignore
    STRUCTLOG_AVAILABLE = False

from .core_config import EmailOpsConfig

__all__ = [
    "get_logger",
    "get_trace_context",
    "init_observability",
    "record_metric",
    "trace_operation",
]

# Context variable for trace correlation
_trace_context: ContextVar[dict[str, str] | None] = ContextVar(
    "trace_context", default=None
)

# Global tracer and meter (initialized lazily)
_tracer = None
_meter = None
_metric_instruments: dict[tuple[str, str], Any] = {}
_initialized = False


def init_observability(
    service_name: str = "emailops",
    enable_tracing: bool = True,
    enable_metrics: bool = True,
    enable_structured_logging: bool = True,
    sample_rate: float = 0.1,
) -> None:
    """
    Initialize observability stack.

    P1-3 FIX: Sets up OpenTelemetry with Cloud Trace/Monitoring exporters.
    P2-7 FIX: Configures structured logging with trace correlation.

    Args:
        service_name: Service name for traces/metrics
        enable_tracing: Enable distributed tracing
        enable_metrics: Enable metrics export
        enable_structured_logging: Enable structlog
        sample_rate: Trace sampling rate (0.0-1.0)

    Example:
        # Call once at application startup
        init_observability(service_name="emailops-production", sample_rate=0.1)
    """
    global _tracer, _meter, _initialized

    if _initialized:
        return

    config = EmailOpsConfig.load()

    # Guardrails for invalid sampling values
    sample_rate = max(0.0, min(sample_rate, 1.0))

    # P1-3 FIX: Initialize OpenTelemetry tracing
    if enable_tracing and OTEL_AVAILABLE and trace is not None:
        try:
            resource = Resource.create(
                {
                    "service.name": service_name,
                    "service.version": "1.0",
                    "deployment.environment": os.getenv("ENVIRONMENT", "development"),
                }
            )

            provider = TracerProvider(
                resource=resource,
                sampler=TraceIdRatioBased(sample_rate),
            )

            # Cloud Trace exporter for GCP
            if config.gcp.gcp_project:
                cloud_trace_exporter = CloudTraceSpanExporter(
                    project_id=config.gcp.gcp_project
                )
                provider.add_span_processor(BatchSpanProcessor(cloud_trace_exporter))

            trace.set_tracer_provider(provider)
            _tracer = trace.get_tracer(__name__)

            # Auto-instrument requests library
            RequestsInstrumentor().instrument()

            logging.info(
                "✓ OpenTelemetry tracing initialized (sample_rate=%.1f)", sample_rate
            )

        except Exception as e:
            logging.warning("Failed to initialize tracing: %s", e)

    # P2-3 FIX: Initialize metrics export
    if enable_metrics and OTEL_AVAILABLE and metrics is not None:
        try:
            resource = Resource.create({"service.name": service_name})

            # Cloud Monitoring exporter for GCP
            if config.gcp.gcp_project:
                exporter = CloudMonitoringMetricsExporter(
                    project_id=config.gcp.gcp_project
                )
                reader = PeriodicExportingMetricReader(
                    exporter, export_interval_millis=60000
                )
                provider = MeterProvider(resource=resource, metric_readers=[reader])
                metrics.set_meter_provider(provider)
                _meter = metrics.get_meter(__name__)

                logging.info("✓ Metrics export initialized")

        except Exception as e:
            logging.warning("Failed to initialize metrics: %s", e)

    # P2-7 FIX: Initialize structured logging
    if enable_structured_logging and STRUCTLOG_AVAILABLE and structlog is not None:
        try:
            structlog.configure(
                processors=[
                    structlog.contextvars.merge_contextvars,
                    structlog.processors.add_log_level,
                    structlog.processors.StackInfoRenderer(),
                    structlog.dev.set_exc_info,
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.JSONRenderer(),
                ],
                wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
                context_class=dict,
                logger_factory=structlog.PrintLoggerFactory(),
                cache_logger_on_first_use=True,
            )

            logging.info("✓ Structured logging initialized")

        except Exception as e:
            logging.warning("Failed to initialize structured logging: %s", e)

    _initialized = True


def trace_operation(operation_name: str, **span_attributes):
    """
    P1-3 FIX: Decorator to trace function execution.

    Creates a span for the operation with automatic timing and error capture.

    Args:
        operation_name: Name for the trace span
        **span_attributes: Additional span attributes

    Example:
        @trace_operation("index_conversation", component="indexing")
        def index_conversation(conv_dir):
            # Automatically traced
            pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not OTEL_AVAILABLE or _tracer is None:
                # Fallback: just execute without tracing
                return func(*args, **kwargs)

            with _tracer.start_as_current_span(operation_name) as span:
                # Add attributes
                for key, value in span_attributes.items():
                    span.set_attribute(key, value)

                # Add trace context to contextvars
                trace_ctx = {
                    "trace_id": format(span.get_span_context().trace_id, "032x"),
                    "span_id": format(span.get_span_context().span_id, "016x"),
                }
                token = _trace_context.set(trace_ctx)

                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("status", "success")
                    return result
                except Exception as e:
                    span.set_attribute("status", "error")
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    span.record_exception(e)
                    raise
                finally:
                    _trace_context.reset(token)

        return wrapper

    return decorator


def record_metric(
    metric_name: str,
    value: float | int,
    labels: dict[str, str] | None = None,
    metric_type: str = "counter",
) -> None:
    """
    P2-3 FIX: Record a metric for export.

    Args:
        metric_name: Metric name (e.g., "conversations_indexed")
        value: Metric value
        labels: Optional labels/dimensions
        metric_type: "counter", "gauge", or "histogram"

    Example:
        record_metric("api_calls", 1, {"endpoint": "answer_query", "status": "success"})
    """
    if not OTEL_AVAILABLE or _meter is None:
        return  # Silently skip if not available

    try:
        labels = labels or {}
        key = (metric_name, metric_type)

        if metric_type == "counter":
            counter = _metric_instruments.get(key)
            if counter is None:
                counter = _meter.create_counter(metric_name, unit="1")
                _metric_instruments[key] = counter
            counter.add(value, labels)
        elif metric_type == "gauge":
            gauge = _metric_instruments.get(key)
            if gauge is None:
                gauge = _meter.create_up_down_counter(metric_name, unit="1")
                _metric_instruments[key] = gauge
            gauge.add(value, labels)
        elif metric_type == "histogram":
            histogram = _metric_instruments.get(key)
            if histogram is None:
                histogram = _meter.create_histogram(metric_name, unit="ms")
                _metric_instruments[key] = histogram
            histogram.record(value, labels)
        else:
            logging.debug("Unknown metric type %s for %s", metric_type, metric_name)

    except Exception as e:
        logging.debug("Failed to record metric %s: %s", metric_name, e)


def get_logger(name: str) -> Any:
    """
    P2-7 FIX: Get structured logger with trace correlation.

    Returns structlog logger if available, otherwise standard logger.
    Logger automatically includes trace_id/span_id in all logs.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("processing_started", conv_id="conv123", file_count=5)
        # Output: {"event": "processing_started", "conv_id": "conv123", ...}
    """
    if STRUCTLOG_AVAILABLE and structlog is not None:
        logger = structlog.get_logger(name)
        # Bind trace context if available
        ctx = _trace_context.get() or {}
        if ctx:
            logger = logger.bind(
                trace_id=ctx.get("trace_id"), span_id=ctx.get("span_id")
            )
        return logger
    else:
        return logging.getLogger(name)


def get_trace_context() -> dict[str, str]:
    """
    Get current trace context for log correlation.

    Returns:
        Dict with trace_id and span_id (empty if no active trace)
    """
    return _trace_context.get() or {}
