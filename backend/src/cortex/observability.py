"""
Observability infrastructure with OpenTelemetry.

Implements §12 of the Canonical Blueprint.
"""

from __future__ import annotations

import functools
import inspect
import logging
import threading
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, Dict

from cortex.config.loader import get_config

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

# Optional OTLP Exporters (for DigitalOcean/Generic)
try:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    OTLP_AVAILABLE = True
except ImportError:
    OTLP_AVAILABLE = False

# Optional structlog
try:
    import structlog

    STRUCTLOG_AVAILABLE = True
except ImportError:
    structlog = None  # type: ignore
    STRUCTLOG_AVAILABLE = False

# Constants
METRIC_EXPORT_INTERVAL_MS = 60000
MAX_METRIC_INSTRUMENTS = 1000

__all__ = (
    "get_logger",
    "get_trace_context",
    "init_observability",
    "record_metric",
    "trace_operation",
)

# Context variable for trace correlation
_trace_context: ContextVar[dict[str, str] | None] = ContextVar(
    "trace_context", default=None
)

# Global tracer and meter (initialized lazily)
_tracer = None
_meter = None
_metric_instruments: dict[tuple[str, str], Any] = {}
_initialized = False


def _init_tracing(service_name: str, sample_rate: float, config: Any) -> Any:
    """Initialize OpenTelemetry tracing. Returns tracer or None."""
    if not OTEL_AVAILABLE or trace is None:
        return None

    try:
        env = config.core.env if config.core else "unknown"
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": "1.0",
                "deployment.environment": env,
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
        elif OTLP_AVAILABLE:
            otlp_exporter = OTLPSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logging.info("✓ OTLP Trace Exporter configured")

        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer(__name__)

        # Auto-instrument requests library
        RequestsInstrumentor().instrument()

        logging.info(
            "✓ OpenTelemetry tracing initialized (sample_rate=%.1f)", sample_rate
        )
        return tracer

    except Exception as e:
        logging.warning("Failed to initialize tracing: %s", e)
        return None


def _init_metrics(service_name: str, config: Any) -> Any:
    """Initialize metrics export. Returns meter or None."""
    if not OTEL_AVAILABLE or metrics is None:
        return None

    try:
        resource = Resource.create({"service.name": service_name})

        # Cloud Monitoring exporter for GCP
        if config.gcp.gcp_project:
            exporter = CloudMonitoringMetricsExporter(project_id=config.gcp.gcp_project)
            reader = PeriodicExportingMetricReader(
                exporter, export_interval_millis=METRIC_EXPORT_INTERVAL_MS
            )
            provider = MeterProvider(resource=resource, metric_readers=[reader])
            metrics.set_meter_provider(provider)
            logging.info("✓ Metrics export initialized (Cloud Monitoring)")
            return metrics.get_meter(__name__)

        if OTLP_AVAILABLE:
            exporter = OTLPMetricExporter()
            reader = PeriodicExportingMetricReader(
                exporter, export_interval_millis=METRIC_EXPORT_INTERVAL_MS
            )
            provider = MeterProvider(resource=resource, metric_readers=[reader])
            metrics.set_meter_provider(provider)
            logging.info("✓ Metrics export initialized (OTLP)")
            return metrics.get_meter(__name__)

        logging.info("OTLP metrics exporter unavailable; metrics disabled")
        return None

    except Exception as e:
        logging.warning("Failed to initialize metrics: %s", e)
        return None


def _init_structured_logging() -> None:
    """Initialize structured logging with structlog."""
    if not STRUCTLOG_AVAILABLE or structlog is None:
        return

    try:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
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


def init_observability(
    service_name: str = "emailops",
    enable_tracing: bool = True,
    enable_metrics: bool = True,
    enable_structured_logging: bool = True,
    sample_rate: float = 0.1,
) -> None:
    """
    Initialize observability stack.

    Blueprint §12.3:
    * Initialize OTel tracing/metrics
    * Configure structured logging
    """
    global _tracer, _meter, _initialized

    if _initialized:
        return

    config = get_config()

    # Guardrails for invalid sampling values
    sample_rate = max(0.0, min(sample_rate, 1.0))

    if enable_tracing:
        _tracer = _init_tracing(service_name, sample_rate, config)

    if enable_metrics:
        _meter = _init_metrics(service_name, config)

    if enable_structured_logging:
        _init_structured_logging()

    _initialized = True


def trace_operation(operation_name: str, **span_attributes):
    """
    Decorator to trace function execution.

    Blueprint §12.3:
    * Starts span
    * Binds trace context
    * Records exceptions
    """

    def decorator(func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not OTEL_AVAILABLE or _tracer is None:
                    # Fallback: just execute without tracing
                    return await func(*args, **kwargs)

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
                        result = await func(*args, **kwargs)
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

            return async_wrapper
        else:

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


_metric_lock = threading.Lock()

# ... (global vars)


def record_metric(
    metric_name: str,
    value: float | int,
    labels: Dict[str, str] | None = None,
    metric_type: str = "counter",
) -> None:
    """
    Record a metric for export.
    Thread-safe implementation.
    """
    if not OTEL_AVAILABLE or _meter is None:
        return  # Silently skip if not available

    try:
        labels = labels or {}
        key = (metric_name, metric_type)

        # Fast path check
        instrument = _metric_instruments.get(key)

        if instrument is None:
            with _metric_lock:
                # Double-check locking pattern
                instrument = _metric_instruments.get(key)
                if instrument is None:
                    # Prevent unbounded growth
                    if len(_metric_instruments) > MAX_METRIC_INSTRUMENTS:
                        # Simple eviction: clear all.
                        # Ideally we'd use LRU, but for bulk review constraint fixes this is sufficient safety.
                        _metric_instruments.clear()
                        logging.warning(
                            "Metrics instrument cache cleared (exceeded %d items)",
                            MAX_METRIC_INSTRUMENTS,
                        )

                    if metric_type == "counter":
                        instrument = _meter.create_counter(metric_name, unit="1")
                    elif metric_type == "gauge":
                        # OTel doesn't support 'gauge' directly in this version, usually up_down_counter
                        instrument = _meter.create_up_down_counter(
                            metric_name, unit="1"
                        )
                    elif metric_type == "histogram":
                        instrument = _meter.create_histogram(metric_name, unit="ms")
                    else:
                        logging.debug(
                            "Unknown metric type %s for %s", metric_type, metric_name
                        )
                        return

                    if instrument:
                        _metric_instruments[key] = instrument

        if instrument:
            if metric_type == "histogram":
                instrument.record(value, labels)
            else:
                instrument.add(value, labels)

    except Exception as e:
        logging.warning("Failed to record metric %s: %s", metric_name, e)


def get_logger(name: str) -> Any:
    """
    Get structured logger with trace correlation.

    Blueprint §12.3:
    * Automatically binds trace context
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
    """
    return _trace_context.get() or {}
