"""
Observability infrastructure with OpenTelemetry.

Implements §12 of the Canonical Blueprint.
"""

from __future__ import annotations

import atexit
import functools
import inspect
import logging
import os
import threading
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from cortex.config.loader import get_config

# Optional OpenTelemetry imports
try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.metrics import Observation
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
    from opentelemetry.trace import Status, StatusCode

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
REQUESTS_INSTRUMENTATION_ENV = "OUTLOOKCORTEX_OTEL_INSTRUMENT_REQUESTS"

AttributeValue = str | bool | int | float

__all__ = (
    "get_logger",
    "get_trace_context",
    "init_observability",
    "shutdown_observability",
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
_tracer_provider = None
_meter_provider = None
_metric_instruments: dict[tuple[str, str], Any] = {}
_gauge_instruments: dict[str, Any] = {}
_gauge_values: dict[tuple[str, tuple[tuple[str, AttributeValue], ...]], float] = {}
_initialized = False
_init_lock = threading.Lock()
_shutdown_registered = False
_metric_lock = threading.Lock()
_gauge_lock = threading.Lock()


def _parse_bool_env(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_gcp_project(config: Any) -> str | None:
    gcp_config = getattr(config, "gcp", None)
    return (
        getattr(gcp_config, "gcp_project", None)
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
    )


def _register_shutdown_hook() -> None:
    global _shutdown_registered
    if _shutdown_registered:
        return
    atexit.register(shutdown_observability)
    _shutdown_registered = True


def _labels_key(
    labels: dict[str, AttributeValue],
) -> tuple[tuple[str, AttributeValue], ...]:
    return tuple(sorted(labels.items(), key=lambda item: item[0]))


def _make_gauge_callback(metric_name: str) -> Callable[[Any], list[Observation]]:
    def callback(_: Any) -> list[Observation]:
        with _gauge_lock:
            return [
                Observation(value, dict(labels_key))
                for (name, labels_key), value in _gauge_values.items()
                if name == metric_name
            ]

    return callback


def _ensure_gauge_instrument(metric_name: str) -> None:
    if _meter is None:
        return
    if metric_name in _gauge_instruments:
        return
    with _metric_lock:
        if metric_name in _gauge_instruments:
            return
        if len(_metric_instruments) + len(_gauge_instruments) >= MAX_METRIC_INSTRUMENTS:
            logging.warning(
                "Metric instrument cache full (%d items). Cannot create gauge '%s'.",
                MAX_METRIC_INSTRUMENTS,
                metric_name,
            )
            return
        create_gauge = getattr(_meter, "create_observable_gauge", None)
        if not create_gauge:
            logging.warning(
                "Observable gauge not supported; skipping gauge '%s'.", metric_name
            )
            return
        _gauge_instruments[metric_name] = create_gauge(
            metric_name,
            callbacks=[_make_gauge_callback(metric_name)],
            unit="1",
        )


def _init_tracing(service_name: str, sample_rate: float, config: Any) -> Any:
    """Initialize OpenTelemetry tracing. Returns tracer or None.

    Requests instrumentation is global; disable via OUTLOOKCORTEX_OTEL_INSTRUMENT_REQUESTS.
    """
    if not OTEL_AVAILABLE or trace is None:
        return None

    global _tracer_provider
    try:
        env = getattr(getattr(config, "core", None), "env", "unknown")
        gcp_project = _get_gcp_project(config)
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
        has_exporter = False

        # Cloud Trace exporter for GCP
        if gcp_project:
            cloud_trace_exporter = CloudTraceSpanExporter(project_id=gcp_project)
            provider.add_span_processor(BatchSpanProcessor(cloud_trace_exporter))
            has_exporter = True
        elif OTLP_AVAILABLE:
            otlp_exporter = OTLPSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            has_exporter = True
            logging.info("✓ OTLP Trace Exporter configured")
        else:
            logging.warning(
                "Tracing enabled but no exporter configured; spans will not be exported."
            )

        trace.set_tracer_provider(provider)
        _tracer_provider = provider
        tracer = trace.get_tracer(__name__)

        # Auto-instrument requests library
        instrument_requests = _parse_bool_env(REQUESTS_INSTRUMENTATION_ENV, True)
        if instrument_requests and has_exporter:
            RequestsInstrumentor().instrument()
            logging.info("✓ Requests instrumentation enabled (global)")
        elif instrument_requests:
            logging.info("Requests instrumentation skipped (no exporter configured)")
        else:
            logging.info(
                "Requests instrumentation disabled via %s",
                REQUESTS_INSTRUMENTATION_ENV,
            )

        if has_exporter:
            logging.info(
                "✓ OpenTelemetry tracing initialized (sample_rate=%.1f)", sample_rate
            )
        return tracer

    except Exception:
        logging.warning("Failed to initialize tracing", exc_info=True)
        return None


def _init_metrics(service_name: str, config: Any) -> Any:
    """Initialize metrics export. Returns meter or None."""
    if not OTEL_AVAILABLE or metrics is None:
        return None

    global _meter_provider
    try:
        gcp_project = _get_gcp_project(config)
        resource = Resource.create({"service.name": service_name})

        # Cloud Monitoring exporter for GCP
        if gcp_project:
            exporter = CloudMonitoringMetricsExporter(project_id=gcp_project)
            reader = PeriodicExportingMetricReader(
                exporter, export_interval_millis=METRIC_EXPORT_INTERVAL_MS
            )
            provider = MeterProvider(resource=resource, metric_readers=[reader])
            metrics.set_meter_provider(provider)
            _meter_provider = provider
            logging.info("✓ Metrics export initialized (Cloud Monitoring)")
            return metrics.get_meter(__name__)

        if OTLP_AVAILABLE:
            exporter = OTLPMetricExporter()
            reader = PeriodicExportingMetricReader(
                exporter, export_interval_millis=METRIC_EXPORT_INTERVAL_MS
            )
            provider = MeterProvider(resource=resource, metric_readers=[reader])
            metrics.set_meter_provider(provider)
            _meter_provider = provider
            logging.info("✓ Metrics export initialized (OTLP)")
            return metrics.get_meter(__name__)

        logging.info("OTLP metrics exporter unavailable; metrics disabled")
        return None

    except Exception:
        logging.warning("Failed to initialize metrics", exc_info=True)
        return None


def _init_structured_logging() -> None:
    """Initialize structured logging with structlog."""
    if not STRUCTLOG_AVAILABLE or structlog is None:
        return

    try:
        # Standard structlog processing chain
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                # Add trace context to all log records
                _bind_trace_context,
                # Use JSON for machine-readable logs
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        logging.info("✓ Structured logging initialized")

    except Exception:
        logging.warning("Failed to initialize structured logging", exc_info=True)


def _bind_trace_context(logger, method_name, event_dict):
    """A structlog processor to inject the current trace context into log records."""
    ctx = _trace_context.get()
    if ctx:
        event_dict["trace_id"] = ctx.get("trace_id")
        event_dict["span_id"] = ctx.get("span_id")
    return event_dict


def init_observability(
    service_name: str = "cortex",
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

    with _init_lock:
        if _initialized:
            return

        try:
            config = get_config()
        except Exception:
            logging.exception("Failed to load config for observability")
            raise

        # Guardrails for invalid sampling values
        sample_rate = max(0.0, min(sample_rate, 1.0))

        if enable_tracing:
            _tracer = _init_tracing(service_name, sample_rate, config)

        if enable_metrics:
            _meter = _init_metrics(service_name, config)

        if enable_structured_logging:
            _init_structured_logging()

        _initialized = True
        _register_shutdown_hook()


def shutdown_observability() -> None:
    """Flush and shutdown OTel providers to avoid losing buffered telemetry."""
    if _tracer_provider is not None:
        try:
            _tracer_provider.shutdown()
        except Exception:
            logging.warning("Failed to shutdown tracer provider", exc_info=True)
    if _meter_provider is not None:
        try:
            _meter_provider.shutdown()
        except Exception:
            logging.warning("Failed to shutdown meter provider", exc_info=True)


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
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR))
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
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR))
                        span.record_exception(e)
                        raise
                    finally:
                        _trace_context.reset(token)

            return wrapper

    return decorator


def record_metric(
    metric_name: str,
    value: float | int,
    labels: dict[str, AttributeValue] | None = None,
    metric_type: str = "counter",
) -> None:
    """
    Record a metric for export.
    Thread-safe implementation.
    """
    if not OTEL_AVAILABLE or _meter is None:
        return  # Silently skip if not available

    try:
        attributes = labels or {}
        if metric_type == "gauge":
            _ensure_gauge_instrument(metric_name)
            if metric_name in _gauge_instruments:
                labels_key = _labels_key(attributes)
                with _gauge_lock:
                    _gauge_values[(metric_name, labels_key)] = float(value)
            return

        key = (metric_name, metric_type)

        # Fast path check
        instrument = _metric_instruments.get(key)

        if instrument is None:
            with _metric_lock:
                # Double-check locking pattern
                instrument = _metric_instruments.get(key)
                if instrument is None:
                    if (
                        len(_metric_instruments) + len(_gauge_instruments)
                        >= MAX_METRIC_INSTRUMENTS
                    ):
                        logging.warning(
                            "Metric instrument cache full (%d items). Cannot create new instrument for '%s'.",
                            MAX_METRIC_INSTRUMENTS,
                            metric_name,
                        )
                        return

                    if metric_type == "counter":
                        instrument = _meter.create_counter(metric_name, unit="1")
                    elif metric_type == "histogram":
                        instrument = _meter.create_histogram(metric_name, unit="ms")
                    else:
                        logging.warning(
                            "Unknown metric type '%s' for metric '%s'",
                            metric_type,
                            metric_name,
                        )
                        return

                    _metric_instruments[key] = instrument

        if instrument:
            if metric_type == "histogram":
                instrument.record(value, attributes)
            else:  # counter
                instrument.add(value, attributes)

    except Exception:
        logging.warning("Failed to record metric %s", metric_name, exc_info=True)


def get_logger(name: str) -> Any:
    """
    Get a logger.

    If structlog is available, returns a structured logger that will automatically
    include trace context. Otherwise, returns a standard library logger.

    Blueprint §12.3:
    * Automatically binds trace context via contextvars processor.
    """
    if STRUCTLOG_AVAILABLE and structlog is not None:
        return structlog.get_logger(name)
    return logging.getLogger(name)


def get_trace_context() -> dict[str, str]:
    """
    Get current trace context for log correlation.
    """
    return _trace_context.get() or {}
