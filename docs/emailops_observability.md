# `emailops.observability`

**Primary Goal:** To provide a centralized, powerful, and easy-to-use framework for application observability. This module integrates industry-standard tools like OpenTelemetry and `structlog` to offer distributed tracing, metrics collection, and structured logging, all of which are essential for debugging, monitoring, and optimizing a production system.

## Directory Mapping

```
.
└── emailops/
    └── observability.py
```

---

## Core Components & Connections

This module is designed to be initialized once at application startup and then used throughout the codebase via its decorators and helper functions.

### `init_observability(...)`

- **Purpose:** This is the main setup function for the entire observability stack. It should be called once when the application starts.
- **Functionality:**
    - **Lazy Initialization:** It uses a global `_initialized` flag to ensure the setup process only runs once, even if called multiple times.
    - **Conditional Setup:** It's highly configurable. Based on its arguments and the availability of optional libraries (`OTEL_AVAILABLE`, `STRUCTLOG_AVAILABLE`), it can enable or disable tracing, metrics, and structured logging independently.
    - **Tracing Setup:** If enabled, it configures an OpenTelemetry `TracerProvider`. It sets up a `CloudTraceSpanExporter` to send trace data to Google Cloud Trace, allowing for visualization of request flows across the application. It also automatically instruments the `requests` library, so any outgoing HTTP calls are automatically included in the traces.
    - **Metrics Setup:** If enabled, it configures an OpenTelemetry `MeterProvider` with a `CloudMonitoringMetricsExporter` to send custom metrics to Google Cloud Monitoring.
    - **Logging Setup:** If enabled, it configures `structlog` to output logs as structured JSON, which is much easier for log aggregation systems to parse than plain text.
- **Connections:** This function is the root of the observability system. It needs to be called from the main entry point of the application (e.g., in `emailops.cli.main` or the main GUI application launcher).

### `@trace_operation(...)` Decorator

- **Purpose:** This is a decorator that makes it incredibly easy to add distributed tracing to any function.
- **Functionality:**
    - When a decorated function is called, it automatically starts a new OpenTelemetry `span`.
    - It records the function's name as the `operation_name` and adds any provided key-value pairs as attributes to the span.
    - **Trace Correlation:** It captures the `trace_id` and `span_id` of the current span and stores them in a `ContextVar` (`_trace_context`). This is a critical feature that allows logs to be correlated with traces.
    - **Automatic Error Handling:** It wraps the function call in a `try...except` block. If the function succeeds, it sets the span's status to "success." If it fails, it automatically records the exception details (type, message) on the span and sets the status to "error" before re-raising the exception.
- **Connections:** This decorator can be applied to any key function throughout the codebase (e.g., `feature_search_draft.draft_email_structured`, `indexing_main.build_corpus`) to get a detailed, hierarchical view of how a request flows through the system.

### `record_metric(...)`

- **Purpose:** A simple function for recording custom application metrics.
- **Functionality:** It uses the OpenTelemetry `Meter` to create and record a metric (e.g., a counter for API calls, a histogram for latency). It allows for attaching labels (dimensions) to the metric for powerful filtering and aggregation in the monitoring backend.
- **Connections:** This can be called from anywhere in the code to track business-level metrics, such as `record_metric("conversations_indexed", 1)`.

### `get_logger(...)`

- **Purpose:** To provide a logger that is automatically integrated with the observability stack.
- **Functionality:**
    - If `structlog` is enabled, it returns a `structlog` logger. Crucially, it automatically **binds the current trace context** (`trace_id` and `span_id`) to the logger.
    - This means any log message emitted by this logger will automatically include the `trace_id`, allowing developers to instantly find all logs related to a specific trace in their logging system.
    - If `structlog` is not available, it gracefully falls back to returning a standard Python `logging` instance.
- **Connections:** This function should be used in place of `logging.getLogger()` throughout the application to get the benefits of structured, correlated logging.

---

## Key Design Patterns

- **Decorator Pattern:** The `@trace_operation` decorator is a classic example of this pattern. It transparently adds tracing functionality to a function without modifying the function's own code.
- **Context Variable (`ContextVar`):** The use of `_trace_context` is a modern and robust way to manage context that needs to flow through an execution path, especially in asynchronous applications. It ensures that the correct `trace_id` is available for logging, even when code is running in different threads or async tasks.
- **Lazy Initialization:** The entire stack is initialized lazily and guarded by a flag, which is a safe and common pattern for setting up global resources.
- **Graceful Degradation:** The module is designed to work even if OpenTelemetry or `structlog` are not installed. In their absence, the functions become no-ops (they do nothing), allowing the core application logic to run without the observability features, which is useful for lightweight local development.