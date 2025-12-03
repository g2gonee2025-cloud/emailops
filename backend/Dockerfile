# EmailOps Backend Production Dockerfile
# Optimized for DigitalOcean Kubernetes (DOKS)
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements
COPY backend/pyproject.toml ./
COPY requirements.txt ./requirements-root.txt

# Install Python dependencies
RUN pip install --no-cache-dir pip wheel setuptools
RUN pip wheel --no-cache-dir --wheel-dir /wheels -e .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-root.txt || true

# Production image
FROM python:3.11-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r emailops && useradd --no-log-init -r -g emailops emailops

WORKDIR /app

# Copy wheels from builder and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl || pip install --no-cache-dir psycopg2-binary redis httpx pydantic fastapi uvicorn google-cloud-aiplatform opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi python-dotenv

# Copy application code
COPY backend/src/ ./
COPY backend/migrations/ ./migrations/

# Create necessary directories
RUN mkdir -p /app/logs /secrets/gcp && chown -R emailops:emailops /app /secrets

# Switch to non-root user
USER emailops

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE 8000

# Run with uvicorn for production (2 workers to conserve memory)
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
