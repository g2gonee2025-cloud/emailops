# Production-ready container for EmailOps Cortex
# This is the unified container for CLI, workers, and UI
FROM python:3.11-slim as builder

# Build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements
COPY requirements.txt ./
COPY backend/pyproject.toml ./backend-pyproject.toml

# Install Python dependencies
RUN pip install --no-cache-dir pip wheel setuptools
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt || true

# Production image
FROM python:3.11-slim

ARG USER_ID=1001
ARG GROUP_ID=1001

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g ${GROUP_ID} emailops \
    && useradd -m -u ${USER_ID} -g emailops emailops

WORKDIR /app

# Copy wheels from builder and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl 2>/dev/null || \
    pip install --no-cache-dir psycopg2-binary redis httpx pydantic fastapi uvicorn \
    google-cloud-aiplatform opentelemetry-api opentelemetry-sdk python-dotenv \
    gunicorn

# Copy Cortex application structure
COPY --chown=emailops:emailops backend/src/ ./backend/src/
COPY --chown=emailops:emailops backend/migrations/ ./backend/migrations/
COPY --chown=emailops:emailops cli/src/ ./cli/src/
COPY --chown=emailops:emailops workers/src/ ./workers/src/
COPY --chown=emailops:emailops ui/ ./ui/
COPY --chown=emailops:emailops docs/ ./docs/
COPY --chown=emailops:emailops tests/ ./tests/
COPY --chown=emailops:emailops *.md ./

# Create necessary directories
RUN mkdir -p /app/logs /secrets/gcp && chown -R emailops:emailops /app /secrets

# Switch to non-root user
USER emailops

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/backend/src:/app/cli/src:/app/workers/src \
    LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from cortex.config.loader import get_config; get_config()" || exit 1

# Expose ports
EXPOSE 8000 8501

# Default command - run the API
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Alternative commands:
# For CLI: CMD ["python", "-m", "cortex_cli.main"]
# For Workers: CMD ["python", "-m", "workers.src.main"]
# For UI (Streamlit): CMD ["streamlit", "run", "ui/emailops_ui.py", "--server.port", "8501"]
