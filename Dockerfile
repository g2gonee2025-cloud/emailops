# Production-ready container for EmailOps with Qwen AI integration
FROM python:3.11-slim

# Build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG USER_ID=1001
ARG GROUP_ID=1001

# Install system dependencies and create non-root user
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g ${GROUP_ID} emailops \
    && useradd -m -u ${USER_ID} -g emailops emailops

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY --chown=emailops:emailops requirements.txt ./

# Install Python dependencies as root (for system packages)
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn \
    && rm -rf /root/.cache/pip

# Copy application code
COPY --chown=emailops:emailops emailops ./emailops
COPY --chown=emailops:emailops README.md .
COPY --chown=emailops:emailops *.md ./

# Switch to non-root user
USER emailops

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LOG_LEVEL=INFO

# Create necessary directories
RUN mkdir -p /app/logs /app/data

# Healthcheck - tests if the Python module loads correctly
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import emailops; print('Healthcheck: OK')" || exit 1

# Expose port for API service (if running)
EXPOSE 8080

# Default command - can be overridden
CMD ["python", "-m", "emailops.email_indexer", "--help"]

# Alternative commands for different use cases:
# For indexing: CMD ["python", "-m", "emailops.email_indexer", "--root", "/data", "--provider", "qwen"]
# For API service: CMD ["gunicorn", "-b", "0.0.0.0:8080", "-w", "2", "--timeout", "300", "api_service:app"]
# For search: CMD ["python", "-m", "emailops.search_and_draft"]
