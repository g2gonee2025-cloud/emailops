# Production-ready container for EmailOps Vertex AI
FROM python:3.11-slim

# Build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG USER_ID=1001
ARG GROUP_ID=1001

# Install system dependencies and create non-root user
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
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

# Copy application structure
COPY --chown=emailops:emailops emailops ./emailops
COPY --chown=emailops:emailops processing ./processing
COPY --chown=emailops:emailops analysis ./analysis
COPY --chown=emailops:emailops diagnostics ./diagnostics
COPY --chown=emailops:emailops tests ./tests
COPY --chown=emailops:emailops setup ./setup
COPY --chown=emailops:emailops utils ./utils
COPY --chown=emailops:emailops ui ./ui
COPY --chown=emailops:emailops data ./data
COPY --chown=emailops:emailops docs ./docs
COPY --chown=emailops:emailops cli.py ./
COPY --chown=emailops:emailops *.md ./
COPY --chown=emailops:emailops .env.example ./

# Create necessary directories
RUN mkdir -p /app/logs /app/.streamlit

# Note: Streamlit config should be copied manually if needed
# or mounted as a volume at runtime

# Switch to non-root user
USER emailops

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LOG_LEVEL=INFO

# Healthcheck - tests if the CLI loads correctly
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python cli.py --help > /dev/null 2>&1 || exit 1

# Expose ports
EXPOSE 8080 8501

# Default command - show help
CMD ["python", "cli.py", "--help"]

# Alternative commands for different use cases:
# For indexing: CMD ["python", "cli.py", "index"]
# For UI: CMD ["python", "cli.py", "ui"]
# For monitoring: CMD ["python", "cli.py", "monitor"]
# For analysis: CMD ["python", "cli.py", "analyze", "--files"]
