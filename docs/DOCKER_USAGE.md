# 🐳 Docker Usage Guide for EmailOps Vertex AI

This guide provides comprehensive instructions for building and running the EmailOps Vertex AI application using Docker.

## 📋 Prerequisites

1. **Docker Installed**: Ensure Docker is installed on your system
   - Windows: [Docker Desktop for Windows](https://docs.docker.com/desktop/windows/install/)
   - Mac: [Docker Desktop for Mac](https://docs.docker.com/desktop/mac/install/)
   - Linux: [Docker Engine](https://docs.docker.com/engine/install/)

2. **Google Cloud Credentials**: Set up your Google Cloud credentials for Vertex AI access

3. **Environment Variables**: Create a `.env` file from `.env.example`:
   ```bash
   cp .env.example .env
   # Edit .env with your actual values
   ```

## 🔨 Building the Docker Image

### Basic Build
```bash
# Build the image with default settings
docker build -t emailops-vertex-ai:latest .
```

### Build with Custom User ID (for Linux/Mac)
```bash
# Build with your current user ID to avoid permission issues
docker build \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  -t emailops-vertex-ai:latest .
```

### Build with No Cache (Fresh Build)
```bash
docker build --no-cache -t emailops-vertex-ai:latest .
```

## 🚀 Running the Container

### 1. Basic Usage - Show Help
```bash
docker run --rm emailops-vertex-ai:latest
```

### 2. Interactive Shell
```bash
# Access the container shell for debugging
docker run -it --rm \
  --entrypoint /bin/bash \
  emailops-vertex-ai:latest
```

### 3. Running CLI Commands

#### Index Emails
```bash
docker run --rm \
  -v $(pwd)/.env:/app/.env:ro \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  emailops-vertex-ai:latest \
  python cli.py index
```

#### Run Analysis
```bash
docker run --rm \
  -v $(pwd)/logs:/app/logs \
  emailops-vertex-ai:latest \
  python cli.py analyze --files
```

#### Monitor Indexing
```bash
docker run --rm \
  -v $(pwd)/logs:/app/logs \
  emailops-vertex-ai:latest \
  python cli.py monitor
```

### 4. Running the Streamlit UI
```bash
docker run -d \
  --name emailops-ui \
  -p 8501:8501 \
  -v $(pwd)/.env:/app/.env:ro \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  emailops-vertex-ai:latest \
  python cli.py ui
```

Access the UI at: http://localhost:8501

### 5. Running with Google Cloud Credentials

#### Option A: Using Service Account Key File
```bash
docker run --rm \
  -v $(pwd)/.env:/app/.env:ro \
  -v $(pwd)/service-account-key.json:/app/credentials.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  emailops-vertex-ai:latest \
  python cli.py index
```

#### Option B: Using ADC (Application Default Credentials)
```bash
# First, authenticate locally
gcloud auth application-default login

# Then run with mounted credentials
docker run --rm \
  -v ~/.config/gcloud:/home/emailops/.config/gcloud:ro \
  -v $(pwd)/.env:/app/.env:ro \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  emailops-vertex-ai:latest \
  python cli.py index
```

## 📦 Docker Compose Setup

Create a `docker-compose.yml` file for easier management:

```yaml
version: '3.8'

services:
  emailops:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        USER_ID: ${USER_ID:-1001}
        GROUP_ID: ${GROUP_ID:-1001}
    image: emailops-vertex-ai:latest
    container_name: emailops-app
    env_file:
      - .env
    environment:
      - GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./.env:/app/.env:ro
      - ./service-account-key.json:/app/credentials.json:ro
      # Optional: Mount Google Cloud config for ADC
      # - ~/.config/gcloud:/home/emailops/.config/gcloud:ro
    command: ["python", "cli.py", "--help"]
    
  emailops-ui:
    image: emailops-vertex-ai:latest
    container_name: emailops-ui
    ports:
      - "8501:8501"
    env_file:
      - .env
    environment:
      - GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./.env:/app/.env:ro
      - ./service-account-key.json:/app/credentials.json:ro
    command: ["python", "cli.py", "ui"]
    depends_on:
      - emailops
```

### Docker Compose Commands

```bash
# Build the services
docker-compose build

# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d emailops-ui

# View logs
docker-compose logs -f emailops-ui

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Run a one-off command
docker-compose run --rm emailops python cli.py analyze --files
```

## 🛠️ Development Mode

For development with live code reloading:

```bash
docker run -it --rm \
  -v $(pwd):/app \
  -v $(pwd)/.env:/app/.env:ro \
  -p 8501:8501 \
  --name emailops-dev \
  emailops-vertex-ai:latest \
  /bin/bash
```

Inside the container:
```bash
# Install additional dev tools if needed
pip install ipython pytest

# Run commands
python cli.py --help
python cli.py index
python cli.py ui
```

## 🔍 Debugging

### View Container Logs
```bash
# View running containers
docker ps

# View logs
docker logs emailops-ui

# Follow logs
docker logs -f emailops-ui

# View last 100 lines
docker logs --tail 100 emailops-ui
```

### Execute Commands in Running Container
```bash
# Access shell in running container
docker exec -it emailops-ui /bin/bash

# Run specific command
docker exec emailops-ui python cli.py analyze --stats
```

### Health Check
```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' emailops-ui
```

## 🌐 Environment Variables

Key environment variables for Docker:

```bash
# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json
GOOGLE_CLOUD_PROJECT=your-project-id

# Application
LOG_LEVEL=INFO
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1

# Vertex AI
VERTEX_AI_LOCATION=us-central1
VERTEX_AI_INDEX_ENDPOINT=your-index-endpoint
```

## 🚨 Common Issues and Solutions

### 1. Permission Denied Errors
```bash
# Build with your user ID
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t emailops-vertex-ai:latest .
```

### 2. Cannot Connect to Google Cloud
```bash
# Ensure credentials are mounted
-v ./service-account-key.json:/app/credentials.json:ro \
-e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json
```

### 3. Port Already in Use
```bash
# Use a different port
docker run -p 8502:8501 ...
```

### 4. Out of Memory
```bash
# Increase Docker memory limit in Docker Desktop settings
# Or use docker run with memory limits
docker run --memory="4g" --memory-swap="4g" ...
```

## 🧹 Cleanup

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune

# Remove all emailops containers
docker rm $(docker ps -a -q --filter "ancestor=emailops-vertex-ai:latest")

# Complete cleanup (careful!)
docker system prune -a --volumes
```

## 📊 Resource Monitoring

```bash
# Monitor container resource usage
docker stats emailops-ui

# Check disk usage
docker system df
```

## 🔐 Security Best Practices

1. **Never commit credentials** to the repository
2. **Use read-only mounts** (`:ro`) for sensitive files
3. **Run as non-root user** (already configured in Dockerfile)
4. **Limit container resources** in production
5. **Use specific image tags** instead of `latest` in production
6. **Scan images for vulnerabilities**:
   ```bash
   docker scan emailops-vertex-ai:latest
   ```

## 📝 Example Workflows

### Complete Indexing Workflow
```bash
# 1. Build the image
docker build -t emailops-vertex-ai:latest .

# 2. Run indexing
docker run --rm \
  -v $(pwd)/.env:/app/.env:ro \
  -v $(pwd)/service-account-key.json:/app/credentials.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  emailops-vertex-ai:latest \
  python cli.py index

# 3. Monitor progress
docker run --rm \
  -v $(pwd)/logs:/app/logs:ro \
  emailops-vertex-ai:latest \
  python cli.py monitor

# 4. Analyze results
docker run --rm \
  -v $(pwd)/logs:/app/logs:ro \
  -v $(pwd)/data:/app/data:ro \
  emailops-vertex-ai:latest \
  python cli.py analyze --files
```

### UI Development Workflow
```bash
# Start UI with auto-reload
docker run -d \
  --name emailops-ui-dev \
  -p 8501:8501 \
  -v $(pwd):/app \
  -v $(pwd)/.env:/app/.env:ro \
  -e PYTHONPATH=/app \
  emailops-vertex-ai:latest \
  python cli.py ui
```

## 🎯 Quick Reference

| Command | Purpose |
|---------|---------|
| `docker build -t emailops-vertex-ai:latest .` | Build image |
| `docker run --rm emailops-vertex-ai:latest` | Show help |
| `docker run -it --rm emailops-vertex-ai:latest /bin/bash` | Interactive shell |
| `docker run -p 8501:8501 emailops-vertex-ai:latest python cli.py ui` | Run UI |
| `docker logs -f container-name` | View logs |
| `docker exec -it container-name /bin/bash` | Access running container |
| `docker-compose up -d` | Start with compose |
| `docker system prune -a` | Clean everything |

---

For more information, see the main [README.md](../README.md) or consult the [Docker documentation](https://docs.docker.com/).