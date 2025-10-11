#!/bin/bash

# SonarQube startup script for emailops_vertex_ai project

echo "Starting SonarQube for code quality analysis..."

# Navigate to the sonarqube directory
cd "$(dirname "$0")"

# Pull the latest images
echo "Pulling latest Docker images..."
docker compose pull

# Start SonarQube containers
echo "Starting SonarQube containers..."
docker compose up -d

# Wait for SonarQube to be ready
echo "Waiting for SonarQube to be ready..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:9000/api/system/status | grep -q "UP"; then
        echo "SonarQube is ready!"
        echo ""
        echo "==================================="
        echo "SonarQube is running!"
        echo "URL: http://localhost:9000"
        echo "Default credentials:"
        echo "  Username: admin"
        echo "  Password: admin"
        echo "==================================="
        echo ""
        echo "You'll be prompted to change the password on first login."
        break
    else
        echo "Waiting for SonarQube to start... (attempt $((attempt+1))/$max_attempts)"
        sleep 10
        attempt=$((attempt+1))
    fi
done

if [ $attempt -eq $max_attempts ]; then
    echo "SonarQube failed to start within the expected time."
    echo "Check the logs with: docker compose logs"
    exit 1
fi