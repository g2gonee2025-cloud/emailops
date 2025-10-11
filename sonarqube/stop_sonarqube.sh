#!/bin/bash

# SonarQube stop script for emailops_vertex_ai project

echo "Stopping SonarQube..."

# Navigate to the sonarqube directory
cd "$(dirname "$0")"

# Stop SonarQube containers
docker compose down

echo "SonarQube has been stopped."