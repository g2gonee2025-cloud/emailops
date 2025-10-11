#!/bin/bash

# Docker Installation Script for WSL2 (Ubuntu)
# This script installs Docker directly in WSL2 without requiring Docker Desktop

echo "=========================================="
echo "Docker Installation for WSL2 (Ubuntu)"
echo "=========================================="
echo ""

# Update package index
echo "1. Updating package index..."
sudo apt-get update

# Install prerequisites
echo "2. Installing prerequisites..."
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
echo "3. Adding Docker's official GPG key..."
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Set up the stable repository
echo "4. Setting up Docker repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Update package index again
echo "5. Updating package index with Docker repository..."
sudo apt-get update

# Install Docker Engine
echo "6. Installing Docker Engine..."
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add current user to docker group
echo "7. Adding user to docker group..."
sudo usermod -aG docker $USER

# Start Docker service
echo "8. Starting Docker service..."
sudo service docker start

# Enable Docker to start on boot
echo "9. Configuring Docker to start on boot..."
sudo systemctl enable docker 2>/dev/null || echo "Note: systemctl not fully supported in WSL2, Docker will need manual start"

# Test Docker installation
echo ""
echo "10. Testing Docker installation..."
sudo docker run hello-world

echo ""
echo "=========================================="
echo "Docker Installation Complete!"
echo "=========================================="
echo ""
echo "IMPORTANT NOTES:"
echo "1. You need to log out and log back in for group changes to take effect"
echo "   Run: exit"
echo "   Then: wsl -d Ubuntu"
echo ""
echo "2. To start Docker service in WSL2, run:"
echo "   sudo service docker start"
echo ""
echo "3. To verify Docker is working without sudo, run (after re-login):"
echo "   docker ps"
echo ""
echo "4. Docker Compose is installed as a plugin. Use:"
echo "   docker compose version"
echo ""
echo "For the EmailOps project:"
echo "   cd /mnt/c/Users/ASUS/Downloads/emailops_vertex_ai"
echo "   docker build -t emailops:latest ."
echo "=========================================="