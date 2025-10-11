# Docker Setup Options for WSL2

## Two Approaches for Docker on Windows 11 with WSL2

### Option 1: Docker Desktop (Not Required)
Docker Desktop provides a GUI and automatic WSL2 integration but requires a license for commercial use in larger organizations.
- **Documentation:** [`docs/DOCKER_WSL2_SETUP.md`](./DOCKER_WSL2_SETUP.md)
- **Verification Script:** [`setup/verify_docker_wsl2.ps1`](../setup/verify_docker_wsl2.ps1)

### Option 2: Native Docker in WSL2 (Recommended - Free)
Install Docker directly inside WSL2 Ubuntu without Docker Desktop. This is completely free and sufficient for development.

## Installing Native Docker in WSL2

### Quick Installation
1. Open WSL2 Ubuntu:
   ```powershell
   wsl -d Ubuntu
   ```

2. Run the installation script:
   ```bash
   cd /mnt/c/Users/ASUS/Downloads/emailops_vertex_ai
   chmod +x setup/install_docker_wsl2.sh
   ./setup/install_docker_wsl2.sh
   ```

### Manual Installation Steps
If you prefer to install manually:

```bash
# Update packages
sudo apt-get update

# Install prerequisites
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER

# Start Docker
sudo service docker start
```

### Post-Installation
1. Exit and restart WSL2:
   ```bash
   exit
   ```
   ```powershell
   wsl -d Ubuntu
   ```

2. Start Docker service (required each time WSL2 starts):
   ```bash
   sudo service docker start
   ```

3. Verify installation:
   ```bash
   docker --version
   docker ps
   docker run hello-world
   ```

## Using Docker with EmailOps Project

Once Docker is installed in WSL2, navigate to the project:

```bash
cd /mnt/c/Users/ASUS/Downloads/emailops_vertex_ai

# Build the EmailOps image
docker build -t emailops:latest .

# Run a container
docker run -it --rm emailops:latest

# Using Docker Compose (if docker-compose.yml exists)
docker compose up
```

## Comparison Table

| Feature | Docker Desktop | Native Docker in WSL2 |
|---------|---------------|---------------------|
| Cost | Free for personal/small business | Always free |
| GUI | Yes | No (CLI only) |
| Auto-start | Yes | Manual start required |
| WSL2 Integration | Automatic | Native |
| Resource Usage | Higher | Lower |
| Complexity | Easier | Slightly more setup |

## Troubleshooting

### Docker service not starting
```bash
# Check Docker status
sudo service docker status

# Start Docker manually
sudo service docker start

# Check for errors
sudo dockerd --debug
```

### Permission denied errors
```bash
# Ensure you're in docker group
groups

# Re-add to docker group
sudo usermod -aG docker $USER

# Logout and login again
exit
# Then: wsl -d Ubuntu
```

### WSL2 not starting
```powershell
# Restart WSL
wsl --shutdown
wsl -d Ubuntu
```

## Current Status
- ✅ WSL2 is installed with Ubuntu distribution
- ✅ Docker installation scripts and documentation prepared
- ⏳ Docker needs to be installed using one of the above methods

Choose the option that best fits your needs. For most development purposes, native Docker in WSL2 is sufficient and completely free.