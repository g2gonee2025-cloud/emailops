# Docker Desktop with WSL2 Setup Guide

## Current Status

### ✅ WSL2 Installation
**Status:** Installed and configured
- **Distribution:** Ubuntu
- **WSL Version:** 2 (WSL2)
- **State:** Currently stopped

### ❌ Docker Desktop Installation
**Status:** Not installed
- Docker command not found in PowerShell
- Requires manual installation

---

## Setup Instructions

### Step 1: Install Docker Desktop (MANUAL STEP REQUIRED)

Since Docker Desktop is not currently installed, you need to download and install it manually:

1. **Download Docker Desktop for Windows:**
   - Visit: https://www.docker.com/products/docker-desktop
   - Click "Download for Windows"
   - Wait for the installer to download (approximately 500MB)

2. **Run the Installer:**
   - Double-click the downloaded `Docker Desktop Installer.exe`
   - Ensure "Use WSL 2 instead of Hyper-V" option is **checked** during installation
   - Follow the installation wizard prompts
   - Click "Install" and wait for the process to complete

3. **Restart Your Computer:**
   - Docker Desktop installation requires a system restart
   - Save all your work and restart Windows

### Step 2: Configure Docker Desktop to Use WSL2

After restarting and launching Docker Desktop for the first time:

1. **Open Docker Desktop Settings:**
   - Right-click the Docker icon in the system tray
   - Click "Settings"

2. **Enable WSL2 Integration:**
   - Navigate to **Settings → General**
   - Ensure "Use the WSL 2 based engine" is checked
   - Click "Apply & Restart"

3. **Enable WSL2 Distribution Integration:**
   - Navigate to **Settings → Resources → WSL Integration**
   - Toggle on "Enable integration with my default WSL distro"
   - Enable "Ubuntu" specifically
   - Click "Apply & Restart"

### Step 3: Verify Docker Installation

After Docker Desktop starts successfully, run these commands in PowerShell to verify:

```powershell
# Check Docker version
docker --version

# Check Docker is running
docker info

# Test Docker with a simple container
docker run hello-world
```

### Step 4: Verify WSL2 Integration

Test Docker functionality from within WSL2:

```powershell
# Start WSL Ubuntu
wsl

# Inside WSL, check Docker
docker --version
docker ps
```

If Docker commands work in both PowerShell and WSL, the setup is complete.

---

## System Requirements

- **Windows 11** ✅ (Confirmed)
- **WSL 2** ✅ (Installed - Ubuntu distribution)
- **64-bit processor with SLAT** (Required for virtualization)
- **4GB RAM minimum** (8GB+ recommended)
- **Hardware virtualization enabled in BIOS**

---

## Troubleshooting

### Docker Desktop Won't Start

1. **Check WSL2 is running:**
   ```powershell
   wsl --status
   wsl --set-default-version 2
   ```

2. **Restart WSL:**
   ```powershell
   wsl --shutdown
   wsl
   ```

3. **Check Windows Features:**
   - Ensure "Virtual Machine Platform" is enabled
   - Ensure "Windows Subsystem for Linux" is enabled
   - Run in PowerShell (as Administrator):
   ```powershell
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   ```

### Docker Commands Not Working in WSL

1. **Restart Docker Desktop**
2. **Re-enable WSL integration:**
   - Docker Desktop → Settings → Resources → WSL Integration
   - Toggle off and on for Ubuntu
   - Apply & Restart

### Performance Issues

1. **Allocate more resources:**
   - Docker Desktop → Settings → Resources
   - Increase CPU and Memory limits
   - Apply & Restart

---

## Post-Installation Configuration

### Configure Docker for Development

1. **Set Resource Limits:**
   - Open Docker Desktop Settings → Resources
   - Recommended for development:
     - CPUs: 4-6 cores
     - Memory: 6-8 GB
     - Swap: 2 GB

2. **Configure File Sharing (if needed):**
   - Settings → Resources → File Sharing
   - Add the `c:/Users/ASUS/Downloads/emailops_vertex_ai` directory

3. **Enable BuildKit (recommended):**
   ```powershell
   # Add to your environment or .env file
   $env:DOCKER_BUILDKIT=1
   ```

---

## Next Steps for EmailOps Project

Once Docker is installed and verified, you can:

1. Build the EmailOps Docker image:
   ```bash
   docker build -t emailops:latest .
   ```

2. Run the EmailOps container:
   ```bash
   docker run -it --rm emailops:latest
   ```

3. Use Docker Compose (if configured):
   ```bash
   docker-compose up
   ```

Refer to `docs/DOCKER_USAGE.md` for project-specific Docker commands and workflows.

---

## Quick Reference Commands

```powershell
# WSL Commands
wsl --list --verbose              # List WSL distributions
wsl --status                      # Check WSL status
wsl --shutdown                    # Stop all WSL instances
wsl -d Ubuntu                     # Start Ubuntu distribution

# Docker Commands
docker --version                  # Check Docker version
docker info                       # Check Docker daemon info
docker ps                         # List running containers
docker images                     # List Docker images
docker system prune -a            # Clean up unused resources

# Docker Desktop
# Start: Launch "Docker Desktop" from Start Menu
# Stop: Right-click Docker icon → Quit Docker Desktop
```

---

## Documentation Links

- [Docker Desktop Documentation](https://docs.docker.com/desktop/windows/install/)
- [WSL2 Documentation](https://docs.microsoft.com/en-us/windows/wsl/install)
- [Docker WSL2 Backend](https://docs.docker.com/desktop/windows/wsl/)

---

**Setup Date:** 2025-10-10  
**Last Updated:** 2025-10-10  
**Prepared for:** EmailOps Vertex AI Project