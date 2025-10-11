# Docker and WSL2 Verification Script
# This script checks the status of Docker Desktop and WSL2 integration
# Run this script after installing Docker Desktop

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Docker Desktop & WSL2 Verification Script" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check 1: WSL2 Status
Write-Host "1. Checking WSL2 Installation..." -ForegroundColor Yellow
try {
    $wslStatus = wsl --list --verbose
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✓ WSL2 is installed" -ForegroundColor Green
        Write-Host ""
        Write-Host "   WSL Distributions:" -ForegroundColor Cyan
        wsl --list --verbose
        Write-Host ""
    } else {
        Write-Host "   ✗ WSL2 check failed" -ForegroundColor Red
    }
} catch {
    Write-Host "   ✗ WSL2 is not installed or not functioning" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
}

# Check 2: Docker Desktop Installation
Write-Host "2. Checking Docker Desktop Installation..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✓ Docker Desktop is installed" -ForegroundColor Green
        Write-Host "   Version: $dockerVersion" -ForegroundColor Cyan
        Write-Host ""
    } else {
        Write-Host "   ✗ Docker Desktop is not installed" -ForegroundColor Red
        Write-Host "   Please download from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
        Write-Host ""
    }
} catch {
    Write-Host "   ✗ Docker Desktop is not installed or not in PATH" -ForegroundColor Red
    Write-Host "   Please download from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    Write-Host ""
}

# Check 3: Docker Daemon Status
Write-Host "3. Checking Docker Daemon Status..." -ForegroundColor Yellow
try {
    $dockerInfo = docker info 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✓ Docker daemon is running" -ForegroundColor Green
        
        # Check for WSL2 backend
        if ($dockerInfo -match "WSL") {
            Write-Host "   ✓ Docker is using WSL2 backend" -ForegroundColor Green
        } else {
            Write-Host "   ⚠ Docker may not be using WSL2 backend" -ForegroundColor Yellow
            Write-Host "   Please check Docker Desktop Settings → General → Use WSL 2 based engine" -ForegroundColor Yellow
        }
        Write-Host ""
    } else {
        Write-Host "   ✗ Docker daemon is not running" -ForegroundColor Red
        Write-Host "   Please start Docker Desktop from the Start Menu" -ForegroundColor Yellow
        Write-Host ""
    }
} catch {
    Write-Host "   ✗ Cannot connect to Docker daemon" -ForegroundColor Red
    Write-Host "   Please ensure Docker Desktop is running" -ForegroundColor Yellow
    Write-Host ""
}

# Check 4: Docker in WSL
Write-Host "4. Checking Docker Availability in WSL..." -ForegroundColor Yellow
try {
    $wslDockerCheck = wsl docker --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✓ Docker is accessible from WSL" -ForegroundColor Green
        Write-Host "   Version: $wslDockerCheck" -ForegroundColor Cyan
        Write-Host ""
    } else {
        Write-Host "   ✗ Docker is not accessible from WSL" -ForegroundColor Red
        Write-Host "   Please enable WSL integration in Docker Desktop:" -ForegroundColor Yellow
        Write-Host "   Settings → Resources → WSL Integration → Enable Ubuntu" -ForegroundColor Yellow
        Write-Host ""
    }
} catch {
    Write-Host "   ✗ Cannot check Docker in WSL" -ForegroundColor Red
    Write-Host ""
}

# Check 5: Test Docker with Hello World
Write-Host "5. Testing Docker with Hello World Container..." -ForegroundColor Yellow
try {
    $helloWorldTest = docker run --rm hello-world 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✓ Docker test successful" -ForegroundColor Green
        Write-Host ""
    } else {
        Write-Host "   ✗ Docker test failed" -ForegroundColor Red
        Write-Host "   Error: $helloWorldTest" -ForegroundColor Red
        Write-Host ""
    }
} catch {
    Write-Host "   ✗ Docker test failed" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
}

# Check 6: Docker Compose
Write-Host "6. Checking Docker Compose..." -ForegroundColor Yellow
try {
    $composeVersion = docker compose version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✓ Docker Compose is available" -ForegroundColor Green
        Write-Host "   Version: $composeVersion" -ForegroundColor Cyan
        Write-Host ""
    } else {
        Write-Host "   ⚠ Docker Compose not available" -ForegroundColor Yellow
        Write-Host ""
    }
} catch {
    Write-Host "   ⚠ Docker Compose check failed" -ForegroundColor Yellow
    Write-Host ""
}

# Summary
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Verification Summary" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$allGood = $true
if (!(wsl --list --verbose 2>&1 | Select-String "Ubuntu")) {
    Write-Host "⚠ Action Required: WSL2 with Ubuntu needs to be properly configured" -ForegroundColor Yellow
    $allGood = $false
}

try {
    docker --version | Out-Null
    if ($LASTEXITCODE -ne 0) { throw }
} catch {
    Write-Host "⚠ Action Required: Install Docker Desktop from https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    $allGood = $false
}

try {
    docker info | Out-Null
    if ($LASTEXITCODE -ne 0) { throw }
} catch {
    Write-Host "⚠ Action Required: Start Docker Desktop" -ForegroundColor Yellow
    $allGood = $false
}

if ($allGood) {
    Write-Host "✓ All checks passed! Docker and WSL2 are properly configured." -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now use Docker commands in both PowerShell and WSL2." -ForegroundColor Cyan
    Write-Host "Project location: c:/Users/ASUS/Downloads/emailops_vertex_ai" -ForegroundColor Cyan
} else {
    Write-Host "Please address the issues above and run this script again." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "For detailed setup instructions, see: docs/DOCKER_WSL2_SETUP.md" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan