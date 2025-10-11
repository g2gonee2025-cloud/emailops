# SonarQube Analysis Script using Docker
# This script runs code quality analysis using SonarQube Scanner Docker image

param(
    [string]$Token = "",
    [string]$ProjectKey = "emailops_vertex_ai"
)

Write-Host "Running SonarQube Analysis for EmailOps Vertex AI (Docker)" -ForegroundColor Green
Write-Host "=========================================================" -ForegroundColor Green

# Check if SonarQube is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9000/api/system/status" -UseBasicParsing -ErrorAction Stop
    if ($response.Content -notmatch "UP") {
        Write-Host "SonarQube is not running. Please start it first using start_sonarqube.ps1" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "SonarQube is not accessible. Please start it first using start_sonarqube.ps1" -ForegroundColor Red
    exit 1
}

# Navigate to project root
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

Write-Host "Project root: $projectRoot" -ForegroundColor Cyan
Write-Host "Using Docker image: sonarsource/sonar-scanner-cli" -ForegroundColor Cyan

# Convert Windows path to Unix path for Docker
$unixPath = $projectRoot -replace '\\', '/' -replace 'C:', '/mnt/c'

# Build Docker command
$dockerCmd = @(
    "docker", "run", "--rm",
    "--network=host",
    "-v", "${projectRoot}:/usr/src",
    "-w", "/usr/src",
    "sonarsource/sonar-scanner-cli"
)

# Add authentication token if provided
if ($Token) {
    $dockerCmd += "-Dsonar.login=$Token"
    Write-Host "Running analysis with authentication token..." -ForegroundColor Yellow
} else {
    Write-Host "Running analysis without authentication (first-time setup)..." -ForegroundColor Yellow
    Write-Host "Note: You'll need to log in to SonarQube first at http://localhost:9000" -ForegroundColor Yellow
    Write-Host "Default credentials: admin/admin (you'll be prompted to change)" -ForegroundColor Yellow
}

# Execute Docker command
Write-Host ""
Write-Host "Executing: $($dockerCmd -join ' ')" -ForegroundColor Gray
& $dockerCmd[0] $dockerCmd[1..($dockerCmd.Length-1)]

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Analysis completed successfully!" -ForegroundColor Green
    Write-Host "View results at: http://localhost:9000/dashboard?id=$ProjectKey" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "First time setup instructions:" -ForegroundColor Cyan
    Write-Host "1. Open http://localhost:9000 in your browser" -ForegroundColor White
    Write-Host "2. Log in with admin/admin" -ForegroundColor White
    Write-Host "3. Change the admin password when prompted" -ForegroundColor White
    Write-Host "4. Navigate to your project: $ProjectKey" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "Analysis failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "- Ensure SonarQube is fully started (may take 30-60 seconds)" -ForegroundColor White
    Write-Host "- Check if you need to authenticate first at http://localhost:9000" -ForegroundColor White
    Write-Host "- Verify Docker is running and can access the network" -ForegroundColor White
}