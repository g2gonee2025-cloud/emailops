# SonarQube Analysis Script for EmailOps Vertex AI
# This script runs code quality analysis using SonarQube Scanner

param(
    [string]$Token = "",
    [string]$ProjectKey = "emailops_vertex_ai"
)

Write-Host "Running SonarQube Analysis for EmailOps Vertex AI" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green

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

# Check if sonar-scanner is installed
$scannerPath = Get-Command sonar-scanner -ErrorAction SilentlyContinue
if (-not $scannerPath) {
    Write-Host "sonar-scanner is not installed or not in PATH" -ForegroundColor Red
    Write-Host ""
    Write-Host "To install sonar-scanner:" -ForegroundColor Yellow
    Write-Host "1. Download from: https://docs.sonarqube.org/latest/analyzing-source-code/scanners/sonarscanner/" -ForegroundColor White
    Write-Host "2. Extract to a directory (e.g., C:\sonar-scanner)" -ForegroundColor White
    Write-Host "3. Add the bin directory to your PATH" -ForegroundColor White
    Write-Host ""
    Write-Host "Alternatively, use Docker to run the scanner:" -ForegroundColor Yellow
    Write-Host "docker run --rm -v ${PWD}:/usr/src sonarsource/sonar-scanner-cli" -ForegroundColor White
    exit 1
}

# Navigate to project root
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

Write-Host "Project root: $projectRoot" -ForegroundColor Cyan

# Run analysis
if ($Token) {
    Write-Host "Running analysis with authentication token..." -ForegroundColor Yellow
    sonar-scanner -Dsonar.login=$Token
} else {
    Write-Host "Running analysis without authentication (first-time setup)..." -ForegroundColor Yellow
    Write-Host "Note: You may need to configure authentication after initial setup" -ForegroundColor Yellow
    sonar-scanner
}

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Analysis completed successfully!" -ForegroundColor Green
    Write-Host "View results at: http://localhost:9000/dashboard?id=$ProjectKey" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "Analysis failed with exit code: $LASTEXITCODE" -ForegroundColor Red
}