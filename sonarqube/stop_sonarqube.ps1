# SonarQube stop script for Windows

Write-Host "Stopping SonarQube..." -ForegroundColor Yellow

# Navigate to the sonarqube directory
Set-Location $PSScriptRoot

# Stop SonarQube containers
docker compose down

Write-Host "SonarQube has been stopped." -ForegroundColor Green