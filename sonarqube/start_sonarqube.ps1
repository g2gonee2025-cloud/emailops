# SonarQube startup script for Windows

Write-Host "Starting SonarQube for code quality analysis..." -ForegroundColor Green

# Navigate to the sonarqube directory
Set-Location $PSScriptRoot

# Pull the latest images
Write-Host "Pulling latest Docker images..." -ForegroundColor Yellow
docker compose pull

# Start SonarQube containers
Write-Host "Starting SonarQube containers..." -ForegroundColor Yellow
docker compose up -d

# Wait for SonarQube to be ready
Write-Host "Waiting for SonarQube to be ready..." -ForegroundColor Yellow
$maxAttempts = 30
$attempt = 0

while ($attempt -lt $maxAttempts) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:9000/api/system/status" -UseBasicParsing -ErrorAction SilentlyContinue
        if ($response.Content -match "UP") {
            Write-Host "`nSonarQube is ready!" -ForegroundColor Green
            Write-Host "===================================" -ForegroundColor Cyan
            Write-Host "SonarQube is running!" -ForegroundColor Green
            Write-Host "URL: http://localhost:9000" -ForegroundColor Yellow
            Write-Host "Default credentials:" -ForegroundColor Yellow
            Write-Host "  Username: admin" -ForegroundColor White
            Write-Host "  Password: admin" -ForegroundColor White
            Write-Host "===================================" -ForegroundColor Cyan
            Write-Host "`nYou'll be prompted to change the password on first login." -ForegroundColor Yellow
            break
        }
    } catch {
        # Ignore errors during startup
    }
    
    Write-Host "Waiting for SonarQube to start... (attempt $($attempt+1)/$maxAttempts)" -ForegroundColor Gray
    Start-Sleep -Seconds 10
    $attempt++
}

if ($attempt -eq $maxAttempts) {
    Write-Host "SonarQube failed to start within the expected time." -ForegroundColor Red
    Write-Host "Check the logs with: docker compose logs" -ForegroundColor Yellow
    exit 1
}