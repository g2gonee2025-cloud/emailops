# PowerShell script to activate the local conda environment

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  EmailOps - Environment Activation" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""

# Get the script directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$envPath = Join-Path $scriptPath ".conda"

# Check if conda is available
try {
    $condaInfo = conda info --json | ConvertFrom-Json
    Write-Host "[INFO] Conda found at: $($condaInfo.conda_prefix)" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Conda is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Anaconda/Miniconda first" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate the environment
Write-Host "Activating local conda environment..." -ForegroundColor Yellow

# Initialize conda for PowerShell if needed
& conda init powershell

# Activate the local environment
& conda activate "$envPath"

if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] Environment activated: .conda" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Environment not found, creating..." -ForegroundColor Yellow
    & conda create --prefix "$envPath" python=3.11 -y
    & conda activate "$envPath"
    Write-Host "Installing requirements..." -ForegroundColor Yellow
    & pip install -r requirements.txt
}

Write-Host ""
Write-Host "Environment Details:" -ForegroundColor Cyan
& python --version
Write-Host ""
Write-Host "Ready to use! You can now run:" -ForegroundColor Yellow
Write-Host "  - python -m emailops.email_indexer" -ForegroundColor White
Write-Host "  - python -m emailops.search_and_draft" -ForegroundColor White
Write-Host ""