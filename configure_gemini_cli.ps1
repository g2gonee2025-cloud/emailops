# Gemini Code CLI Configuration Script
# This script helps configure and optimize your Gemini CLI settings

param(
    [Parameter()]
    [ValidateSet("gca", "vertex-ai")]
    [string]$AuthType = "gca",

    [Parameter()]
    [switch]$BackupSettings,

    [Parameter()]
    [switch]$InstallVSCodeExtension,

    [Parameter()]
    [switch]$SetupAliases
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Gemini Code CLI Configuration Tool   " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Gemini CLI is installed
$geminiCmd = Get-Command gemini -ErrorAction SilentlyContinue
if (-not $geminiCmd) {
    Write-Host "[ERROR] Gemini CLI not found. Installing..." -ForegroundColor Red
    npm install -g @google/gemini-cli
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install Gemini CLI. Please install manually: npm install -g @google/gemini-cli" -ForegroundColor Red
        exit 1
    }
} else {
    $version = & gemini --version 2>$null
    Write-Host "[OK] Gemini CLI found (version: $version)" -ForegroundColor Green
}

# Paths
$geminiConfigDir = "$env:USERPROFILE\.gemini"
$settingsFile = "$geminiConfigDir\settings.json"

# Create config directory if it doesn't exist
if (-not (Test-Path $geminiConfigDir)) {
    New-Item -ItemType Directory -Force -Path $geminiConfigDir | Out-Null
    Write-Host "[CREATED] Gemini config directory" -ForegroundColor Yellow
}

# Backup existing settings if requested
if ($BackupSettings -and (Test-Path $settingsFile)) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupFile = "$settingsFile.backup_$timestamp"
    Copy-Item $settingsFile $backupFile
    Write-Host "[BACKUP] Backed up settings to: $backupFile" -ForegroundColor Green
}

# Create optimized settings
$settings = @{
    model = @{
        name = @{
            name = "gemini-2.5-pro"
        }
    }
    general = @{
        preferredEditor = "vscode"
        autoSave = $true
        contextWindow = "large"
    }
    ide = @{
        hasSeenNudge = $true
        vscodeIntegration = $true
    }
    auth = @{
        type = $AuthType
    }
    approvalMode = "auto_edit"
    context = @{
        fileFiltering = @{
            respectGitignore = $true
            excludePatterns = @(
                "*.pyc",
                "__pycache__",
                ".git",
                ".venv",
                "*.egg-info",
                "htmlcov",
                ".pytest_cache",
                ".scannerwork",
                ".sonarlint",
                ".ruff_cache",
                "*.log",
                "*.tmp"
            )
        }
        includeDirectories = @(
            "c:\Users\ASUS\Downloads\emailops_vertex_ai"
        )
        loadMemoryFromIncludeDirectories = $true
        maxFileSize = 1048576
        maxFiles = 500
    }
    features = @{
        codeCompletion = $true
        codeExplanation = $true
        testGeneration = $true
        refactoring = $true
        documentation = $true
    }
}

# Remove conflicting auth settings
if ($settings.security) {
    $settings.Remove("security")
}

# Write settings
$settings | ConvertTo-Json -Depth 10 | Set-Content $settingsFile -Force
Write-Host "[OK] Updated Gemini settings.json" -ForegroundColor Green
Write-Host "   Auth type: $AuthType" -ForegroundColor Gray

# Install VS Code extension if requested
if ($InstallVSCodeExtension) {
    Write-Host ""
    Write-Host "[INSTALL] Installing VS Code Gemini extension..." -ForegroundColor Cyan

    # Check if VS Code is installed
    $codeCmd = Get-Command code -ErrorAction SilentlyContinue
    if ($codeCmd) {
        code --install-extension google.gemini-cli-companion --force
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] VS Code extension installed" -ForegroundColor Green
        } else {
            Write-Host "[WARNING] Failed to install VS Code extension automatically" -ForegroundColor Yellow
            Write-Host "   Please install manually: Search for Gemini CLI Companion in VS Code Extensions" -ForegroundColor Yellow
        }
    } else {
        Write-Host "[WARNING] VS Code not found in PATH" -ForegroundColor Yellow
        Write-Host "   Please install the extension manually from VS Code" -ForegroundColor Yellow
    }
}

# Setup PowerShell aliases if requested
if ($SetupAliases) {
    Write-Host ""
    Write-Host "[SETUP] Setting up PowerShell aliases..." -ForegroundColor Cyan

    $profileContent = @"

# Gemini CLI Aliases and Functions
function Start-GeminiProject {
    param(
        [string]`$Query = ""
    )

    # Ensure conda environment if it exists
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        if (`$env:CONDA_DEFAULT_ENV -ne "emailops") {
            conda activate emailops 2>`$null
        }
    }

    # Start Gemini with optional query
    if (`$Query) {
        echo `$Query | gemini
    } else {
        gemini
    }
}

# Quick access aliases
Set-Alias -Name gai -Value Start-GeminiProject
Set-Alias -Name gemini-project -Value Start-GeminiProject

# Quick command functions
function Gemini-Explain {
    param([string]`$File)
    echo "/file `$File``nExplain this code in detail" | gemini
}

function Gemini-Test {
    param([string]`$File)
    echo "/file `$File``nGenerate comprehensive unit tests" | gemini
}

function Gemini-Optimize {
    param([string]`$File)
    echo "/file `$File``nOptimize this code for performance" | gemini
}

function Gemini-Doc {
    param([string]`$File)
    echo "/file `$File``nGenerate comprehensive documentation" | gemini
}
"@

    # Check if profile exists
    if (-not (Test-Path $PROFILE)) {
        New-Item -ItemType File -Force -Path $PROFILE | Out-Null
    }

    # Check if aliases already exist
    $existingContent = Get-Content $PROFILE -Raw -ErrorAction SilentlyContinue
    if ($existingContent -notmatch "Gemini CLI Aliases") {
        Add-Content -Path $PROFILE -Value $profileContent
        Write-Host "[OK] Added Gemini aliases to PowerShell profile" -ForegroundColor Green
        Write-Host "   Reload your shell or run: . `$PROFILE" -ForegroundColor Gray
    } else {
        Write-Host "[INFO] Gemini aliases already exist in profile" -ForegroundColor Yellow
    }
}

# Test authentication
Write-Host ""
Write-Host "[AUTH] Testing authentication..." -ForegroundColor Cyan

# Create a test command to check auth status
$testScript = @"
/auth status
/exit
"@

$testResult = $testScript | gemini 2>&1
if ($testResult -match "authenticated" -or $testResult -match "logged in") {
    Write-Host "[OK] Authentication is configured" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Authentication may need configuration" -ForegroundColor Yellow
    Write-Host "   Run gemini and use /auth login to authenticate" -ForegroundColor Gray
}

# Display summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "         Configuration Complete          " -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Quick Start Commands:" -ForegroundColor Yellow
Write-Host "  gemini              - Start interactive mode" -ForegroundColor Gray
Write-Host "  gemini --version    - Check version" -ForegroundColor Gray
Write-Host "  .\run_gemini_cli.bat - Use batch script" -ForegroundColor Gray

if ($SetupAliases) {
    Write-Host ""
    Write-Host "PowerShell Aliases (after reload):" -ForegroundColor Yellow
    Write-Host "  gai                 - Quick start Gemini" -ForegroundColor Gray
    Write-Host "  Gemini-Explain file - Explain code" -ForegroundColor Gray
    Write-Host "  Gemini-Test file    - Generate tests" -ForegroundColor Gray
    Write-Host "  Gemini-Optimize file- Optimize code" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Run gemini to start" -ForegroundColor Gray
Write-Host "2. Use /auth login if needed" -ForegroundColor Gray
Write-Host "3. Use /project . to set context" -ForegroundColor Gray
Write-Host "4. Start coding with AI assistance!" -ForegroundColor Gray
Write-Host ""
