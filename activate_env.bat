@echo off
:: Activate the local conda environment for EmailOps project

echo ===================================================
echo  EmailOps Qwen AI - Environment Activation
echo ===================================================
echo.

:: Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Conda is not installed or not in PATH
    echo Please install Anaconda/Miniconda first
    pause
    exit /b 1
)

:: Activate the local environment
echo Activating local conda environment...
call conda activate "%~dp0.conda"

if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate environment
    echo Creating new environment...
    call conda create --prefix "%~dp0.conda" python=3.11 -y
    call conda activate "%~dp0.conda"
    echo Installing requirements...
    pip install -r requirements.txt
) else (
    echo [SUCCESS] Environment activated: .conda
)

echo.
echo Environment Details:
python --version
echo.
echo Project: EmailOps with Qwen AI Integration
echo API: Using Qwen/Qwen3-Embedding-8B
echo.
echo Ready to use! You can now run:
echo   - python -m emailops.email_indexer
echo   - python -m emailops.search_and_draft
echo.

:: Keep the window open with the activated environment
cmd /k