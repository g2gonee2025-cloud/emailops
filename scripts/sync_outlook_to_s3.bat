@echo off
REM ============================================
REM EmailOps S3 Sync Script for Windows
REM Syncs local Outlook folder to DigitalOcean Spaces
REM ============================================

SET RCLONE_DIR=%~dp0
SET SOURCE_DIR=C:\Users\ASUS\Desktop\Outlook
SET DEST=do-spaces-nyc3:emailops-bucket/Outlook

echo ============================================
echo  EmailOps S3 Sync
echo ============================================
echo Source: %SOURCE_DIR%
echo Destination: %DEST%
echo.

REM Check if source exists
if not exist "%SOURCE_DIR%" (
    echo ERROR: Source directory not found: %SOURCE_DIR%
    pause
    exit /b 1
)

REM Run rclone sync with progress
echo Starting sync...
"%RCLONE_DIR%rclone.exe" --config "%RCLONE_DIR%rclone.conf" sync "%SOURCE_DIR%" %DEST% --progress --transfers 8 --checkers 16

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================
    echo  Sync completed successfully!
    echo ============================================
) else (
    echo.
    echo ============================================
    echo  Sync failed with error code: %ERRORLEVEL%
    echo ============================================
)

pause
