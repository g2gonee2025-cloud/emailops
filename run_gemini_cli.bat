@echo off
REM Script to run Gemini Code Assist CLI with conda environment

REM Add npm global bin to PATH
set PATH=%PATH%;C:\Users\ASUS\AppData\Roaming\npm

REM Activate conda environment if not already active
if "%CONDA_DEFAULT_ENV%" NEQ "emailops" (
    echo Activating emailops conda environment...
    call C:\Users\ASUS\anaconda3\Scripts\activate.bat emailops
)

REM Run Gemini CLI
echo Starting Gemini Code Assist CLI...
echo.
call C:\Users\ASUS\AppData\Roaming\npm\gemini.cmd %*
