@echo off
echo Setting up Vertex AI environment variables...

set GCP_PROJECT=semiotic-nexus-470620-f3
set GOOGLE_CLOUD_PROJECT=semiotic-nexus-470620-f3
set GCP_REGION=global
set GOOGLE_CLOUD_LOCATION=global
set EMBED_PROVIDER=vertex
set VERTEX_EMBED_MODEL=gemini-embedding-001
set GOOGLE_GENAI_USE_VERTEXAI=True

echo.
echo Environment variables set:
echo   GCP_PROJECT=%GCP_PROJECT%
echo   GCP_REGION=%GCP_REGION%
echo   EMBED_PROVIDER=%EMBED_PROVIDER%
echo   VERTEX_EMBED_MODEL=%VERTEX_EMBED_MODEL%
echo.

if "%1"=="preflight" (
    echo Running preflight check...
    python vertex_ai_preflight_check.py
) else if "%1"=="index" (
    echo Starting indexing...
    python -m emailops.email_indexer --root "C:\Users\ASUS\Desktop\Outlook"
) else (
    echo Usage:
    echo   setup_vertex_env.bat preflight  - Run preflight check
    echo   setup_vertex_env.bat index      - Start indexing
    echo   setup_vertex_env.bat            - Just set environment variables
)