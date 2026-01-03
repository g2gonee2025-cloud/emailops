#!/usr/bin/env bash
set -euo pipefail

# CI-equivalent scan using official SonarSource Docker image.
# Uses same env vars as GitHub workflow (SONAR_TOKEN, SONAR_HOST_URL).
#
# Usage:
#   export SONAR_TOKEN="..."
#   export SONAR_HOST_URL="http://your-sonar-server:9000"
#   ./scripts/sonar_scan.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load .env file if it exists
if [ -f "${REPO_ROOT}/.env" ]; then
    set +u
    set -a
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/.env"
    set +a
    set -u
fi

: "${SONAR_TOKEN:?Set SONAR_TOKEN in .env or environment}"
: "${SONAR_HOST_URL:?Set SONAR_HOST_URL in .env or environment}"

# --- Pre-flight checks ---
if ! command -v docker &> /dev/null; then
    echo "ERROR: 'docker' command not found. Please install Docker." >&2
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "ERROR: Docker daemon is not running or not accessible." >&2
    echo "Hint: Start Docker and ensure you have permissions to access it." >&2
    exit 1
fi

# --- Configuration ---
# Safely determine cache directory, checking HOME only if SONAR_CACHE_DIR is not set.
if [[ -z "${SONAR_CACHE_DIR-}" && -z "${HOME-}" ]]; then
    echo "ERROR: Cannot determine cache directory. Please set HOME or SONAR_CACHE_DIR." >&2
    exit 1
fi
CACHE_DIR="${SONAR_CACHE_DIR:-${HOME}/.sonar-cache}"

mkdir -p "$CACHE_DIR"

echo "Running SonarScanner via Docker..."
echo "  Repo root: ${REPO_ROOT}"
echo "  Cache dir: ${CACHE_DIR}"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -w /usr/src \
  -e SONAR_HOST_URL="${SONAR_HOST_URL}" \
  -e SONAR_TOKEN="${SONAR_TOKEN}" \
  -v "${CACHE_DIR}:/opt/sonar-scanner/.sonar/cache" \
  -v "${REPO_ROOT}:/usr/src" \
  sonarsource/sonar-scanner-cli:5.0.1 \
  -Dsonar.working.directory=.scannerwork

echo "Scan complete. report-task.txt should be in .scannerwork/"
