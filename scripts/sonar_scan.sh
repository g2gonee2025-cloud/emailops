#!/usr/bin/env bash
set -euo pipefail

# CI-equivalent scan using official SonarSource Docker image.
# Uses same env vars as GitHub workflow (SONAR_TOKEN, SONAR_HOST_URL).
#
# Usage:
#   export SONAR_TOKEN="..."
#   export SONAR_HOST_URL="http://your-sonar-server:9000"
#   ./scripts/sonar_scan.sh

# Load .env file if it exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

: "${SONAR_TOKEN:?Set SONAR_TOKEN in .env or environment}"
: "${SONAR_HOST_URL:?Set SONAR_HOST_URL in .env or environment}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${SONAR_CACHE_DIR:-${HOME}/.sonar-cache}"

mkdir -p "$CACHE_DIR"

echo "Running SonarScanner via Docker..."
echo "  SONAR_HOST_URL: ${SONAR_HOST_URL}"
echo "  Repo root: ${REPO_ROOT}"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e SONAR_HOST_URL="${SONAR_HOST_URL}" \
  -e SONAR_TOKEN="${SONAR_TOKEN}" \
  -v "${CACHE_DIR}:/opt/sonar-scanner/.sonar/cache" \
  -v "${REPO_ROOT}:/usr/src" \
  sonarsource/sonar-scanner-cli

echo "Scan complete. report-task.txt should be in .scannerwork/"
