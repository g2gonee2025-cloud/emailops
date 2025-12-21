#!/usr/bin/env bash
set -euo pipefail

# Combined scan + quality gate check.
# This is the single command an agent can run repeatedly.
# Exits 0 only if Quality Gate is OK.
#
# Usage:
#   export SONAR_TOKEN="..."
#   export SONAR_HOST_URL="http://your-sonar-server:9000"
#   ./scripts/sonar_gate.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "STEP 1: Running SonarQube Scan"
echo "========================================"
"${SCRIPT_DIR}/sonar_scan.sh"

echo ""
echo "========================================"
echo "STEP 2: Checking Quality Gate"
echo "========================================"
"${SCRIPT_DIR}/sonar_qg.sh"
