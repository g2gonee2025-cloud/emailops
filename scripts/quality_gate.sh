#!/usr/bin/env bash
set -euo pipefail

# Put your repo's canonical local checks here (ideally copy from CI):
# Examples: lint, typecheck, unit tests, formatting check, build.
# You can hardcode them OR set LOCAL_QA_CMD in the environment.
LOCAL_QA_CMD="${LOCAL_QA_CMD:-}"

# Put your repo's canonical Sonar analysis command here (same as CI).
# Examples:
#   SONAR_SCAN_CMD="./gradlew sonarqube"
#   SONAR_SCAN_CMD="mvn -B test sonar:sonar"
#   SONAR_SCAN_CMD="sonar-scanner"
SONAR_SCAN_CMD="${SONAR_SCAN_CMD:-}"

if [[ -n "$LOCAL_QA_CMD" ]]; then
  echo "== Running local QA =="
  bash -lc "$LOCAL_QA_CMD"
fi

if [[ -z "$SONAR_SCAN_CMD" ]]; then
  echo "ERROR: Set SONAR_SCAN_CMD to your repo's Sonar analysis command." >&2
  exit 2
fi

echo "== Running Sonar analysis =="
bash -lc "$SONAR_SCAN_CMD"

echo "== Waiting for Quality Gate =="
./scripts/sonar_qg.sh
