#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
# Load .env if present, securely
if [ -f .env ]; then
  # We only export the variables we know we need.
  while IFS='=' read -r line || [[ -n "$line" ]]; do
    # Strip comments and empty lines
    [[ "$line" =~ ^#.*$ ]] || [[ -z "$line" ]] && continue
    # Export known variables
    case "$line" in
      SONAR_*) export "$line" ;;
    esac
  done < .env
fi

# Ensure these are set in your environment (or .env file)
: "${SONAR_TOKEN:?Set SONAR_TOKEN}"
: "${SONAR_HOST_URL:?Set SONAR_HOST_URL}"

# Detect Python for JSON parsing
PYTHON=""
if command -v python3 >/dev/null 2>&1; then PYTHON="python3"; elif command -v python >/dev/null 2>&1; then PYTHON="python"; else echo "ERROR: python required"; exit 1; fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- Step 1: Run Analysis ---
echo "== 1. Running SonarQube Analysis =="

# Priority: 1. Custom Command, 2. Local Scanner, 3. Docker
if [[ -n "${SONAR_SCAN_CMD:-}" ]]; then
    bash -c "$SONAR_SCAN_CMD"
elif command -v sonar-scanner >/dev/null 2>&1; then
    sonar-scanner
elif command -v docker >/dev/null 2>&1; then
    # Docker fallback (Standard SonarSource Image)
    CACHE_DIR="${HOME:?HOME environment variable must be set}/.sonar/cache"
    mkdir -p "$CACHE_DIR"
    # Use --env-file for secrets
    DOCKER_ENV_FILE=$(mktemp)
    # shellcheck disable=SC2064
    trap 'rm -f "$DOCKER_ENV_FILE"' EXIT
    {
        echo "SONAR_HOST_URL=${SONAR_HOST_URL}"
        echo "SONAR_TOKEN=${SONAR_TOKEN}"
    } > "$DOCKER_ENV_FILE"

    docker run --rm \
        --env-file "$DOCKER_ENV_FILE" \
        -v "${REPO_ROOT}:/usr/src" \
        -v "${CACHE_DIR}:/opt/sonar-scanner/.sonar/cache" \
        sonarsource/sonar-scanner-cli@sha256:3c2242b135b597c413c6b2298e87d3a242a5c192d1a37c867290f1f2923984d7
else
    echo "ERROR: No sonar-scanner found. Install it or set SONAR_SCAN_CMD."
    exit 1
fi

# --- Step 2: Locate Report File ---
REPORT_TASK_FILE=""
# Check common locations
for p in ".scannerwork/report-task.txt" "target/sonar/report-task.txt" "build/sonar/report-task.txt" ".sonarqube/out/.sonar/report-task.txt"; do
    if [[ -f "${REPO_ROOT}/$p" ]]; then REPORT_TASK_FILE="${REPO_ROOT}/$p"; break; fi
done

if [[ -z "$REPORT_TASK_FILE" ]]; then
    echo "ERROR: report-task.txt not found. Scan failed?"
    exit 1
fi

# Extract Metadata
# We read the file line-by-line to avoid command injection vulnerabilities and handle missing keys gracefully.
serverUrl=""
ceTaskId=""
while IFS='=' read -r key value; do
    case "$key" in
        serverUrl) serverUrl="$value" ;;
        ceTaskId)  ceTaskId="$value" ;;
    esac
done < "$REPORT_TASK_FILE"

# Validate required properties
if [[ -z "$serverUrl" ]]; then
    echo "ERROR: Could not find 'serverUrl' in '$REPORT_TASK_FILE'" >&2
    exit 1
fi
if [[ -z "$ceTaskId" ]]; then
    echo "ERROR: Could not find 'ceTaskId' in '$REPORT_TASK_FILE'" >&2
    exit 1
fi

ceTaskUrl="${serverUrl}/api/ce/task?id=${ceTaskId}"

# --- Step 3: Wait for Compute Engine (Processing) ---
echo "== 2. Waiting for Quality Gate check... =="
timeout=600 # 10 minutes max
deadline=$((SECONDS + timeout))

analysisId=""
while :; do
    if (( SECONDS > deadline )); then echo "Timeout waiting for Sonar processing."; exit 1; fi

    ce_json="$(curl -sSf --netrc-file <(echo "machine $(hostname) login ${SONAR_TOKEN}") "$ceTaskUrl")"
    status="$($PYTHON -c "import json,sys; print(json.load(sys.stdin)['task']['status'])" <<< "$ce_json")"

    if [[ "$status" == "SUCCESS" ]]; then
        analysisId="$($PYTHON -c "import json,sys; print(json.load(sys.stdin)['task']['analysisId'])" <<< "$ce_json")"
        break
    elif [[ "$status" == "FAILED" || "$status" == "CANCELED" ]]; then
        echo "Sonar Processing Failed."
        exit 1
    fi
    sleep 3
done

# --- Step 4: Check Quality Gate Status ---
echo "== 3. Querying Quality Gate Result =="
qg_json="$(curl -sSf --netrc-file <(echo "machine $(hostname) login ${SONAR_TOKEN}") "${serverUrl}/api/qualitygates/project_status?analysisId=${analysisId}")"

# Parse Result and Print Failures
export QG_JSON="$qg_json"
$PYTHON - <<EOF
import json, sys, os
try:
    d = json.loads(os.environ["QG_JSON"])
    status = d["projectStatus"]["status"]
    print(f"QUALITY_GATE_STATUS={status}")

    if status != "OK":
        print("\n!!! QUALITY GATE FAILED !!!")
        print("Conditions causing failure:")
        for cond in d["projectStatus"].get("conditions", []):
            if cond["status"] not in ["OK", "NONE"]:
                print(f" - {cond['metricKey']}: {cond['status']} (Actual: {cond['actualValue']}, Threshold: {cond.get('errorThreshold', 'N/A')})")
        sys.exit(1)
    else:
        print("\n*** QUALITY GATE PASSED ***")
        sys.exit(0)
except Exception as e:
    print(f"Error parsing Quality Gate result: {e}")
    sys.exit(1)
EOF
