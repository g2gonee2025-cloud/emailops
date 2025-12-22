#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
# Load .env if present
if [ -f .env ]; then
  set -a
  source .env
  set +a
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
    CACHE_DIR="${HOME}/.sonar/cache"
    mkdir -p "$CACHE_DIR"
    docker run --rm \
        -e SONAR_HOST_URL="${SONAR_HOST_URL}" \
        -e SONAR_TOKEN="${SONAR_TOKEN}" \
        -v "${REPO_ROOT}:/usr/src" \
        -v "${CACHE_DIR}:/opt/sonar-scanner/.sonar/cache" \
        sonarsource/sonar-scanner-cli
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
get_prop() { grep -E "^$1=" "$REPORT_TASK_FILE" | head -n1 | cut -d= -f2-; }
serverUrl="$(get_prop serverUrl)"
ceTaskId="$(get_prop ceTaskId)"
ceTaskUrl="${serverUrl}/api/ce/task?id=${ceTaskId}"

# --- Step 3: Wait for Compute Engine (Processing) ---
echo "== 2. Waiting for Quality Gate check... =="
timeout=600 # 10 minutes max
deadline=$((SECONDS + timeout))

analysisId=""
while :; do
    if (( SECONDS > deadline )); then echo "Timeout waiting for Sonar processing."; exit 1; fi

    ce_json="$(curl -sS -u "${SONAR_TOKEN}:" "$ceTaskUrl")"
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
echo "DEBUG: analysisId='${analysisId}'"
qg_json="$(curl -sS -u "${SONAR_TOKEN}:" "${serverUrl}/api/qualitygates/project_status?analysisId=${analysisId}")"
echo "DEBUG: qg_json='${qg_json}'"

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
