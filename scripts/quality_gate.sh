#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF >&2
ERROR: Sonar analysis script not found or not executable.
Create an executable script at 'scripts/run_sonar.sh'
that contains your repo's Sonar analysis command.
e.g., echo "mvn sonar:sonar" > scripts/run_sonar.sh && chmod +x scripts/run_sonar.sh
EOF
  exit 1
}

# To run local checks, create an executable script at 'scripts/run_local_qa.sh'.
# This script can contain commands for linting, testing, etc.
LOCAL_QA_SCRIPT="$(dirname "$0")/run_local_qa.sh"
if [[ -x "$LOCAL_QA_SCRIPT" ]]; then
  echo "== Running local QA =="
  "$LOCAL_QA_SCRIPT"
fi

# To run Sonar analysis, create an executable script at 'scripts/run_sonar.sh'.
# This script should contain the command to run a Sonar scan.
SONAR_SCRIPT="$(dirname "$0")/run_sonar.sh"
if [[ ! -x "$SONAR_SCRIPT" ]]; then
  usage
fi

echo "== Running Sonar analysis =="
"$SONAR_SCRIPT"

echo "== Waiting for Quality Gate =="
"$(dirname "$0")/sonar_qg.sh"
