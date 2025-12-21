#!/usr/bin/env bash
set -euo pipefail

: "${SONAR_TOKEN:?Set SONAR_TOKEN in your environment}"

# Optional: set REPORT_TASK_FILE to the exact path if your repo is unusual.
REPORT_TASK_FILE="${REPORT_TASK_FILE:-}"

if [[ -z "$REPORT_TASK_FILE" ]]; then
  # Common locations; if your scanner writes elsewhere, set REPORT_TASK_FILE explicitly.
  REPORT_TASK_FILE="$(
    find . -maxdepth 8 -type f -name report-task.txt \
      \( -path "*/.scannerwork/*" -o -path "*/target/sonar/*" -o -path "*/build/sonar/*" \) \
      2>/dev/null | head -n 1 || true
  )"
fi

if [[ -z "$REPORT_TASK_FILE" || ! -f "$REPORT_TASK_FILE" ]]; then
  echo "ERROR: report-task.txt not found. Run your Sonar analysis first." >&2
  exit 2
fi

get_prop() { grep -E "^$1=" "$REPORT_TASK_FILE" | head -n1 | cut -d= -f2-; }

serverUrl="$(get_prop serverUrl)"
ceTaskId="$(get_prop ceTaskId)"
ceTaskUrl="$(get_prop ceTaskUrl)"
projectKey="$(get_prop projectKey || true)"
branch="$(get_prop branch || true)"

if [[ -z "${serverUrl}" || -z "${ceTaskId}" ]]; then
  echo "ERROR: Could not parse serverUrl/ceTaskId from ${REPORT_TASK_FILE}" >&2
  exit 2
fi

if [[ -z "${ceTaskUrl}" ]]; then
  ceTaskUrl="${serverUrl}/api/ce/task?id=${ceTaskId}"
fi

deadline=$((SECONDS + ${SONAR_CE_TIMEOUT_SECONDS:-300}))
analysisId=""

while :; do
  if (( SECONDS > deadline )); then
    echo "ERROR: Timed out waiting for SonarQube Compute Engine task." >&2
    exit 2
  fi

  ce_json="$(curl -sS --fail -u "${SONAR_TOKEN}:" "$ceTaskUrl")"

  status="$(echo "$ce_json" | python3 - <<'PY'
import json,sys
d=json.load(sys.stdin)
print(d["task"]["status"])
PY
)"

  case "$status" in
    SUCCESS)
      analysisId="$(echo "$ce_json" | python3 - <<'PY'
import json,sys
d=json.load(sys.stdin)
print(d["task"].get("analysisId",""))
PY
)"
      break
      ;;
    FAILED|CANCELED)
      echo "ERROR: SonarQube CE task ended with status=$status" >&2
      exit 2
      ;;
    *)
      sleep 3
      ;;
  esac
done

if [[ -z "$analysisId" ]]; then
  echo "ERROR: analysisId missing from CE task response." >&2
  exit 2
fi

qg_json="$(curl -sS --fail -u "${SONAR_TOKEN}:" \
  "${serverUrl}/api/qualitygates/project_status?analysisId=${analysisId}")"

set +e
echo "$qg_json" | python3 - <<'PY'
import json,sys
d=json.load(sys.stdin)
ps=d["projectStatus"]
status=ps.get("status","UNKNOWN")
print(f"QUALITY_GATE={status}")
bad=[]
for c in ps.get("conditions",[]):
  if c.get("status") not in ("OK","NONE"):
    bad.append(c)
if bad:
  print("FAILING_CONDITIONS:")
  for c in bad:
    metric=c.get("metricKey")
    st=c.get("status")
    act=c.get("actualValue")
    thr=c.get("errorThreshold") or c.get("warningThreshold")
    print(f"- {metric}: {st} actual={act} threshold={thr}")
sys.exit(0 if status=="OK" else 1)
PY
qg_rc=$?
set -e

# Optional: print top issues to speed up fixing (best-effort; may require permissions).
if [[ $qg_rc -ne 0 && -n "${projectKey:-}" ]]; then
  limit="${SONAR_ISSUE_LIMIT:-20}"
  # URL-encode projectKey/branch
  pk="$(python3 - <<PY
import urllib.parse
print(urllib.parse.quote('''$projectKey'''))
PY
)"
  issues_url="${serverUrl}/api/issues/search?componentKeys=${pk}&resolved=false&ps=${limit}&p=1"
  if [[ -n "${branch:-}" ]]; then
    br="$(python3 - <<PY
import urllib.parse
print(urllib.parse.quote('''$branch'''))
PY
)"
    issues_url="${issues_url}&branch=${br}"
  fi

  set +e
  issues_json="$(curl -sS --fail -u "${SONAR_TOKEN}:" "$issues_url")"
  curl_rc=$?
  set -e

  if [[ $curl_rc -eq 0 ]]; then
    echo "$issues_json" | python3 - <<'PY'
import json,sys
d=json.load(sys.stdin)
issues=d.get("issues",[])
if not issues:
  print("TOP_ISSUES: none returned (or filtered/permissions).")
  sys.exit(0)
print("TOP_ISSUES:")
for i in issues:
  comp=i.get("component","")
  file=comp.split(":",1)[-1] if ":" in comp else comp
  line=i.get("line")
  rule=i.get("rule")
  sev=i.get("severity")
  msg=i.get("message","").replace("\n"," ").strip()
  loc=f"{file}:{line}" if line else file
  print(f"- [{sev}] {loc} ({rule}) {msg}")
PY
  fi
fi

exit $qg_rc
