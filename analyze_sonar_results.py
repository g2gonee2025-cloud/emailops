import os

import requests

SONAR_TOKEN = os.environ.get("SONAR_TOKEN")
SONAR_HOST_URL = "http://localhost:9000"
PROJECT_KEY = "emailops-vertex-ai"

auth = (SONAR_TOKEN, "")


def get_hotspots():
    url = f"{SONAR_HOST_URL}/api/hotspots/search"
    params = {"projectKey": PROJECT_KEY, "status": "TO_REVIEW"}
    r = requests.get(url, auth=auth, params=params)
    r.raise_for_status()
    return r.json()


def get_coverage():
    url = f"{SONAR_HOST_URL}/api/measures/component_tree"
    params = {
        "component": PROJECT_KEY,
        "metricKeys": "new_coverage,new_lines_to_cover,new_uncovered_lines",
        "qualifiers": "FIL",
        "ps": 100,
    }
    r = requests.get(url, auth=auth, params=params)
    r.raise_for_status()
    return r.json()


def get_issues():
    url = f"{SONAR_HOST_URL}/api/issues/search"
    params = {
        "componentKeys": PROJECT_KEY,
        "resolved": "false",
    }
    r = requests.get(url, auth=auth, params=params)
    r.raise_for_status()
    # Filter for new code period manually if needed, or just print all
    return r.json()


def main():
    print("=== VIOLATIONS (New Code) ===")
    issues = get_issues()
    for issue in issues.get("issues", []):
        print(
            f"[{issue['type']}] {issue['component']}: {issue['message']} (Line {issue.get('line', '?')})"
        )

    print("\n=== SECURITY HOTSPOTS ===")
    hotspots = get_hotspots()
    for h in hotspots.get("hotspots", []):
        print(
            f"[{h['securityCategory']}] {h['component']}: {h['message']} (Line {h.get('line', '?')})"
        )
        print(f"  - Rule: {h['ruleKey']}")
        print(f"  - Key: {h['key']}")

    print("\n=== LOW COVERAGE FILES (New Code) ===")
    coverage = get_coverage()
    for comp in coverage.get("components", []):
        measures = {
            m["metric"]: m.get("period", {}).get("value", m.get("value"))
            for m in comp.get("measures", [])
        }
        new_lines = float(measures.get("new_lines_to_cover", 0))
        if new_lines > 0:
            cov = float(measures.get("new_coverage", 0))
            if cov < 80.0:
                print(f"{comp['path']}: {cov}% (Lines to cover: {int(new_lines)})")


if __name__ == "__main__":
    main()
