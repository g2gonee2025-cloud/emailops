import requests

SONAR_URL = "http://localhost:9000"
AUTH = ("admin", "emailops-strong-password")
PROJECT_KEY = "emailops-vertex-ai"


def check_gate():
    try:
        r = requests.get(
            f"{SONAR_URL}/api/qualitygates/project_status?projectKey={PROJECT_KEY}",
            auth=AUTH,
        )
        if r.status_code != 200:
            print(f"Failed to get gate status: {r.text}")
            return "UNKNOWN"

        status = r.json()["projectStatus"]["status"]
        print(f"Quality Gate Status: {status}")

        # Print conditions
        for cond in r.json()["projectStatus"]["conditions"]:
            s = cond["status"]
            if s != "OK":
                print(
                    f"  FAILED: {cond['metricKey']} = {cond['actualValue']} (Threshold: {cond['errorThreshold']})"
                )

        return status
    except Exception as e:
        print(f"Error checking gate: {e}")
        return "ERROR"


def list_issues():
    print("\nFetching unresolved issues...")
    try:
        r = requests.get(
            f"{SONAR_URL}/api/issues/search?componentKeys={PROJECT_KEY}&resolved=false&types=BUG",
            auth=AUTH,
        )
        if r.status_code != 200:
            print(f"Failed to search issues: {r.text}")
            return

        issues = r.json()["issues"]
        print(f"Found {len(issues)} issues (limit 50 shown).")
        for issue in issues:
            msg = issue["message"]
            comp = issue["component"]
            rule = issue["rule"]
            severity = issue["severity"]
            print(f"[{severity}] {comp}: {msg} ({rule})")

    except Exception as e:
        print(f"Error checking issues: {e}")


def show_overall_metrics():
    print("\n--- Overall Code Metrics ---")
    keys = "ncloc,coverage,duplicated_lines_density,bugs,vulnerabilities,code_smells,reliability_rating,security_rating,sqale_rating"
    try:
        r = requests.get(
            f"{SONAR_URL}/api/measures/component?component={PROJECT_KEY}&metricKeys={keys}",
            auth=AUTH,
        )
        if r.status_code != 200:
            print(f"Failed to get measures: {r.text}")
            return

        measures = r.json()["component"]["measures"]
        for m in measures:
            metric = m["metric"]
            value = m["value"]
            print(f"{metric}: {value}")

    except Exception as e:
        print(f"Error checking metrics: {e}")


if __name__ == "__main__":
    status = check_gate()
    show_overall_metrics()

    print("\nListing Bugs:")
    # Re-using list_issues but forcing it to print bugs.
    # Actually list_issues queries everything unresolved.
    # Let's adjust list_issues to query bugs specifically if we want, but the existing query gets everything.
    # We'll just run list_issues unconditionally for now to see the bug.
    list_issues()

    if status != "OK":
        exit(1)
    else:
        print("\nQuality Gate Passed.")
        exit(0)
