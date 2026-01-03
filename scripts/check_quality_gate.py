import os
import sys

import requests

# Try to load .env file for local development.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("dotenv not installed, skipping loading .env file.")


SONAR_URL = os.environ.get("SONAR_URL", "http://localhost:9000")
SONAR_USER = os.environ.get("SONAR_USER")
SONAR_PASSWORD = os.environ.get("SONAR_PASSWORD")
PROJECT_KEY = os.environ.get("SONAR_PROJECT_KEY", "emailops-vertex-ai")

if not SONAR_USER or not SONAR_PASSWORD:
    print("SONAR_USER and SONAR_PASSWORD environment variables must be set.")
    sys.exit(1)

AUTH = (SONAR_USER, SONAR_PASSWORD)

if SONAR_URL.startswith("http://"):
    print("Warning: SONAR_URL is using insecure HTTP. Use HTTPS in production.")


def check_gate():
    try:
        r = requests.get(
            f"{SONAR_URL}/api/qualitygates/project_status?projectKey={PROJECT_KEY}",
            auth=AUTH,
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        project_status = data.get("projectStatus", {})
        status = project_status.get("status", "UNKNOWN")
        print(f"Quality Gate Status: {status}")

        # Print conditions
        for cond in project_status.get("conditions", []):
            s = cond.get("status")
            if s != "OK":
                metric_key = cond.get("metricKey", "N/A")
                actual_value = cond.get("actualValue", "N/A")
                error_threshold = cond.get("errorThreshold", "N/A")
                print(
                    f"  FAILED: {metric_key} = {actual_value} (Threshold: {error_threshold})"
                )
        return status
    except requests.exceptions.RequestException as e:
        print(f"Error checking gate: {e}")
        return "ERROR"


def list_bugs():
    print("\nFetching unresolved bugs...")
    all_issues = []
    page = 1
    while True:
        try:
            r = requests.get(
                f"{SONAR_URL}/api/issues/search?componentKeys={PROJECT_KEY}&resolved=false&types=BUG&p={page}",
                auth=AUTH,
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            issues = data.get("issues", [])
            all_issues.extend(issues)

            if len(issues) < 100:  # Assuming default page size is 100
                break
            page += 1

        except requests.exceptions.RequestException as e:
            print(f"Error checking issues: {e}")
            return

    total_issues = len(all_issues)
    print(f"Found {total_issues} issues.")

    for issue in all_issues:
        message = issue.get("message", "N/A")
        component = issue.get("component", "N/A")
        rule = issue.get("rule", "N/A")
        severity = issue.get("severity", "N/A")
        print(f"[{severity}] {component}: {message} ({rule})")


def show_overall_metrics():
    print("\n--- Overall Code Metrics ---")
    keys = "ncloc,coverage,duplicated_lines_density,bugs,vulnerabilities,code_smells,reliability_rating,security_rating,sqale_rating"
    try:
        r = requests.get(
            f"{SONAR_URL}/api/measures/component?component={PROJECT_KEY}&metricKeys={keys}",
            auth=AUTH,
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        component = data.get("component", {})
        measures = component.get("measures", [])
        for m in measures:
            metric = m.get("metric", "N/A")
            value = m.get("value", "N/A")
            print(f"{metric}: {value}")

    except requests.exceptions.RequestException as e:
        print(f"Error checking metrics: {e}")


if __name__ == "__main__":
    status = check_gate()
    show_overall_metrics()

    # list_bugs fetches unresolved bugs specifically.
    list_bugs()

    if status != "OK":
        print("\nQuality Gate Failed.")
        sys.exit(1)
    else:
        print("\nQuality Gate Passed.")
        sys.exit(0)
