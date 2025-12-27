import json
import os


def check_successes(report_path):
    if not os.path.exists(report_path):
        print(f"Report file {report_path} not found.")
        return

    try:
        with open(report_path, "r") as f:
            results = json.load(f)

        # Determine structure
        if isinstance(results, dict) and "results" in results:
            results = results["results"]
        elif not isinstance(results, list):
            print(f"Unexpected JSON structure in {report_path}")
            return

        successes = [r for r in results if r.get("status") == "success"]

        print(f"Found {len(successes)} successful sessions in {report_path}:")
        for s in successes:
            print(f"- {s.get('file')} (Session ID: {s.get('session_id')})")

    except json.JSONDecodeError:
        print(f"Error decoding JSON from {report_path}")
    except Exception as e:
        print(f"Error reading report: {e}")


if __name__ == "__main__":
    check_successes("jules_batch_report.json")
    if os.path.exists("jules_retry_report.json"):
        print("\nChecking retry report...")
        check_successes("jules_retry_report.json")
