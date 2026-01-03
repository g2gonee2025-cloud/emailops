import argparse
import json
import subprocess
import sys
import time


def check_pod(label_selector: str, namespace: str, timeout: int, interval: int) -> bool:
    print(f"Waiting for pod with label '{label_selector}' in namespace '{namespace}' to be Ready...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            output = subprocess.check_output(
                [
                    "kubectl",
                    "get",
                    "pods",
                    "-l",
                    label_selector,
                    "-n",
                    namespace,
                    "-o",
                    "json",
                ],
                text=True,
            )
            pods = json.loads(output)
            if not pods["items"]:
                print(f"No pods found with label '{label_selector}'", file=sys.stderr)
                time.sleep(interval)
                continue

            # Check the first pod found by the selector
            pod = pods["items"][0]
            name = pod["metadata"]["name"]

            # Correctly check readiness for all containers
            if "containerStatuses" not in pod["status"]:
                time.sleep(interval)
                continue

            all_ready = all(cs["ready"] for cs in pod["status"]["containerStatuses"])

            if all_ready and pod["status"]["phase"] == "Running":
                print(f"Pod '{name}' is Ready.")
                return True
        except subprocess.CalledProcessError as e:
            print(f"Warning: kubectl command failed: {e}", file=sys.stderr)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse kubectl JSON output: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Unexpected error in pod check: {e}", file=sys.stderr)
        time.sleep(interval)
    print(f"Timeout: Pod with label '{label_selector}' did not become ready within {timeout} seconds.", file=sys.stderr)
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wait for a Kubernetes pod to become ready.")
    parser.add_argument(
        "-l",
        "--label-selector",
        required=True,
        help="The label selector to identify the pod (e.g., 'app=myapp').",
    )
    parser.add_argument(
        "-n",
        "--namespace",
        default="default",
        help="The namespace where the pod is running.",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=900,
        help="Timeout in seconds.",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=10,
        help="Polling interval in seconds.",
    )
    args = parser.parse_args()

    if check_pod(args.label_selector, args.namespace, args.timeout, args.interval):
        sys.exit(0)
    else:
        sys.exit(1)
