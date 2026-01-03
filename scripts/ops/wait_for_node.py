import subprocess
import sys
import time

# Constants for configuration
TIMEOUT_SECONDS = 600  # 10 minutes
SLEEP_INTERVAL_SECONDS = 10


def check_nodes() -> bool:
    """
    Checks for a node with 'pool-gpu-h200' in its name and waits for it to be Ready.

    Returns:
        bool: True if the node becomes Ready within the timeout, False otherwise.
    """
    print("Waiting for pool-gpu-h200 to join and become Ready...")
    start = time.time()
    node_name_substring = "pool-gpu-h200"

    while time.time() - start < TIMEOUT_SECONDS:
        try:
            output = subprocess.check_output(["kubectl", "get", "nodes"], text=True)
            lines = output.strip().splitlines()

            node_found_but_not_ready = False
            for line in lines[1:]:  # Skip header row
                columns = line.split()
                if not columns:
                    continue

                node_name = columns[0]
                if node_name_substring in node_name:
                    status = columns[1]
                    if status == "Ready":
                        print(f"Node '{node_name}' is Ready.")
                        return True
                    else:
                        print(f"Node '{node_name}' found but not Ready (Status: {status}).")
                        node_found_but_not_ready = True
                        break  # Found the node, no need to check other lines

            if not node_found_but_not_ready:
                print("Node not yet in list...")

        except subprocess.CalledProcessError as e:
            print(f"Warning: 'kubectl get nodes' command failed: {e}", file=sys.stderr)
        except (KeyboardInterrupt, SystemExit):
            print("\nInterrupted. Exiting.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)  # Exit on unexpected errors

        time.sleep(SLEEP_INTERVAL_SECONDS)

    print(f"Timeout of {TIMEOUT_SECONDS}s reached. Node did not become Ready.")
    return False


if __name__ == "__main__":
    if check_nodes():
        sys.exit(0)
    else:
        sys.exit(1)
