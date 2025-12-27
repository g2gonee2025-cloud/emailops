import subprocess
import sys
import time


def check_nodes() -> None:
    print("Waiting for pool-gpu-h200 to join...")
    start = time.time()
    while time.time() - start < 600:  # 10 min timeout
        try:
            output = subprocess.check_output(["kubectl", "get", "nodes"], text=True)
            if "pool-gpu-h200" in output:
                print("Node found in list!")
                # Check if Ready
                for line in output.splitlines():
                    if "pool-gpu-h200" in line:
                        if "Ready" in line:
                            print(f"Node is Ready: {line}")
                            return True
                        else:
                            print(f"Node found but not Ready: {line}")
            else:
                print("Node not yet in list...")
        except subprocess.CalledProcessError as e:
            # Log kubectl command failures for debugging
            print(f"Warning: kubectl command failed: {e}", file=sys.stderr)
        except (KeyboardInterrupt, SystemExit):
            # Allow termination signals to propagate
            raise
        except Exception as e:
            # Log unexpected errors but continue retrying
            print(f"Warning: Unexpected error in node check: {e}", file=sys.stderr)
        time.sleep(10)
    return False


if __name__ == "__main__":
    if check_nodes():
        sys.exit(0)
    else:
        sys.exit(1)
