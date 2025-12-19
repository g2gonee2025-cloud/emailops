import subprocess
import sys
import time


def check_pod():
    print("Waiting for embeddings-api to be Ready...")
    start = time.time()
    while time.time() - start < 900:  # 15 min timeout (large image + model load)
        try:
            output = subprocess.check_output(
                ["kubectl", "get", "pods", "-n", "emailops"], text=True
            )
            for line in output.splitlines():
                if "embeddings-api" in line and "Running" in line:
                    if "1/1" in line:
                        print(f"Pod is Ready: {line}")
                        return True
                    else:
                        # 0/1 Running means started but probe not passed (model loading)
                        pass
        except subprocess.CalledProcessError as e:
            # Log kubectl command failures for debugging
            print(f"Warning: kubectl command failed: {e}", file=sys.stderr)
        except Exception as e:
            # Log unexpected errors but continue retrying
            print(f"Warning: Unexpected error in pod check: {e}", file=sys.stderr)
        time.sleep(10)
    return False


if __name__ == "__main__":
    if check_pod():
        sys.exit(0)
    else:
        sys.exit(1)
