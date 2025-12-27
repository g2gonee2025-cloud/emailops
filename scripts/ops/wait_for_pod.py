import subprocess
import sys
import time

TIMEOUT_SECONDS = 15 * 60  # Timeout in seconds (15 minutes)


def check_pod() -> bool:
    print("Waiting for embeddings-api to be Ready...")
    start = time.time()
    while (
        time.time() - start < TIMEOUT_SECONDS
    ):  # 15 min timeout (large image + model load)
        try:
            output = subprocess.check_output(
                ["kubectl", "get", "pods", "-n", "emailops"], text=True
            )
            for line in output.splitlines():
                parts = line.split()
                if not parts or parts[0] == "NAME":
                    continue
                name = parts[0]
                ready = parts[1] if len(parts) > 1 else ""
                status = parts[2] if len(parts) > 2 else ""
                if name.startswith("embeddings-api") and status == "Running":
                    if ready == "1/1":
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
