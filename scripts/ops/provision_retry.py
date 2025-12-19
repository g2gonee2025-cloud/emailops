import subprocess
import sys
import time

# Candidates with >24GB VRAM
SLUGS = ["gpu-l40sx1-48gb", "gpu-6000adax1-48gb", "gpu-h100x1-80gb"]

CLUSTER_ID = "23c013d9-4d8d-4d3d-a813-7e5cbc3d0af1"


def try_provision(slug):
    pool_name = f"pool-{slug.split('-')[1]}"  # e.g. pool-l40sx1
    pool_name = pool_name.replace("x1", "")  # pool-l40s

    print(f"--- Attempting to provision {slug} as {pool_name} ---")

    cmd = [
        "doctl",
        "kubernetes",
        "cluster",
        "node-pool",
        "create",
        CLUSTER_ID,
        "--name",
        pool_name,
        "--size",
        slug,
        "--count",
        "1",
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"SUCCESS: Provisioned {slug}")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {slug}")
        print(f"Error: {e.stderr.strip()}")
        return False


def main():
    for slug in SLUGS:
        if try_provision(slug):
            print("Provisioning successful! Exiting loop.")
            sys.exit(0)
        print("Waiting 2s before next attempt...")
        time.sleep(2)

    print("All attempts failed.")
    sys.exit(1)


if __name__ == "__main__":
    main()
