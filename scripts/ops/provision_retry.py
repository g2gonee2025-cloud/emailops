"""
Provision GPU machines with exponential backoff and jitter.

Retries provisioning across a list of candidate GPU instance types on a target cluster.
The script reads the cluster UUID from the DO_CLUSTER_ID environment variable.

CLI arguments allow configuration of node count and initial backoff period.
"""

import argparse
import os
import random
import subprocess
import sys
import time

# Candidates with >24GB VRAM
SLUGS = ["gpu-l40sx1-48gb", "gpu-6000adax1-48gb", "gpu-h100x1-80gb"]

CLUSTER_ID = os.environ.get("DO_CLUSTER_ID")
if not CLUSTER_ID:
    print("Error: DO_CLUSTER_ID environment variable not set.", file=sys.stderr)
    sys.exit(1)


def try_provision(slug: str, node_count: int) -> bool:
    # Validate slug to prevent command injection via unexpected characters
    if (
        not isinstance(slug, str)
        or not slug
        or not all(c.isalnum() or c == "-" for c in slug)
    ):
        raise ValueError("Invalid slug: only letters, numbers, and '-' are allowed")
    parts = slug.split("-")
    if len(parts) < 2 or not parts[1] or not parts[1].isalnum():
        raise ValueError("Invalid slug format")
    core = parts[1]
    pool_name = f"pool-{core}"  # e.g. pool-l40sx1
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
        str(node_count),
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            encoding="utf-8",
            errors="surrogateescape",
            timeout=300,
        )
        print(f"SUCCESS: Provisioned {slug}")
        return True
    except FileNotFoundError:
        print("Error: 'doctl' command not found. Is DigitalOcean CLI installed and in your PATH?")
        sys.exit(1)
    except OSError as e:
        print(f"Error executing 'doctl': {e}", file=sys.stderr)
        return False
    except subprocess.TimeoutExpired:
        print(f"FAILED: {slug} (timeout)", file=sys.stderr)
        return False
    except subprocess.CalledProcessError:
        # The stderr is already printed to the console.
        print(f"FAILED: {slug}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Provision GPU nodes with retry logic.")
    parser.add_argument(
        "--count", type=int, default=1, help="Number of nodes to provision."
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=2,
        help="Initial wait time in seconds for backoff.",
    )
    args = parser.parse_args()

    wait_time = args.wait
    for i, slug in enumerate(SLUGS):
        try:
            if try_provision(slug, args.count):
                print("Provisioning successful! Exiting loop.")
                sys.exit(0)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if i < len(SLUGS) - 1:
            # Exponential backoff with jitter
            sleep_time = wait_time * (2**i) + random.uniform(0, 1)
            print(f"Waiting {sleep_time:.2f}s before next attempt...")
            time.sleep(sleep_time)

    print("All attempts failed.")
    sys.exit(1)


if __name__ == "__main__":
    main()
