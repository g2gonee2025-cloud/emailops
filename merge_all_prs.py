import json
import subprocess
import time

REPO = "g2gonee2025-cloud/emailops"


def run_command(cmd, check=True):
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, check=check, capture_output=True, text=True
        )
        if not check and result.returncode != 0:
            print(f"Command failed (Exit {result.returncode}): {result.stderr}")
            return None
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        if check:
            raise
        return None


def main():
    print("Fetching open PRs for Smart Merge...")
    # Get JSON list
    out = run_command(
        f"gh pr list --repo {REPO} --limit 50 --json number,title,state,mergeStateStatus,isDraft",
        check=False,
    )

    if not out:
        print("No open PRs found.")
        return

    prs = json.loads(out)

    if not prs:
        print("No open PRs found.")
        return

    print(f"Found {len(prs)} PRs to process.")

    # Sort by number ascending
    prs.sort(key=lambda x: x["number"])

    for pr in prs:
        num = pr["number"]
        title = pr["title"]
        print(f"\nProcessing PR #{num}: {title}")

        # 1. Check if Draft, mark ready
        if pr["isDraft"]:
            print(f"PR #{num} is Draft. Marking ready...")
            run_command(f"gh pr ready {num} --repo {REPO}", check=False)

        # 2. Merge
        print(f"Attempting to merge PR #{num}...")
        res = run_command(
            f"gh pr merge {num} --merge --delete-branch --repo {REPO}", check=False
        )

        if res is None:
            print(
                f"Merge failed for PR #{num}. Attempting 'update-branch' to resolve conflicts..."
            )
            # Try to update branch
            update_res = run_command(
                f"gh pr update-branch {num} --repo {REPO}", check=False
            )

            if update_res is not None:
                print("Branch updated. Waiting 15s for checks/rebase...")
                time.sleep(15)
                print(f"Retrying merge for PR #{num}...")
                res_retry = run_command(
                    f"gh pr merge {num} --merge --delete-branch --repo {REPO}",
                    check=False,
                )
                if res_retry is not None:
                    print(f"Successfully merged PR #{num} after update.")
                else:
                    print(
                        f"Still failed to merge PR #{num}. Manual intervention required."
                    )
            else:
                print(
                    f"Could not update branch for PR #{num}. Likely complex conflict."
                )
        else:
            print(f"Successfully merged PR #{num}.")
            time.sleep(2)

    print("\nBatch merge complete.")


if __name__ == "__main__":
    main()
