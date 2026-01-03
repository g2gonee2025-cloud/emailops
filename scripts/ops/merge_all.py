import re
import subprocess


def run(cmd, check=False, capture_output=False):
    """A wrapper around subprocess.run to execute commands safely."""
    # By default, cmd is a list of strings, and shell=False.
    # text=True decodes stdout/stderr as text.
    return subprocess.run(
        cmd, check=check, capture_output=capture_output, text=True
    )


def main():
    # Fetch again to be sure
    print("Fetching from origin...")
    # Use check=True to ensure the script exits if fetch fails.
    run(["git", "fetch", "origin"], check=True)

    # Get all remote branches and filter for PRs in Python
    res = run(["git", "branch", "-r"], check=True, capture_output=True)
    branch_lines = res.stdout.splitlines()

    prs_with_num = []
    for line in branch_lines:
        line = line.strip()
        # Ensure we only match branches like 'origin/pr/123'
        match = re.search(r"origin/pr/(\d+)", line)
        if match:
            pr_number = int(match.group(1))
            # Store the full branch name, e.g., 'origin/pr/123'
            prs_with_num.append((pr_number, line))

    if not prs_with_num:
        print("No PRs found to merge.")
        return

    # Sort by PR number (the first element of the tuple)
    prs_with_num.sort()

    # Get the list of branch names, now correctly sorted
    all_prs = [branch_name for pr_number, branch_name in prs_with_num]

    # Get the current branch's remote tracking branch to merge against.
    # This avoids hardcoding 'main' and ensures we compare against the remote state.
    target_branch_proc = run(
        ["git", "rev-parse", "--abbrev-ref", "@{u}"],
        check=True,
        capture_output=True,
    )
    target_branch = target_branch_proc.stdout.strip()
    print(f"Target branch for merge checks is: {target_branch}")

    merged = []
    failed = []

    for pr in all_prs:
        # Check if already merged against the dynamic target branch.
        is_ancestor_proc = run(
            ["git", "merge-base", "--is-ancestor", pr, target_branch]
        )
        if is_ancestor_proc.returncode == 0:
            # Already merged, so we can skip.
            continue

        print(f"Attempting merge: {pr}")
        # Merge
        # --no-edit to use default msg
        # We need to capture output to show it in case of a conflict.
        proc = run(["git", "merge", "--no-edit", pr], capture_output=True)
        if proc.returncode == 0:
            print(f"SUCCESS: Merged {pr}")
            merged.append(pr)
        else:
            print(f"CONFLICT: Could not merge {pr}")
            print(proc.stdout)
            print(proc.stderr)
            # Use check=True to ensure abort succeeds, preventing repo corruption.
            run(["git", "merge", "--abort"], check=True)
            failed.append(pr)

    print("\nSummary:")
    print(f"Merged: {len(merged)}")
    print(f"Failed (Conflict): {len(failed)}")
    for p in failed:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
