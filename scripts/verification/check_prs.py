import subprocess


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Execute a command and return the completed process result.

    Args:
        cmd: The command to execute as a list of arguments.

    Returns:
        The CompletedProcess object with stdout/stderr as strings.
    """
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    # Get all remote branches matching pr without using a shell
    try:
        res = subprocess.run(["git", "branch", "-r"], capture_output=True, text=True)
    except OSError as e:
        print(f"Failed to run git: {e}")
        return
    if res.returncode != 0:
        print("No PRs found")
        return
    # Filter branches for 'origin/pr/' like the previous grep
    branches = [
        line.strip() for line in res.stdout.splitlines() if "origin/pr/" in line
    ]
    if not branches:
        print("No PRs found")
        return
    res.stdout = "\n".join(branches)

    prs = [line.strip() for line in res.stdout.splitlines()]
    unmerged = []

    print(f"Checking {len(prs)} PRs...")
    for pr in prs:
        # Check if merged
        # git merge-base --is-ancestor <commit> <main>
        # If return code 0, it is an ancestor (merged)
        check = run(["git", "merge-base", "--is-ancestor", pr, "HEAD"])
        if check.returncode != 0:
            unmerged.append(pr)
            # Get commit msg
            try:
                res = run(["git", "log", "-1", "--pretty=%s", pr])
                if res.returncode == 0:
                    msg = res.stdout.strip()
                else:
                    msg = "failed to retrieve commit message"
            except Exception as e:
                msg = f"failed to retrieve commit message: {e}"
            print(f"[UNMERGED] {pr}: {msg}")
        else:
            # print(f"[MERGED] {pr}")
            pass

    print(f"\nFound {len(unmerged)} unmerged PRs.")


if __name__ == "__main__":
    main()
