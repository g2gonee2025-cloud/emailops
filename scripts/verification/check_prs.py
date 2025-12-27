import subprocess
import sys


def run_git_command(cmd: list[str]) -> subprocess.CompletedProcess:
    """Execute a Git command and return the completed process result."""
    try:
        return subprocess.run(
            ["git"] + cmd, capture_output=True, text=True, check=False
        )
    except OSError as e:
        print(f"Error running git: {e}", file=sys.stderr)
        sys.exit(1)


def get_unmerged_prs(main_branch: str) -> list[tuple[str, str]]:
    """Get all unmerged PR branches."""
    res = run_git_command(["branch", "-r"])
    if res.returncode != 0:
        print("Failed to get remote branches.", file=sys.stderr)
        return []

    pr_branches = [
        line.strip()
        for line in res.stdout.splitlines()
        if "origin/pr/" in line and "->" not in line
    ]

    if not pr_branches:
        return []

    unmerged = []
    for pr in pr_branches:
        check = run_git_command(["merge-base", "--is-ancestor", pr, main_branch])
        if check.returncode != 0:
            msg_res = run_git_command(["log", "-1", "--pretty=%s", pr])
            if msg_res.returncode == 0:
                msg = msg_res.stdout.strip()
            else:
                msg = "failed to retrieve commit message"
            unmerged.append((pr, msg))
    return unmerged


def main():
    """Main function."""
    main_branch = "origin/main"
    unmerged_prs = get_unmerged_prs(main_branch)

    if not unmerged_prs:
        print("No unmerged PRs found.")
        return

    print(f"Found {len(unmerged_prs)} unmerged PRs:")
    for pr, msg in unmerged_prs:
        print(f"  - [UNMERGED] {pr}: {msg}")


if __name__ == "__main__":
    main()
