import subprocess
import sys


def run_command(cmd: list[str]) -> subprocess.CompletedProcess:
    """Execute a command and return the completed process result."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as e:
        print(f"Error running command '{cmd[0]}': {e}", file=sys.stderr)
        sys.exit(1)


def get_unmerged_prs(main_branch: str) -> list[tuple[str, str]]:
    """Get all unmerged PR branches."""
    # Check if gh is installed
    gh_check = run_command(["gh", "--version"])
    if gh_check.returncode != 0:
        print(
            "GitHub CLI 'gh' not found. Please install it to use this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Get open PR branches from GitHub CLI
    res = run_command(
        ["gh", "pr", "list", "--json", "headRefName", "--jq", ".[] | .headRefName"]
    )
    if res.returncode != 0:
        print(f"Failed to get PRs from GitHub: {res.stderr}", file=sys.stderr)
        sys.exit(1)

    pr_branches = [f"origin/{line.strip()}" for line in res.stdout.splitlines()]

    # Get all branches that are not merged into the main branch
    unmerged_res = run_command(["git", "branch", "-r", "--no-merged", main_branch])
    if unmerged_res.returncode != 0:
        print(
            f"Failed to get unmerged branches: {unmerged_res.stderr}", file=sys.stderr
        )
        sys.exit(1)

    unmerged_branches = {line.strip() for line in unmerged_res.stdout.splitlines()}

    # Find the intersection of PR branches and unmerged branches
    unmerged_prs = [pr for pr in pr_branches if pr in unmerged_branches]

    if not unmerged_prs:
        return []

    # Get the commit message for each unmerged PR
    final_unmerged_prs = []
    for pr in unmerged_prs:
        log_res = run_command(["git", "log", "-1", "--pretty=%s", pr])
        if log_res.returncode == 0:
            msg = log_res.stdout.strip()
        else:
            msg = "failed to retrieve commit message"
        final_unmerged_prs.append((pr, msg))

    return final_unmerged_prs


def main():
    """Main function."""
    if len(sys.argv) > 1:
        main_branch = sys.argv[1]
    else:
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
