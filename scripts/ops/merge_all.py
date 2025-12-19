import re
import subprocess


def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def main():
    # Fetch again to be sure
    run("git fetch origin")

    # Get unmerged PRs
    res = run("git branch -r | grep 'origin/pr/'")
    if res.returncode != 0:
        print("No PRs found")
        return

    all_prs = [line.strip() for line in res.stdout.splitlines()]

    # Sort by number
    # origin/pr/10 -> 10
    def get_num(s):
        m = re.search(r"/pr/(\d+)", s)
        return int(m.group(1)) if m else 0

    all_prs.sort(key=get_num)

    merged = []
    failed = []

    for pr in all_prs:
        # Check if already merged
        if run(f"git merge-base --is-ancestor {pr} main").returncode == 0:
            continue

        print(f"Attempting merge: {pr}")
        # Merge
        # --no-edit to use default msg
        # --no-ff to create merge commit? Or default.
        # Default is fine.
        proc = run(f"git merge --no-edit {pr}")
        if proc.returncode == 0:
            print(f"SUCCESS: Merged {pr}")
            merged.append(pr)
        else:
            print(f"CONFLICT: Could not merge {pr}")
            print(proc.stdout)
            print(proc.stderr)
            run("git merge --abort")
            failed.append(pr)

    print("\nSummary:")
    print(f"Merged: {len(merged)}")
    print(f"Failed (Conflict): {len(failed)}")
    for p in failed:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
