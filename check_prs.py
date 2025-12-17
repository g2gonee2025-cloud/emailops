import subprocess


def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def main():
    # Get all remote branches matching pr
    res = run("git branch -r | grep 'origin/pr/'")
    if res.returncode != 0:
        print("No PRs found")
        return

    prs = [line.strip() for line in res.stdout.splitlines()]
    unmerged = []

    print(f"Checking {len(prs)} PRs...")
    for pr in prs:
        # Check if merged
        # git merge-base --is-ancestor <commit> <main>
        # If return code 0, it is an ancestor (merged)
        check = run(f"git merge-base --is-ancestor {pr} HEAD")
        if check.returncode != 0:
            unmerged.append(pr)
            # Get commit msg
            msg = run(f"git log -1 --pretty=%s {pr}").stdout.strip()
            print(f"[UNMERGED] {pr}: {msg}")
        else:
            # print(f"[MERGED] {pr}")
            pass

    print(f"\nFound {len(unmerged)} unmerged PRs.")


if __name__ == "__main__":
    main()
