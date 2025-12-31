import json
import re
from collections import defaultdict
from pathlib import Path

LOG_FILE = Path("bulk_review_all.log")
OUTPUT_MD = Path("preliminary_review_report.md")


def parse_log():
    if not LOG_FILE.exists():
        print(f"Log file {LOG_FILE} not found.")
        return

    issues_by_file = defaultdict(int)
    failed_files = []

    # Regex to capture content
    # 2025-12-30 15:13:13,769 [WARNING] ⚠️ routes_auth.py (openai-gpt-5): 10 issues
    # 2025-12-30 15:52:14,552 [ERROR] ❌ async_cache.py (openai-gpt-5): Max retries exceeded

    issue_pattern = re.compile(r"⚠️ (.+?) \(.*\): (\d+) issues")
    fail_pattern = re.compile(r"❌ (.+?) \(.*\): Max retries exceeded")

    with open(LOG_FILE) as f:
        for line in f:
            issue_match = issue_pattern.search(line)
            if issue_match:
                filename = issue_match.group(1)
                count = int(issue_match.group(2))
                issues_by_file[filename] = count
                continue

            fail_match = fail_pattern.search(line)
            if fail_match:
                filename = fail_match.group(1)
                failed_files.append(filename)

    # Generate Markdown
    with open(OUTPUT_MD, "w") as f:
        f.write("# Preliminary Bulk Code Review Report\n\n")
        f.write(
            "> **Note:** The review process is incomplete due to API rate limits. This report summarizes findings from the logs so far.\n\n"
        )

        total_files = len(issues_by_file)
        total_issues = sum(issues_by_file.values())

        f.write(f"- **Files Analyzed Successfully:** {total_files}\n")
        f.write(f"- **Total Issues Found:** {total_issues}\n")
        f.write(f"- **Failed Files:** {len(failed_files)}\n\n")

        f.write("## Top Issues by File\n\n")
        f.write("| File | Issues |\n")
        f.write("|------|--------|\n")

        # Sort by issue count descending
        sorted_files = sorted(issues_by_file.items(), key=lambda x: x[1], reverse=True)

        for filename, count in sorted_files:
            f.write(f"| `{filename}` | {count} |\n")

        f.write("\n## Failed Files (Max Retries)\n\n")
        for filename in failed_files:
            f.write(f"- `{filename}`\n")

    print(f"Report generated at {OUTPUT_MD}")

    # Generate JSON for successful files
    json_output_path = Path("processed_files_review.json")
    successful_data = {
        "summary": {"total_files": len(issues_by_file), "total_issues": total_issues},
        "files": [{"filename": f, "issue_count": c} for f, c in sorted_files],
    }

    with open(json_output_path, "w") as f:
        json.dump(successful_data, f, indent=2)
    print(f"JSON export generated at {json_output_path}")


if __name__ == "__main__":
    parse_log()
