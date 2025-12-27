import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PRIORITY = 10
HIGH_PRIORITY_THRESHOLD = 3
REPORT_FILENAME = "bulk_review_report_v2.json"

CATEGORY_PRIORITY = {
    "LOGIC_ERRORS": 1,
    "SECURITY": 2,
    "PERFORMANCE": 3,
    "EXCEPTION_HANDLING": 4,
    "NULL_SAFETY": 5,
    "TYPE_ERRORS": 6,
    "STYLE": 7,
}


def prioritize_issues() -> None:
    report_path = Path(REPORT_FILENAME)
    if not report_path.exists():
        logger.error("Report not found at %s", report_path)
        return

    try:
        with open(report_path) as f:
            data = json.load(f)
    except Exception as e:
        logger.error("Failed to load report: %s", e)
        return

    issues_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)

    # Files already fixed in Phase 1-3
    FIXED_FILES = {
        "backend/src/cortex/context.py",
        "backend/src/cortex/cmd_doctor.py",
        "backend/src/cortex/observability.py",
        "backend/src/cortex/utils/atomic_io.py",
        "backend/src/cortex/utils/__init__.py",
        "backend/src/cortex/email_processing.py",
        "backend/src/cortex/cli.py",
        "backend/src/cortex/common/exceptions.py",
        "backend/src/cortex/rag_api/models.py",
        "backend/src/cortex/rag_api/routes_search.py",
    }

    for issue in data.get("issues", []):
        if issue.get("file") in FIXED_FILES:
            continue
        if "file" in issue:
            issues_by_file[issue["file"]].append(issue)

    # Sort files by highest priority issue found
    file_scores = []
    for file_path, issues in issues_by_file.items():
        # Get min priority score (1 is highest)
        best_score = min(
            CATEGORY_PRIORITY.get(i.get("category", ""), DEFAULT_PRIORITY)
            for i in issues
        )
        file_scores.append((best_score, file_path, issues))

    file_scores.sort(key=lambda x: x[0])

    logger.info("Found issues in %d files (excluding fixed).", len(file_scores))
    logger.info("Top 10 Files to Fix:")
    for score, fname, issues in file_scores[:10]:
        logger.info("\nFile: %s (Priority: %s)", fname, score)
        # specific high prio issues
        high_prio = [
            i
            for i in issues
            if CATEGORY_PRIORITY.get(i.get("category", ""), DEFAULT_PRIORITY)
            <= HIGH_PRIORITY_THRESHOLD
        ]
        for idx, i in enumerate(high_prio[:5]):
            desc = i.get("description", "")
            logger.info("  %d. [%s] %s...", idx + 1, i.get("category"), desc[:100])


if __name__ == "__main__":
    prioritize_issues()
