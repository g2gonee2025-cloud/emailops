"""
Central registry for known job types in Cortex.

Provides a single source of truth for job types that can be processed
by the queueing system. This avoids hardcoding job type strings
across different parts of the application.
"""

# List of all recognized job types
# This list should be updated when new worker types are added.
KNOWN_JOB_TYPES: list[str] = [
    "ingest",
    "reindex",
    # Add new job types here
]


def get_known_job_types() -> list[str]:
    """
    Returns a copy of the list of known job types.

    Returns:
        A list of strings, where each string is a registered job type.
    """
    return KNOWN_JOB_TYPES.copy()
