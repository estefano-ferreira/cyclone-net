# src/utils/git.py
"""
Utility functions for retrieving Git repository information.
Used for provenance tracking in experiment snapshots.
"""

import subprocess
from pathlib import Path


def get_git_revision_short_hash() -> str:
    """
    Returns the short SHA hash of the current Git commit.
    If not in a Git repository or Git is not available, returns 'unknown'.
    """
    try:
        # Determine the repository root (assume this file is inside src/utils/)
        repo_root = Path(__file__).parent.parent.parent
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"