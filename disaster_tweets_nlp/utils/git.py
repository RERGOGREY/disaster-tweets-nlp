from __future__ import annotations

import subprocess


def get_git_commit_id() -> str:
    """Return current git commit hash if available, else 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"
