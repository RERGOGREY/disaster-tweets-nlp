from __future__ import annotations

from pathlib import Path

from rich.console import Console

console = Console()


def dvc_pull_if_possible(repo_root: Path, targets: list[str] | None = None) -> bool:
    """
    Attempt to pull data via DVC Python API. Returns True if succeeded, else False.
    """
    try:
        from dvc.repo import Repo  # local import to avoid hard fail if dvc misconfigured

        repo = Repo(str(repo_root))
        repo.pull(targets=targets)
        return True
    except Exception as exc:
        console.print(f"[yellow]DVC pull failed:[/yellow] {exc}")
        return False
