from __future__ import annotations

from pathlib import Path

from rich.console import Console

console = Console()


def download_data(raw_dir: Path) -> None:
    """
    Fallback загрузка данных из открытого источника.

    ВАЖНО:
    Для стабильной проверки лучше использовать DVC remote (dvc pull).
    Эту функцию можно позже заменить на конкретный публичный источник.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    raise RuntimeError(
        "download_data() is not configured. Please set up DVC remote and run `dvc pull`, "
        "or implement open-source downloading here."
    )
