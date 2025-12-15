from __future__ import annotations

from pathlib import Path
from typing import Any

import fire
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def _config_dir() -> Path:
    return (Path(__file__).resolve().parents[1] / "configs").resolve()


def _compose_cfg(overrides: list[str] | None = None) -> DictConfig:
    overrides = overrides or []
    with initialize_config_dir(config_dir=str(_config_dir()), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def train(*overrides: str) -> None:
    """
    Train Lightning model.
    Usage:
      disaster-tweets train
      disaster-tweets train train.max_epochs=3 train.batch_size=128
    """
    cfg = _compose_cfg(list(overrides))
    repo_root = Path(__file__).resolve().parents[1]
    from disaster_tweets_nlp.training import train as run_train

    run_train(cfg=cfg, repo_root=repo_root)


def baseline_train(*overrides: str) -> None:
    """
    Train baseline model (sklearn).
    Usage:
      disaster-tweets baseline-train
      disaster-tweets baseline-train baseline.vectorizer.type=lsa baseline.lsa.n_components=200
    """
    cfg = _compose_cfg(list(overrides))
    repo_root = Path(__file__).resolve().parents[1]
    from disaster_tweets_nlp.baseline_training import baseline_train as run_baseline_train

    run_baseline_train(cfg=cfg, repo_root=repo_root)


def main() -> Any:
    commands = {
        "train": train,
        "baseline-train": baseline_train,
    }
    return fire.Fire(commands)


if __name__ == "__main__":
    main()
