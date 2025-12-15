from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from rich.console import Console
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from disaster_tweets_nlp.data.dataset import TweetDataset
from disaster_tweets_nlp.data.download import download_data
from disaster_tweets_nlp.data.dvc_manager import dvc_pull_if_possible
from disaster_tweets_nlp.data.io import read_csv
from disaster_tweets_nlp.data.preprocessing import build_model_text
from disaster_tweets_nlp.data.tokenizer import SimpleTokenizer
from disaster_tweets_nlp.models.lightning_module import DisasterTweetsLitModule
from disaster_tweets_nlp.utils.git import get_git_commit_id
from disaster_tweets_nlp.utils.seed import set_global_seed

console = Console()


def _ensure_data(cfg: DictConfig, repo_root: Path) -> None:
    train_csv = Path(cfg.data.files.train_csv)
    test_csv = Path(cfg.data.files.test_csv)

    if train_csv.exists() and test_csv.exists():
        return

    console.print("[yellow]Data files not found. Trying DVC pull...[/yellow]")
    pulled = dvc_pull_if_possible(repo_root=repo_root, targets=[str(train_csv), str(test_csv)])

    if train_csv.exists() and test_csv.exists():
        console.print("[green]Data is available after DVC pull.[/green]")
        return

    if pulled and (train_csv.exists() and test_csv.exists()):
        return

    console.print("[yellow]DVC pull did not provide data. Trying download_data()...[/yellow]")
    download_data(raw_dir=Path(cfg.data.paths.raw_dir))


def train(cfg: DictConfig, repo_root: Path) -> None:
    set_global_seed(int(cfg.train.seed))
    console.print("[cyan]Train config:[/cyan]")
    console.print(OmegaConf.to_yaml(cfg.train))

    _ensure_data(cfg=cfg, repo_root=repo_root)

    train_df = read_csv(cfg.data.files.train_csv)

    use_keyword = bool(cfg.data.text.use_keyword)
    use_location = bool(cfg.data.text.use_location)

    texts = [
        build_model_text(
            raw_text=str(row["text"]),
            keyword=row.get("keyword"),
            location=row.get("location"),
            use_keyword=use_keyword,
            use_location=use_location,
        )
        for _, row in train_df.iterrows()
    ]
    labels = [int(v) for v in train_df["target"].tolist()]

    x_train, x_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=float(cfg.data.split.val_size),
        random_state=int(cfg.data.split.seed),
        stratify=labels,
    )

    tokenizer = SimpleTokenizer.build(
        texts=x_train,
        min_freq=int(cfg.data.text.min_freq),
        max_len=int(cfg.data.text.max_len),
    )

    train_dataset = TweetDataset(texts=x_train, labels=y_train, tokenizer=tokenizer)
    val_dataset = TweetDataset(texts=x_val, labels=y_val, tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
    )

    lit_model = DisasterTweetsLitModule(
        vocab_size=len(tokenizer.vocab.itos),
        pad_id=tokenizer.vocab.pad_id,
        embedding_dim=int(cfg.model.architecture.embedding_dim),
        hidden_dim=int(cfg.model.architecture.hidden_dim),
        num_layers=int(cfg.model.architecture.num_layers),
        dropout=float(cfg.model.architecture.dropout),
        bidirectional=bool(cfg.model.architecture.bidirectional),
        lr=float(cfg.model.optimization.lr),
        weight_decay=float(cfg.model.optimization.weight_decay),
    )

    mlflow_logger = MLFlowLogger(
        tracking_uri=str(cfg.mlflow.tracking_uri),
        experiment_name=str(cfg.mlflow.experiment_name),
        run_name=str(cfg.mlflow.run_name) if cfg.mlflow.run_name else None,
    )
    mlflow_logger.log_hyperparams({"git_commit_id": get_git_commit_id()})
    mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    callbacks = [
        ModelCheckpoint(monitor="val/f1", mode="max", save_top_k=1),
        LearningRateMonitor(logging_interval="step"),
    ]
    if bool(cfg.train.early_stopping.enabled):
        callbacks.append(
            EarlyStopping(
                monitor=str(cfg.train.early_stopping.monitor),
                mode=str(cfg.train.early_stopping.mode),
                patience=int(cfg.train.early_stopping.patience),
            )
        )

    trainer = Trainer(
        max_epochs=int(cfg.train.max_epochs),
        logger=mlflow_logger,
        callbacks=callbacks,
        log_every_n_steps=int(cfg.train.trainer.log_every_n_steps),
        accelerator=str(cfg.train.trainer.accelerator),
        devices=cfg.train.trainer.devices,
    )

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = artifacts_dir / "tokenizer.pt"
    tokenizer.save(tokenizer_path)
    console.print(f"[green]Tokenizer saved to:[/green] {tokenizer_path}")

    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    console.print("[green]Training finished.[/green]")
