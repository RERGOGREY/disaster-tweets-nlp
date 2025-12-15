from __future__ import annotations

from pathlib import Path

import mlflow
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from disaster_tweets_nlp.data.download import download_data
from disaster_tweets_nlp.data.dvc_manager import dvc_pull_if_possible
from disaster_tweets_nlp.data.io import read_csv
from disaster_tweets_nlp.data.preprocessing import build_model_text
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


def _build_texts(cfg: DictConfig, dataframe) -> tuple[list[str], list[int]]:
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
        for _, row in dataframe.iterrows()
    ]
    labels = [int(v) for v in dataframe["target"].tolist()]
    return texts, labels


def _make_vectorizer(cfg: DictConfig):
    vec_type = str(cfg.baseline.vectorizer.type).lower()
    ngram_range = tuple(int(x) for x in cfg.baseline.vectorizer.ngram_range)
    max_features = int(cfg.baseline.vectorizer.max_features)
    min_df = int(cfg.baseline.vectorizer.min_df)
    max_df = float(cfg.baseline.vectorizer.max_df)

    if vec_type == "bow":
        return CountVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
        )
    if vec_type in ("tfidf", "lsa"):
        return TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
        )

    raise ValueError(f"Unknown baseline.vectorizer.type: {vec_type}")


def _make_model(cfg: DictConfig):
    model_type = str(cfg.baseline.model.type).lower()
    if model_type != "logreg":
        raise ValueError(f"Unknown baseline.model.type: {model_type}")

    return LogisticRegression(
        C=float(cfg.baseline.model.C),
        max_iter=int(cfg.baseline.model.max_iter),
        class_weight=str(cfg.baseline.model.class_weight),
        solver="lbfgs",
        n_jobs=1,
    )


def baseline_train(cfg: DictConfig, repo_root: Path) -> None:
    set_global_seed(int(cfg.train.seed))
    console.print("[cyan]Baseline config:[/cyan]")
    console.print(OmegaConf.to_yaml(cfg.baseline))

    _ensure_data(cfg=cfg, repo_root=repo_root)

    train_df = read_csv(cfg.data.files.train_csv)
    texts, labels = _build_texts(cfg=cfg, dataframe=train_df)

    x_train, x_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=float(cfg.data.split.val_size),
        random_state=int(cfg.data.split.seed),
        stratify=labels,
    )

    vectorizer = _make_vectorizer(cfg)
    classifier = _make_model(cfg)

    steps = [("vectorizer", vectorizer)]

    vec_type = str(cfg.baseline.vectorizer.type).lower()
    lsa_enabled = bool(cfg.baseline.lsa.enabled) or (vec_type == "lsa")
    if lsa_enabled:
        steps.append(
            ("svd", TruncatedSVD(n_components=int(cfg.baseline.lsa.n_components), random_state=42))
        )

    steps.append(("clf", classifier))
    pipeline = Pipeline(steps=steps)

    mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))
    mlflow.set_experiment(str(cfg.mlflow.experiment_name))

    run_name = cfg.mlflow.run_name or "baseline"
    with mlflow.start_run(run_name=str(run_name)):
        mlflow.log_param("git_commit_id", get_git_commit_id())
        mlflow.log_params(OmegaConf.to_container(cfg.baseline, resolve=True))
        mlflow.log_params(
            {
                "data.use_keyword": bool(cfg.data.text.use_keyword),
                "data.use_location": bool(cfg.data.text.use_location),
                "split.val_size": float(cfg.data.split.val_size),
                "split.seed": int(cfg.data.split.seed),
            }
        )

        pipeline.fit(x_train, y_train)

        val_probs = pipeline.predict_proba(x_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)

        val_acc = float(accuracy_score(y_val, val_preds))
        val_f1 = float(f1_score(y_val, val_preds))
        try:
            val_auc = float(roc_auc_score(y_val, val_probs))
        except Exception:
            val_auc = float("nan")

        mlflow.log_metric("val/accuracy", val_acc)
        mlflow.log_metric("val/f1", val_f1)
        mlflow.log_metric("val/roc_auc", val_auc)

        # Save artifact (not in git)
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        import joblib  # noqa: PLC0415

        model_path = artifacts_dir / "baseline_model.joblib"
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(str(model_path))

        console.print("[green]Baseline training finished.[/green]")
        console.print(
            f"[green]val/accuracy={val_acc:.4f} val/f1={val_f1:.4f} val/roc_auc={val_auc:.4f}[/green]"
        )
        console.print(f"[green]Saved baseline artifact:[/green] {model_path}")
