import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import load_data_config, load_model_config, load_paths  # noqa: E402

MODEL_CLASSES = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def build_model(model_name: str):
    """Instantiate a model from its config/<model_name>.yaml hyperparameters."""
    model_cls = MODEL_CLASSES.get(model_name)
    if model_cls is None:
        raise ValueError(f"Model '{model_name}' not found. Known models: {list(MODEL_CLASSES)}")
    params = load_model_config(model_name)["params"]
    return model_cls(**params), params


def load_split(processed_dir: Path, target_column: str, split: str):
    X = pd.read_csv(processed_dir / f"X_{split}.csv")
    y = pd.read_csv(processed_dir / f"y_{split}.csv")[target_column]
    return X, y


def evaluate(model, X, y, pos_label: int) -> dict:
    predictions = model.predict(X)
    neg_label = 1 - pos_label
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[neg_label, pos_label]).ravel()
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, pos_label=pos_label)),
        "recall": float(recall_score(y, predictions, pos_label=pos_label)),
        "f1": float(f1_score(y, predictions, pos_label=pos_label)),
        "confusion_matrix": {
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
        },
    }


def train_and_evaluate(model_name: str, processed_dir: Path, target_column: str, pos_label: int) -> dict:
    """Train model_name on the training split and evaluate it on val and test.

    pos_label identifies which class precision/recall/F1 are scored
    against (see config/data_config.yaml: this dataset's phishing class
    is 0, not sklearn's default positive label of 1).
    """
    model, params = build_model(model_name)

    X_train, y_train = load_split(processed_dir, target_column, "train")
    model.fit(X_train, y_train)

    results = {
        "model": model_name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "positive_class": pos_label,
        "params": params,
    }

    for split in ("val", "test"):
        X_split, y_split = load_split(processed_dir, target_column, split)
        metrics = evaluate(model, X_split, y_split, pos_label)
        summary = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "confusion_matrix")
        print(f"[{model_name}] {split}: {summary}")
        results[split] = metrics

    return results


def save_metrics(results: dict, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{results['model']}_metrics.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(results, f, sort_keys=False)
    print(f"Metrics saved to {out_path}")
    return out_path


if __name__ == "__main__":
    paths = load_paths()
    data_config = load_data_config()

    for model_name in MODEL_CLASSES:
        results = train_and_evaluate(
            model_name,
            paths["processed_dir"],
            data_config["target_column"],
            data_config["positive_class"],
        )
        save_metrics(results, paths["results_dir"])
