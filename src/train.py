"""Train RandomForest on Wine dataset, log to MLflow, save model and metrics."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
METRICS_PATH = PROJECT_ROOT / "metrics.json"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT = "hw5-wine-classification"


def load_params() -> dict:
    with PARAMS_PATH.open() as f:
        return yaml.safe_load(f)


def load_processed() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_train, y_train = train.drop(columns=["target"]), train["target"]
    X_test, y_test = test.drop(columns=["target"]), test["target"]
    return X_train, y_train, X_test, y_test


def evaluate(y_true: pd.Series, y_pred) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def main() -> None:
    params = load_params()
    train_params = params["train"]

    X_train, y_train, X_test, y_test = load_processed()
    log.info("Loaded train=%d, test=%d", len(X_train), len(X_test))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=train_params["n_estimators"],
            max_depth=train_params["max_depth"],
            random_state=train_params["random_state"],
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        metrics = evaluate(y_test, model.predict(X_test))
        log.info("Metrics: %s", metrics)

        mlflow.log_params(
            {
                "model": "RandomForestClassifier",
                "n_estimators": train_params["n_estimators"],
                "max_depth": train_params["max_depth"],
                "random_state": train_params["random_state"],
                "split_ratio": params["prepare"]["split_ratio"],
            }
        )
        mlflow.log_metrics(metrics)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model_pickle")
        mlflow.sklearn.log_model(model, name="model")

        with METRICS_PATH.open("w") as f:
            json.dump(metrics, f, indent=2)

        log.info("Saved model to %s", MODEL_PATH)
        log.info("Saved metrics to %s", METRICS_PATH)


if __name__ == "__main__":
    main()
