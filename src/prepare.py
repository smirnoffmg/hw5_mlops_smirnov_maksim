"""Prepare Wine dataset: load, split into train/test, save as CSV."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "wine.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params() -> dict:
    with PARAMS_PATH.open() as f:
        return yaml.safe_load(f)["prepare"]


def load_or_create_raw() -> pd.DataFrame:
    if RAW_PATH.exists():
        log.info("Loading raw dataset from %s", RAW_PATH)
        return pd.read_csv(RAW_PATH)

    log.info("Raw dataset not found, generating from sklearn.datasets.load_wine")
    bunch = load_wine(as_frame=True)
    df = bunch.frame
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    log.info("Saved raw dataset to %s (rows=%d)", RAW_PATH, len(df))
    return df


def main() -> None:
    params = load_params()
    df = load_or_create_raw()

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["split_ratio"],
        random_state=params["random_state"],
        stratify=y,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    log.info("Saved train (%d rows) -> %s", len(train_df), train_path)
    log.info("Saved test  (%d rows) -> %s", len(test_df), test_path)


if __name__ == "__main__":
    main()
