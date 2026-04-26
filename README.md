# HW5 MLOps

Домашнее задание №5 курса «Развертывание ML моделей» (МФТИ, 501).

## Цель проекта

Собрать минимальный, но рабочий MLOps-контур для классификации Wine dataset: версионирование данных через DVC, трекинг экспериментов через MLflow, декларативный пайплайн и воспроизводимость от коммита до метрик. Дополнительно реализован Feature Store на Feast с Postgres в качестве хранилища.

## Как запустить

```bash
uv sync
uv run dvc repro
uv run dvc metrics show
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Краткое описание пайплайна

Декларативный DVC-пайплайн из двух стадий:

- **prepare** (`src/prepare.py`): загружает Wine dataset, делает стратифицированный train/test split. Параметры: `prepare.split_ratio`, `prepare.random_state`. Выход: `data/processed/train.csv`, `data/processed/test.csv`.
- **train** (`src/train.py`):обучает `RandomForestClassifier`, считает метрики (accuracy, precision_macro, recall_macro, f1_macro), логирует параметры, метрики и модель в MLflow. Параметры: `train.n_estimators`, `train.max_depth`, `train.random_state`. Выход: `models/model.pkl`, `metrics.json`.

Все гиперпараметры в `params.yaml`. При их изменении `dvc repro` пересчитывает только зависимые стадии.

## Где смотреть UI MLflow

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Открыть [http://127.0.0.1:5000](http://127.0.0.1:5000). Эксперимент: `hw5-wine-classification`.

## Feature Store

Поднимается отдельно через Docker Compose, не зависит от основного пайплайна. Подробности - в [feature_store/README.md](feature_store/README.md).

## Отчёты

- [Marimo vs Jupyter](reports/marimo_vs_jupyter.md)
- [Готовность ML-системы к продакшену](reports/production.md)
- [Архитектура системы блюра лиц](reports/face_blur_architecture.md)
