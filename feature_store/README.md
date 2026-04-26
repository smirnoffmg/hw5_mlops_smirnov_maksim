# Feature Store (Feast + Postgres)

Feature Store на Feast с Postgres в качестве registry, online и offline стора. Используется демо-репозиторий `driver_hourly_stats` из шаблона `feast init -t postgres`.

## Запуск

Поднять Postgres:

```bash
cd feature_store
docker compose up -d --wait
```

Применить feature definitions:

```bash
cd feature_repo
uv run feast apply
```

Materialize онлайн-стор:

```bash
CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")
uv run feast materialize-incremental $CURRENT_TIME
```

Запустить serve:

```bash
uv run feast serve --host 0.0.0.0 --port 6566
```

## Проверка эндпоинта

```bash
curl -X POST http://localhost:6566/get-online-features \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      "driver_hourly_stats:conv_rate",
      "driver_hourly_stats:acc_rate",
      "driver_hourly_stats:avg_daily_trips"
    ],
    "entities": {"driver_id": [1001, 1002, 1003]}
  }'
```

## Остановка

```bash
docker compose down
```

## Конфигурация

`feature_repo/feature_store.yaml` использует шаблон `postgres`:

- `registry.registry_type: sql` - registry в Postgres
- `online_store.type: postgres` - онлайн-стор в Postgres
- `offline_store.type: postgres` - офлайн-стор в Postgres

Подключение: `localhost:5432`, db `feast`, user `feast`, password `feast`, sslmode `disable`.