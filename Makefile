.PHONY: up down migrate universe ingest screen test lint dashboard

up:
	docker compose up -d timescaledb mlflow

down:
	docker compose down

migrate:
	python -m data.schema.migrate

universe:
	python -m data.ingest.universe --scrape

ingest:
	python -m data.ingest.prices --universe --backfill-years 5

screen:
	python -m models.screener --feature-set-id v3 --universe --top-k 10

test:
	pytest -v

lint:
	ruff check .

dashboard:
	python -m monitoring.dashboard.server
