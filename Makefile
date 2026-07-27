.PHONY: up down migrate ingest test lint dashboard

up:
	docker compose up -d timescaledb mlflow

down:
	docker compose down

migrate:
	python -m data.schema.migrate

ingest:
	python -m data.ingest.prices --symbols SPY,QQQ,TQQQ,SQQQ --backfill-years 5

test:
	pytest -v

lint:
	ruff check .

dashboard:
	streamlit run monitoring/dashboard/app.py
