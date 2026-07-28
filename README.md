# Trading System

Weekly/biweekly rebalanced equities system using 3x leveraged ETFs for
leverage (no margin loan), built for a 3-person team on $25k–$110k capital.

This is engineering scaffolding for the system described in the build plan —
**not financial advice, and not a signal source.** Nothing here should be
used with real money until it has been through Phases 6–7 (paper trading,
then small live capital) end to end.

## What's implemented vs. stubbed

Fully working now:
- Docker-composed TimescaleDB + MLflow (Phase 0)
- DB schema + migrations (Phase 1)
- Price ingestion (Alpaca or yfinance fallback) with upsert + validators (Phase 1)
- Fundamentals + news ingestion via Polygon (needs `POLYGON_API_KEY`) (Phase 1)
- Headline sentiment scoring via Claude (needs `ANTHROPIC_API_KEY`) (Phase 2)
- Macro calendar: FOMC dates scraped from federalreserve.gov, CPI/Jobs dates
  from FRED's API (needs `FRED_API_KEY` — bls.gov itself blocks scraping),
  monthly systemd timer in `infra/systemd/macro-calendar-refresh.*` (Phase 1)
- Quant feature functions: momentum, volatility, mean-reversion (Phase 2)
- Leveraged-ETF **daily-reset decay simulator** — the thing the plan calls
  out as the #1 modeling mistake (Phase 4)
- Event-driven backtest engine + cost model (Phase 4)
- Walk-forward training harness (Phase 3)
- Position sizing + circuit breakers (Phase 5)
- Alpaca paper/live broker wrapper (Phase 5)
- **IBKR broker wrapper** via TWS/IB Gateway (default; same ports as Blue Chip bot)
- Streamlit monitoring dashboard: decisions, price history, forecast-accuracy
  trend, equity/drawdown chart, circuit-breaker status panel (Phase 8) — the
  latter two need `monitoring.equity.record_equity_snapshot()` and
  `monitoring.breaker_state.check_and_record_breakers()` wired into the
  execution loop once paper trading is running, to have data to show
- Slack alerting (needs `SLACK_WEBHOOK_URL`)
- Unit tests for the highest-risk modules (decay sim, validators, sizing,
  circuit breakers) plus the vendor-integration and reconciliation modules
  added above

Stubbed with a clear interface, needs a vendor/API key + your judgment calls:
- Regime (trend-vs-chop) classifier (rule-based ADX stub — swap in a trained
  model once you have walk-forward infra validated and labeled regime history;
  no vendor needed, this one's a training-data problem, not an API problem)

## Quickstart

```bash
cp .env.example .env        # BROKER=ibkr by default; start TWS paper on port 7497
docker compose up -d        # starts TimescaleDB + MLflow
pip install -r requirements.txt
python -m data.schema.migrate            # applies data/schema/*.sql
python -m data.ingest.prices --backfill-years 5 --symbols SPY,QQQ,TQQQ,SQQQ
pytest                                    # run unit tests
```

Then wire up cron/systemd timers (examples in `infra/systemd/`) for the
daily ingestion jobs, and start iterating on `features/`, `models/train.py`,
and `backtest/engine.py` per the phase plan.

## Repo map

See the original build plan for the phase-by-phase timeline. Directory
layout matches it 1:1:

```
data/        ingestion, schema, validators           (Phase 1)
features/    quant / qualitative / event-risk factors (Phase 2)
models/      forecast + regime models, walk-forward   (Phase 3)
backtest/    event-driven engine, decay sim, costs    (Phase 4)
risk/        sizing, circuit breakers                 (Phase 5)
execution/   broker integration (IBKR default, Alpaca optional), reconciliation (Phase 5)
monitoring/  dashboard, alerts                         (Phase 8)
infra/       docker-compose, terraform (optional), systemd timers
config/      settings / env loading
tests/       unit tests
```

## Design decisions this scaffold assumes (from the plan)

- One deployable unit, no microservices, until there's live P&L.
- Cron/systemd + one Prefect flow, not Airflow/K8s.
- CPU-only compute — GBTs on weekly tabular data don't need a GPU.
- IBKR as primary broker (TWS/IB Gateway paper first); Alpaca optional via `BROKER=alpaca`.
- TimescaleDB (Postgres) for everything time-series.
- MLflow (self-hosted) for model/run tracking.
