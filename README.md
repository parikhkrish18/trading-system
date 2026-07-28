# Trading System

Weekly/biweekly rebalanced **equity long/short screener**: scans the S&P 500,
scores every name with an ensemble forecast model, and only shortlists
trades the model is actually confident about (ensemble members agree on
direction + predicted return clears a threshold) — long or short, whichever
the data supports. Built for a 3-person team on $25k–$110k capital.

This is engineering scaffolding for the system described in the build plan —
**not financial advice, and not a signal source.** Nothing here should be
used with real money until it has been through Phases 6–7 (paper trading,
then small live capital) end to end. Directional accuracy on the current
feature set has hovered around ~50% (a coin flip) in walk-forward testing —
see `models/train.py`'s `directional_accuracy_when_confident` metric before
trusting any shortlist this produces.

**Not the leveraged-ETF strategy anymore.** The original design traded a
fixed list of 4 leveraged ETFs (SPY/QQQ/TQQQ/SQQQ). That's been replaced by
the cross-sectional S&P 500 screener above. The leveraged-ETF daily-reset
**decay simulator** (`backtest/decay_sim.py`) and its cost modeling are
still in the repo, validated and tested — just not wired into the default
pipeline. Revisit them if leveraged products come back into scope.

## What's implemented vs. stubbed

Fully working now:
- DB schema + migrations (Phase 1) — TimescaleDB is optional: every
  `create_hypertable` call is skipped if the extension isn't installed,
  falling back to plain Postgres tables (no functional difference for
  anything in this repo, just no automatic time-partitioning)
- S&P 500 universe, scraped from Wikipedia (`data/ingest/universe.py`) —
  drives every ingestion/training/screening script's `--universe` flag
- Price ingestion (yfinance batch, or Alpaca) with upsert + validators (Phase 1)
- Fundamentals + news ingestion via Polygon (needs `POLYGON_API_KEY`), with
  429-aware backoff and pacing for scanning hundreds of symbols on a
  rate-limited tier (Phase 1)
- Headline sentiment scoring via Claude (needs `ANTHROPIC_API_KEY`) (Phase 2)
- Macro calendar: FOMC dates scraped from federalreserve.gov, CPI/Jobs dates
  from FRED's API (needs `FRED_API_KEY` — bls.gov itself blocks scraping),
  monthly systemd timer in `infra/systemd/macro-calendar-refresh.*` (Phase 1)
- Quant + qualitative (sentiment) + event-risk (macro countdown) + fundamentals
  features, vectorized to scale to the full universe (Phase 2)
- Walk-forward training harness with a **bagged ensemble** forecast model
  (`models/forecast/ensemble.py`) — a single LightGBM model has no native
  confidence signal, so this trains K models with different seeds and only
  treats a prediction as trustworthy if they agree on direction (Phase 3)
- **The screener** (`models/screener.py`): scores the full universe, ranks
  by conviction (agreement × magnitude), sizes the shortlist via
  `risk/sizing.py`, and logs candidates to `decisions` (mode=paper,
  nothing executed) — it does not place orders, see Non-goals below
- Long **and short** support: `MAX_SHORT_POSITION_PCT` (more conservative
  than the long cap — short losses are structurally uncapped), an
  Alpaca `is_shortable()` pre-check, IBKR's lack of an equivalent
  documented in `execution/broker_ibkr.py`
- Leveraged-ETF **daily-reset decay simulator** (Phase 4, kept but inactive
  — see above)
- Event-driven backtest engine + cost model (Phase 4)
- Position sizing + circuit breakers (Phase 5)
- Alpaca paper/live broker wrapper (Phase 5)
- **IBKR broker wrapper** via TWS/IB Gateway (default; same ports as Blue Chip bot)
- Streamlit monitoring dashboard: decisions, price history, forecast-accuracy
  trend, equity/drawdown chart, circuit-breaker status panel (Phase 8) — the
  latter two need `monitoring.equity.record_equity_snapshot()` and
  `monitoring.breaker_state.check_and_record_breakers()` wired into the
  execution loop once paper trading is running, to have data to show
- Slack alerting (needs `SLACK_WEBHOOK_URL`)
- Unit tests across ingestion, features, the ensemble model, the screener,
  sizing, and the highest-risk modules (decay sim, validators, circuit breakers)

Stubbed with a clear interface, needs a vendor/API key + your judgment calls:
- Regime (trend-vs-chop) classifier (rule-based ADX stub — swap in a trained
  model once you have walk-forward infra validated and labeled regime history;
  no vendor needed, this one's a training-data problem, not an API problem)

Explicitly **not built yet**: nothing wires the screener's output to
`execution/broker.py` — it produces and logs candidates, it doesn't place
orders. That's the next step before paper trading can actually run.

## Quickstart

```bash
cp .env.example .env        # BROKER=ibkr by default; start TWS paper on port 7497
docker compose up -d        # starts TimescaleDB + MLflow (or run both natively — see below)
pip install -r requirements.txt
python -m data.schema.migrate            # applies data/schema/*.sql
python -m data.ingest.universe --scrape  # populates the S&P 500 universe table
python -m data.ingest.prices --universe --backfill-years 5
python -m data.ingest.fundamentals --universe   # slow on a rate-limited Polygon key, see --sleep-seconds
python -m data.ingest.news --universe
python -m features.build_features --universe --feature-set-id v3
python -m models.train --universe --feature-set-id v3 --n-folds 6
python -m models.screener --universe --feature-set-id v3 --top-k 10
pytest                                    # run unit tests
```

If Docker isn't available (this repo has been run against native Homebrew
Postgres + a local `mlflow server` process without issue — just point
`MLFLOW_TRACKING_URI` at whatever port you actually started it on; macOS's
AirPlay Receiver squats on the default 5000).

Then wire up cron/systemd timers (examples in `infra/systemd/`) for the
daily ingestion jobs, and start iterating on `features/`, `models/train.py`,
and `models/screener.py` per the phase plan.

## Repo map

See the original build plan for the phase-by-phase timeline. Directory
layout matches it 1:1:

```
data/        ingestion (incl. S&P 500 universe), schema, validators (Phase 1)
features/    quant / qualitative / event-risk factors (Phase 2)
models/      forecast ensemble + screener + regime + walk-forward (Phase 3)
backtest/    event-driven engine, decay sim (inactive), costs   (Phase 4)
risk/        sizing (long + short), circuit breakers            (Phase 5)
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
- TimescaleDB (Postgres) preferred for everything time-series, but not required.
- MLflow (self-hosted) for model/run tracking.
- S&P 500 as the tradeable universe — not a broader index, to keep vendor
  rate limits and data quality manageable on a free-tier API key.
