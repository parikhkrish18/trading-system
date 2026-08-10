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
- **The screener** (`models/screener.py`): scores the full universe,
  concentrates into the top 2 highest-conviction picks (confidence-weighted
  split, not a diversified top-10 book), and attaches genuine per-decision
  reasoning (LightGBM `pred_contrib` — real Tree SHAP feature contributions,
  not just global feature importance)
- **`execution/trading_loop.py`**: takes the screener's shortlist and
  actually places (paper) orders — pre/post-trade circuit breaker checks,
  full rebalance (closes anything not in the new shortlist), reconciliation,
  equity recording, extended-hours support (limit orders outside RTH, since
  market orders aren't accepted at all then). Hard-enforced paper-only by
  construction: never passes `confirm_live=True`. `scripts/run_weekly_cycle.py`
  chains the full pipeline (universe/price/fundamentals/news refresh →
  screen → trade); `infra/launchd/` schedules it weekly on macOS.
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
- Custom monitoring dashboard (`monitoring/dashboard/server.py` — FastAPI +
  vanilla JS, no Streamlit): every open position with live P&L and the
  model's actual reasoning for entering it, decision history, walk-forward
  analysis from MLflow, live directional hit-rate, equity/drawdown,
  circuit-breaker status, and the test suite runnable on demand. `make dashboard`
- Alerting: Slack webhook (`SLACK_WEBHOOK_URL`) with automatic fallback to
  the Telegram approval bot when Slack is unconfigured/down; entry points
  also log to a rotating `logs/trading-system.log`
- Unit tests across ingestion, features, the ensemble model, the screener,
  sizing, and the highest-risk modules (decay sim, validators, circuit breakers)

Stubbed with a clear interface, needs a vendor/API key + your judgment calls:
- Regime (trend-vs-chop) classifier (rule-based ADX stub — swap in a trained
  model once you have walk-forward infra validated and labeled regime history;
  no vendor needed, this one's a training-data problem, not an API problem)

The screener-to-broker wiring above (`execution/trading_loop.py`, driven by
`scripts/run_weekly_cycle.py`) is paper-only by construction: it never passes
`confirm_live=True` to `get_broker()`, so going live requires deliberately
editing code, not flipping an environment variable.

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
python -m models.screener --universe --feature-set-id v3   # book size comes from SCREENER_TOP_K in .env, default 10
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

## Known evaluation caveat: survivorship bias

The universe is scraped from **today's** Wikipedia S&P 500 list. Any
backtest or walk-forward run over history therefore only ever sees
companies that survived long enough to still be in the index now — the
losers that were removed (bankrupt, acquired at a discount, demoted) are
exactly the names the evaluation can't see, which biases every historical
metric upward. Every universe refresh now appends a dated membership
snapshot to the `universe_snapshot` table (`data/schema/006_universe_snapshot.sql`),
so runs from here on can use point-in-time membership; history from before
the first snapshot stays biased and should be read accordingly.
