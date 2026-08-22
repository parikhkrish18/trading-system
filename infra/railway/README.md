# Running this system on Railway

Three services, one image. All three are deployed from the same repo and
the same Dockerfile (`monitoring/dashboard/Dockerfile`, selected by
`railway.toml` at the repo root); they differ only in their start command
and schedule. One image means the process that placed an order is
byte-for-byte the process the dashboard reports on.

| Service | Start command | Schedule | Restart policy |
|---|---|---|---|
| `dashboard` | *(leave blank — the image's own `CMD`)* | none, always on | On failure |
| `weekly-cycle` | `python -m scripts.run_weekly_cycle --feature-set-id v4` | `0 22 * * 1` | Never |
| `contradiction-monitor` | `python -m execution.contradiction_monitor` | `0 14-20 * * 1-5` | Never |

Cron times are **UTC** — Railway has no timezone setting. `0 22 * * 1` is
Monday 18:00 US/Eastern in summer and 17:00 in winter; both are after the
16:00 close, which is what matters. The monitor window covers roughly the
market session (13:30–20:00 UTC in summer). Daylight saving shifts the
local time by an hour twice a year and nothing else — do not "fix" it by
editing the cron unless the run drifts across the close.

The scheduled services must have their restart policy set to **Never**. A
cron job is supposed to exit; "On failure" would restart the weekly cycle
in a loop, and every restart re-sends trade proposals to Telegram.

## Why a Dockerfile and not Railway's automatic detection

Two things auto-detection gets wrong here:

- **LightGBM needs a system library.** Its wheel `dlopen()`s `libgomp.so.1`
  (OpenMP) at import time. That library is not in the Python base image and
  is not something a `requirements.txt` can pull in. Its absence fails at
  `import lightgbm` — i.e. mid-screening-run, not at build time. The
  Dockerfile installs `libgomp1` explicitly.
- **Python version.** `requirements.txt` is pinned against a 3.12 venv and
  several pins require it (numpy 2.5.1 publishes no wheel below 3.12). The
  `FROM python:3.12-slim` line is the pin; move it and the requirements
  pins together, never one alone.

`psycopg2-binary` bundles its own libpq and needs nothing extra.

## The database

Railway's Postgres is stock Postgres, not TimescaleDB. That is fine:
`data/schema/001_init.sql` wraps every `CREATE EXTENSION timescaledb` and
`create_hypertable(...)` in an existence check, so on stock Postgres the
tables are created as ordinary tables and the hypertable calls are skipped.
Hypertables are a partitioning optimization here, not a correctness
requirement.

Point the app at the database with Railway *variable references* rather
than copied literals, so a credential rotation on the database service
propagates by itself:

```
DB_HOST=${{Postgres.PGHOST}}
DB_PORT=${{Postgres.PGPORT}}
DB_NAME=${{Postgres.PGDATABASE}}
DB_USER=${{Postgres.PGUSER}}
DB_PASSWORD=${{Postgres.PGPASSWORD}}
```

`PGHOST` resolves to the private-network hostname, so database traffic
never leaves Railway and never crosses the public internet.

Initialise a fresh database in this order (each is idempotent, so a
re-run is safe):

```
python -m data.schema.migrate                        # tables and indexes
python -m data.ingest.universe --scrape              # ~503 S&P 500 symbols
python -m scripts.run_daily_ingest --universe --source yfinance
python -m features.build_features --universe --feature-set-id v4
```

Prices come from yfinance deliberately: it is free and unthrottled, and
takes minutes rather than the ~2 hours that Polygon's free-tier rate limit
imposes on fundamentals and news for a 500-symbol universe.

## Environment variables

Set these on **every** service — the scheduled ones need the database and
Telegram just as much as the dashboard does. In Railway, shared values
belong in a project-level shared variable rather than being pasted three
times.

| Variable | Value | Where it comes from |
|---|---|---|
| `DB_*` | see above | Railway Postgres service |
| `BROKER` | `alpaca` | fixed — the default is `ibkr`, which needs a TWS socket on localhost that no container has |
| `TRADING_MODE` | `paper` | fixed |
| `ALPACA_PAPER_API_KEY` | secret | Alpaca paper dashboard |
| `ALPACA_PAPER_SECRET_KEY` | secret | Alpaca paper dashboard |
| `TELEGRAM_BOT_TOKEN` | secret | BotFather |
| `TELEGRAM_CHAT_ID` | negative number | the group's id — negative because it is a group, not a DM |
| `APPROVAL_MODE` | `telegram` | fixed |
| `APPROVAL_TIMEOUT_S` | `900` | must stay under 3600; the hourly monitor shares the one bot |
| `DASHBOARD_API_TOKEN` | secret | generate one; any long random string |
| `FEATURE_SET_ID` | `v4` | must match what features were built with |

`DASHBOARD_HOST` is already `0.0.0.0` in the image and does not need
setting. **Do not set `PORT`** — Railway injects it and routes the domain
to whatever it chose; overriding it points the health check at a dead port.

Left unset on purpose: `MLFLOW_TRACKING_URI` (no MLflow server is deployed;
the two dashboard analysis panels that use it degrade to empty and nothing
else touches it), `POLYGON_API_KEY`, `ANTHROPIC_API_KEY`, `FRED_API_KEY`,
`SLACK_WEBHOOK_URL`. Each of those degrades gracefully when blank — the
corresponding features are simply absent, and LightGBM handles the missing
columns natively.

## What the token protects

`DASHBOARD_API_TOKEN` gates **every** `/api` route — the reads as much as
`POST /api/tests/run` and `POST /api/jobs/*/run`. Once the dashboard is
bound to a public interface it is not optional: with a non-loopback bind
and no token, `monitoring/dashboard/server.py::_require_api_token` refuses
outright rather than leaving the interface one blank variable from being
open.

Reads are gated because of what they return — every open position and its
size, the model's reasoning for holding it, and the equity curve. On paper
money that is only embarrassing; the point is that publishing it must not
become the habit before real money, and a URL that was public for months
does not quietly become private later.

The gate is declared once on the `FastAPI` app rather than per route, so an
endpoint added later is private by default instead of private only if its
author remembered. A test walks the route table and fails if any `/api`
route escapes it.

The static page itself (HTML, JS, CSS) stays reachable — it has to load
before anyone can type a token into it, and it carries no data of its own.

On a hosted dashboard, paste the token into the **Operator token** box at
the top right. It is kept in that browser's localStorage and sent as a
bearer header; the server remains the thing that enforces it. Until it is
entered the page shows a single explanatory banner rather than a dozen
broken panels. On localhost nothing changes: a loopback bind with no token
configured needs no ceremony.

## What hosting does not change

The trading cycle still walks through the Telegram approval gate before any
order exists. The "Run trading cycle" button starts
`scripts/run_weekly_cycle.py` — it authorizes computation, not trades — and
the cron service runs the same entrypoint. `APPROVAL_MODE=auto` is the only
way to bypass the human, and it has to be set deliberately; blank Telegram
credentials in `telegram` mode reject the whole batch rather than
auto-approving.

The weekly cycle blocks for up to `APPROVAL_TIMEOUT_S` waiting for a reply.
That is normal and Railway tolerates it: a cron service is a deployment
that runs to completion, with no request timeout, and Railway will not
start a second execution while one is still running.

Do not run two replicas of a scheduled service. The approval poll holds a
Postgres advisory lock, so a second copy rejects its whole batch rather
than stealing the first one's Telegram replies — safe, but silently a no-op.
