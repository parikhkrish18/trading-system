# Running this system on Railway

Three services, one image. All three are deployed from the same repo and
the same Dockerfile (`monitoring/dashboard/Dockerfile`, selected by
`railway.toml` at the repo root); they differ only in their start command
and schedule. One image means the process that placed an order is
byte-for-byte the process the dashboard reports on.

| Service | Start command | Schedule | Restart policy |
|---|---|---|---|
| `dashboard` | *(leave blank — the image's own `CMD`)* | none, always on | On failure |
| `weekly-cycle` | `python -m scripts.run_weekly_cycle --feature-set-id v4` | `0 8 * * 1` | Never |
| `contradiction-monitor` | `python -m execution.contradiction_monitor` | `0 14-20 * * 1-5` | Never |

Cron times are **UTC** — Railway has no timezone setting.

`weekly-cycle`'s `0 8 * * 1` is Monday 08:00 UTC — 4:00am US/Eastern in
summer, 3:00am in winter — chosen over running the day before (Sunday) as
a deliberate tradeoff: Monday morning picks up whatever news broke
overnight/premarket that a Sunday run would have missed, at the cost of a
tighter (though still real) buffer before the 9:30am open. The full
pipeline (universe refresh, price/fundamentals/news ingestion —
fundamentals+news alone run "roughly two hours" on Polygon's free tier
per scripts/run_weekly_cycle.py's own docstring — sentiment scoring,
feature build, then the actual screen-and-trade cycle) is genuinely
multi-hour, so 08:00 UTC leaves 5.5–6.5 hours depending on the season —
comfortable for the typical run, not the ~day of slack a Sunday start
would have had. If it ever does run long enough to still be going at
9:30am ET, nothing is lost or silently skipped: submit_target_position()
(execution/broker_alpaca.py) checks the market clock at submit time, not
once at the start of the cycle, so an order that would have queued as a
premarket DAY order instead goes out as a regular live market order the
moment the pipeline reaches it — later in Monday's session rather than
right at the open, but still Monday, still automatic. Since `APPROVAL_MODE`
defaults to `auto` and neither `weekly-cycle` nor `contradiction-monitor`
override it (see "What hosting does not change" below), none of this
waits on a human reply — no one needs to be awake to approve anything, in
any timezone, regardless of how long a given week's run takes.

To run a cycle immediately rather than waiting for the next Monday
trigger — e.g. to catch up after a schedule change made once Monday's
08:00 UTC tick has already passed for that week — use the "Deploy" action
on the `weekly-cycle` service in the Railway dashboard (three-dot menu on
the service, or its Deployments tab). That's the supported way to fire a
cron-configured Railway service on demand; the public API's `redeploy`
only rebuilds/re-registers the service's current deployment; it does not
itself invoke the start command outside of an actual cron tick (confirmed
empirically — a `redeploy` call and a temporary near-term one-off cron
value both produced no process output, where a real invocation logs
immediately on start).

`0 14-20 * * 1-5` (contradiction-monitor) fires hourly, on the hour, at
UTC 14:00 through 20:00. Neither cron expression moves for daylight saving
— that's expected for `weekly-cycle` given how much slack it has either
side, but it does shift `contradiction-monitor`'s actual coverage of the
US/Eastern session (9:30am–4:00pm) twice a year:

- **Summer (EDT, UTC-4):** firings land at 10am–4pm ET — misses the first
  30 minutes after the open, catches the close.
- **Winter (EST, UTC-5):** firings land at 9am–3pm ET — covers the open
  (and the 30 minutes before it), but misses the last hour before the
  close.

That's a pre-existing gap in contradiction-monitor's own schedule, not
something this change touches — worth knowing about if a position that
should have been closed late in a winter trading day wasn't caught until
the next hourly check.

The scheduled services must have their restart policy set to **Never**. A
cron job is supposed to exit; "On failure" would restart the weekly cycle
in a loop, and every restart runs the cycle again — trades execute
immediately by default (see "What hosting does not change" below), so a
restart loop here means repeated, unintended trading, not just repeated
messages.

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
| `DASHBOARD_PASSWORD` | secret | any long random string |
| `FEATURE_SET_ID` | `v4` | must match what features were built with |

`DASHBOARD_HOST` is already `0.0.0.0` in the image and does not need
setting. **Do not set `PORT`** — Railway injects it and routes the domain
to whatever it chose; overriding it points the health check at a dead port.
`APPROVAL_MODE` already defaults to `auto` (trade immediately, notify
Telegram after) and does not need setting either — only add it if you
deliberately want the old `telegram` pre-trade approval gate back, in
which case also set `APPROVAL_TIMEOUT_S` (must stay under 3600 — the
hourly monitor shares the one bot).

Left unset on purpose: `MLFLOW_TRACKING_URI` (no MLflow server is deployed;
the two dashboard analysis panels that use it degrade to empty and nothing
else touches it), `POLYGON_API_KEY`, `ANTHROPIC_API_KEY`, `FRED_API_KEY`,
`SLACK_WEBHOOK_URL`. Each of those degrades gracefully when blank — the
corresponding features are simply absent, and LightGBM handles the missing
columns natively.

## What the password protects

`DASHBOARD_PASSWORD` gates **everything** — the static page itself, every
`/api` route (reads as much as the state-changing `POST /api/tests/run`),
all behind one login page at `/login`. Once the dashboard is bound to a
public interface it is not optional: with a
non-loopback bind and no password,
`monitoring/dashboard/server.py::_check_dashboard_auth` refuses outright
(503) rather than leaving the interface one blank variable from being open.

Reads are gated because of what they return — every open position and its
size, the model's reasoning for holding it, and the equity curve. On paper
money that is only embarrassing; the point is that publishing it must not
become the habit before real money, and a URL that was public for months
does not quietly become private later.

The gate is one middleware in front of every request — the page, mounted
static files, and every declared route alike — so an endpoint added later
is private by default instead of private only if its author remembered.

On a hosted dashboard, open the URL and you land on `/login`: type the
password once and a session cookie covers everything else (the page, every
panel, every button) until you log out or the password changes — no
separate token to paste in anywhere. There used to be a second, coarser
HTTP Basic Auth prompt on top of a per-endpoint operator token; both are
gone in favor of this one gate. On localhost nothing changes: a loopback
bind with no password configured needs no ceremony.

## What hosting does not change

Hosting does not change what `APPROVAL_MODE` does — it just means the
default (`auto`) now runs unattended for real, on a schedule, rather than
on a laptop when someone happens to trigger it. In `auto` mode
`scripts.run_weekly_cycle` and `execution.contradiction_monitor` both
execute their approved proposals immediately, no human reply required, and
send a "cycle complete" style message to Telegram (if configured —
best-effort, never blocking) once orders have actually been submitted, not
before. See the `execution/approval_gate.py` module docstring for the full
picture.

Switching a service to `APPROVAL_MODE=telegram` brings back the old
pre-trade human gate — every open/close needs an "approve"/"reject" reply
on the phone before it executes, and blank Telegram credentials reject the
whole batch rather than auto-approving. In that mode the weekly cycle
blocks for up to `APPROVAL_TIMEOUT_S` waiting for a reply, which is normal
and Railway tolerates it: a cron service is a deployment that runs to
completion, with no request timeout, and Railway will not start a second
execution while one is still running. Also in that mode: do not run two
replicas of a scheduled service — the approval poll holds a Postgres
advisory lock, so a second copy rejects its whole batch rather than
stealing the first one's Telegram replies — safe, but silently a no-op.
