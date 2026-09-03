# Infra notes

## systemd timers (recommended starting point, per the plan)

```bash
sudo cp infra/systemd/price-ingest.service infra/systemd/price-ingest.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now price-ingest.timer
systemctl list-timers | grep price-ingest   # verify it's scheduled
```

Add one `.service` + `.timer` pair per ingestion job (fundamentals, news,
macro calendar refresh, feature build, retraining) following the same
pattern. Keep each one-shot and idempotent so a missed run just gets caught
by the next one, or by `Persistent=true` firing it on next boot.

## Every unit a complete self-hosted deployment needs

Ingestion alone doesn't trade anything — a complete deployment needs the
trading units too. `infra/systemd/` ships:

| Unit(s) | What it runs | Schedule |
|---|---|---|
| `price-ingest.service` + `.timer` | `data.ingest.prices` | daily, after close |
| `macro-calendar-refresh.service` + `.timer` | `data.ingest.macro_calendar` | monthly |
| `news-stream.service` (no timer — long-running) | `data.ingest.news_stream` | continuous |
| `weekly-cycle.service` + `.timer` | `scripts.run_weekly_cycle` (universe refresh, ingest, screen, trade) | weekly, Monday 08:00 UTC |
| `contradiction-monitor.service` + `.timer` | `execution.contradiction_monitor` (mid-week emergency close check) | hourly, weekday market hours |

The last two are the actual trading cycle — without `weekly-cycle` and
`contradiction-monitor` enabled, the ingestion units above just keep the
database fresh; nothing ever screens or places an order. See
`infra/launchd/README.md` for the equivalent macOS jobs (same five units,
scheduled via `launchd` instead) and `infra/railway/README.md` for how the
same commands map onto Railway's own cron services.

Install the two trading units the same way as `price-ingest` above:

```bash
sudo cp infra/systemd/weekly-cycle.service infra/systemd/weekly-cycle.timer /etc/systemd/system/
sudo cp infra/systemd/contradiction-monitor.service infra/systemd/contradiction-monitor.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now weekly-cycle.timer contradiction-monitor.timer
systemctl list-timers | grep -E 'weekly-cycle|contradiction-monitor'
```

## `news-stream.service` (the one long-running exception)

Every other job above is a one-shot `.service` fired by a `.timer` on a
schedule. `news-stream.service` is different on purpose: it holds Alpaca's
news websocket open and writes to `news_events` continuously instead of on
a poll (see `data/ingest/news_stream.py`), so it has no matching `.timer` —
it's `Type=simple`, `Restart=always`, meant to just stay up:

```bash
sudo cp infra/systemd/news-stream.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now news-stream.service
systemctl status news-stream.service   # verify it's running, not scheduled
```

It needs an Alpaca API key configured (`ALPACA_PAPER_API_KEY` /
`ALPACA_PAPER_SECRET_KEY`, or the `_LIVE_` variants) even when `BROKER=ibkr`
— it only reads news, never places an order, and Alpaca's free paper keys
are enough on their own. This is a data-freshness job only: it does not
change how often the model screens or trades, which stays the weekly cycle
plus the existing hourly contradiction monitor.

## Prefect (optional)

The plan allows "one lightweight Prefect flow" if the team wants a UI over
the pipeline instead of `systemctl list-timers`. Not included here to avoid
committing to it before you know if cron/systemd is enough — Prefect Cloud's
free tier is enough to wrap the same scripts if you do want it later.

## Terraform (optional, once you outgrow manual VM setup)

Deliberately empty at this stage — the plan calls this optional, and a
single VM provisioned by hand is the right amount of infra for a 3-person
team before there's live P&L. Add Terraform here if/when you're managing
more than one environment (e.g. separate paper vs. live VMs) and manual
setup starts causing drift between them.
