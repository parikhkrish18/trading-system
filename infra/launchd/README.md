# macOS scheduling (launchd)

`infra/systemd/` only works on Linux — this machine is macOS, so the jobs
below are scheduled via `launchd` instead. A LaunchAgent (not a
LaunchDaemon) is used deliberately: it only runs while you're logged in,
which is the right scope for a personal-machine paper-trading job — no root,
no running unattended before anyone's confirmed the Mac is actually on.

Three jobs:
- **`weekly-cycle`** — full universe refresh, re-ingest, retrain, screen and
  trade. Runs once a week.
- **`contradiction-monitor`** — checks currently held positions against
  fresh news sentiment, short-term price momentum, and each position's own
  take-profit/stop-loss, closing anything the evidence (or the position's
  own resolution) has turned against. Runs hourly, no-ops itself outside
  market hours (see `execution/contradiction_monitor.py`) rather than
  relying on a DST-sensitive fixed schedule.
- **`news-stream`** — a long-lived process, not a periodic job: it holds
  Alpaca's news websocket open and writes headlines to `news_events`
  continuously instead of on a weekly poll (see
  `data/ingest/news_stream.py`). `KeepAlive` restarts it if it ever exits;
  `RunAtLoad` starts it as soon as the plist is loaded. This is a
  data-freshness job only — it does not screen or trade; that stays on
  `weekly-cycle` plus the hourly `contradiction-monitor` above.

## Install

```bash
mkdir -p logs
cp infra/launchd/com.trading-system.weekly-cycle.plist ~/Library/LaunchAgents/
cp infra/launchd/com.trading-system.contradiction-monitor.plist ~/Library/LaunchAgents/
cp infra/launchd/com.trading-system.news-stream.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.trading-system.weekly-cycle.plist
launchctl load ~/Library/LaunchAgents/com.trading-system.contradiction-monitor.plist
launchctl load ~/Library/LaunchAgents/com.trading-system.news-stream.plist
```

`news-stream` needs an Alpaca API key configured (`ALPACA_PAPER_API_KEY` /
`ALPACA_PAPER_SECRET_KEY`, or the `_LIVE_` variants) even if `BROKER=ibkr` —
it only reads news, it never places an order, and Alpaca's paper keys are
free and enough on their own.

## Verify it's loaded

```bash
launchctl list | grep trading-system
```

## Trigger a job manually once (don't wait for the schedule to prove it works)

```bash
launchctl start com.trading-system.weekly-cycle
tail -f logs/weekly-cycle.log logs/weekly-cycle.error.log

launchctl start com.trading-system.contradiction-monitor
tail -f logs/contradiction-monitor.log logs/contradiction-monitor.error.log

launchctl start com.trading-system.news-stream
tail -f logs/news-stream.log logs/news-stream.error.log
```

## Uninstall

```bash
launchctl unload ~/Library/LaunchAgents/com.trading-system.weekly-cycle.plist
rm ~/Library/LaunchAgents/com.trading-system.weekly-cycle.plist

launchctl unload ~/Library/LaunchAgents/com.trading-system.contradiction-monitor.plist
rm ~/Library/LaunchAgents/com.trading-system.contradiction-monitor.plist

launchctl unload ~/Library/LaunchAgents/com.trading-system.news-stream.plist
rm ~/Library/LaunchAgents/com.trading-system.news-stream.plist
```

## Notes

- The plist hardcodes the absolute path to this repo and its `.venv`
  (`/Users/kp/Downloads/trading-system`) — if you move the repo, update the
  `ProgramArguments` and `WorkingDirectory` paths and reload.
- Only fires while this Mac is on, awake, and you're logged in. If the Mac
  is asleep at the scheduled time, launchd runs it as soon as the Mac wakes
  (it doesn't just skip the run), but if it's fully off, that week's cycle
  is missed — check `logs/weekly-cycle.log` periodically. `news-stream` is
  the same story in reverse: it just stops collecting news while the Mac is
  off/asleep and picks back up on wake/login, it doesn't backfill the gap.
- `--feature-set-id v4` is baked into the plist; edit `ProgramArguments`
  and reload (`launchctl unload` + `launchctl load`) to change it.
  `scripts/run_weekly_cycle.py` has no `--top-k` flag — the diversified
  book's size is `SCREENER_TOP_K` (config/settings.py, default 10), an env
  var read at runtime, not a CLI argument, so it's set in `.env`, not here.
