# macOS scheduling (launchd)

`infra/systemd/` only works on Linux — this machine is macOS, so the jobs
below are scheduled via `launchd` instead. A LaunchAgent (not a
LaunchDaemon) is used deliberately: it only runs while you're logged in,
which is the right scope for a personal-machine paper-trading job — no root,
no running unattended before anyone's confirmed the Mac is actually on.

Two jobs:
- **`weekly-cycle`** — full universe refresh, re-ingest, retrain, screen and
  trade. Runs once a week.
- **`contradiction-monitor`** — checks currently held positions against
  fresh news sentiment and short-term price momentum, closing anything the
  evidence has turned against. Runs hourly, no-ops itself outside market
  hours (see `execution/contradiction_monitor.py`) rather than relying on a
  DST-sensitive fixed schedule.

## Install

```bash
mkdir -p logs
cp infra/launchd/com.trading-system.weekly-cycle.plist ~/Library/LaunchAgents/
cp infra/launchd/com.trading-system.contradiction-monitor.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.trading-system.weekly-cycle.plist
launchctl load ~/Library/LaunchAgents/com.trading-system.contradiction-monitor.plist
```

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
```

## Uninstall

```bash
launchctl unload ~/Library/LaunchAgents/com.trading-system.weekly-cycle.plist
rm ~/Library/LaunchAgents/com.trading-system.weekly-cycle.plist

launchctl unload ~/Library/LaunchAgents/com.trading-system.contradiction-monitor.plist
rm ~/Library/LaunchAgents/com.trading-system.contradiction-monitor.plist
```

## Notes

- The plist hardcodes the absolute path to this repo and its `.venv`
  (`/Users/kp/Downloads/trading-system`) — if you move the repo, update the
  `ProgramArguments` and `WorkingDirectory` paths and reload.
- Only fires while this Mac is on, awake, and you're logged in. If the Mac
  is asleep at the scheduled time, launchd runs it as soon as the Mac wakes
  (it doesn't just skip the run), but if it's fully off, that week's cycle
  is missed — check `logs/weekly-cycle.log` periodically.
- `--feature-set-id v3` and `--top-k 10` are baked into the plist; edit
  `ProgramArguments` and reload (`launchctl unload` + `launchctl load`) to
  change them.
