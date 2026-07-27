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
