"""
Continuously scores news_events rows as they arrive, instead of waiting
for execution/contradiction_monitor.py's next hourly pass to even attempt
it. contradiction_monitor.py no longer calls backfill_unscored_news at
all -- this process owns ongoing sentiment scoring, so a headline scores
within minutes of landing rather than sitting unscored for up to an hour.
(scripts/run_weekly_cycle.py keeps its own backfill_unscored_news call as
a correctness safety net for the weekly screen's features -- see that
module. Running alongside this worker, it becomes a fast no-op in the
normal case, since there's rarely anything left unscored by the time it
runs.)

Batches on whichever trigger comes first:
  - _BATCH_SIZE (20) unscored rows pending -- the same batch size
    features/qualitative/sentiment.py's score_sentiment already uses per
    Claude call, so a burst of news scores at full batch efficiency with
    no added latency.
  - _MAX_WAIT_SECONDS since the OLDEST currently-unscored row arrived
    (its ingested_at, not the headline's own published ts -- a historical
    re-ingest with old published timestamps must not look "overdue" the
    instant it lands).

Deliberately NOT a flat "poll every N seconds and score whatever's
pending" loop: during a quiet stretch, that would burn a full Claude call
(the ~930-token system prompt, uncached since the gap between polls would
usually exceed the 5-minute cache TTL, plus its own per-call cost) to
score just one or two headlines, over and
over, for no freshness benefit real urgency actually needs. The
_MAX_WAIT_SECONDS bound below gives the same "never stuck for an hour"
guarantee without that cost -- a lone urgent headline during a quiet
stretch still scores within _MAX_WAIT_SECONDS, just not within the next
30-second poll tick.

Meant to run as one long-lived process (a systemd service, macOS launchd
LaunchAgent, or a Railway service with no cron schedule -- see
data/ingest/news_stream.py for the identical "always-on, not a cron job"
pattern this follows), not a cron/timer job: a strict timer would either
score tiny batches constantly (if frequent) or reintroduce the same
hours-long freshness lag this exists to fix (if infrequent).

Usage:
    python -m features.qualitative.sentiment_worker
"""
from __future__ import annotations

import logging
import time

import pandas as pd

from data.ingest.db import get_engine
from execution.approval_gate import advisory_lock, send_followup
from features.qualitative.sentiment import backfill_unscored_news
from monitoring.alerts import configure_file_logging

logger = logging.getLogger(__name__)

# How often to check whether either trigger below has fired. Cheap: just a
# COUNT/MIN query, not a Claude call -- how tight this is has no bearing on
# API spend, only on how quickly a growing backlog gets noticed.
_POLL_INTERVAL_SECONDS = 30.0

# Score immediately once this many rows are pending, same as
# score_sentiment's own per-call batch size -- a burst of news scores at
# full efficiency with no added latency over the old hourly-only design.
_BATCH_SIZE = 20

# Otherwise, score whatever is pending once the OLDEST of it has been
# waiting this long. Caps worst-case freshness lag during a quiet stretch
# (a single urgent headline that never reaches 20 siblings) at 5 minutes --
# far below the up-to-60-minute wait the old hourly-only design had --
# without triggering a Claude call on every 30-second poll tick.
_MAX_WAIT_SECONDS = 5 * 60

# Own key, distinct from contradiction_monitor.py's _CONTRADICTION_LOCK_KEY
# and reactivation_monitor.py's _REACTIVATION_LOCK_KEY (different
# resources). Guards against two copies of this same long-lived process
# overlapping -- e.g. a redeploy briefly running the old and new instance
# side by side -- and double-paying Claude to score the same rows twice.
_SENTIMENT_WORKER_LOCK_KEY = 903220


def _pending_unscored(engine) -> tuple[int, float | None]:
    """(count, seconds since the oldest pending row's ingested_at) for unscored rows. age is None when count is 0."""
    df = pd.read_sql(
        "SELECT count(*) AS n, extract(epoch FROM now() - min(ingested_at)) AS oldest_age_s "
        "FROM news_events WHERE sentiment IS NULL",
        engine,
    )
    row = df.iloc[0]
    count = int(row["n"])
    age = float(row["oldest_age_s"]) if count and pd.notna(row["oldest_age_s"]) else None
    return count, age


def run_once(engine=None) -> int:
    """
    Scores one pending batch if either trigger (see module docstring) has
    fired, else no-ops. Returns how many rows were scored (0 if neither
    trigger fired or nothing is pending).

    Sends one Telegram line per batch actually scored -- the "it's doing
    stuff" signal this otherwise silent, always-on process would have no
    other way to give. In steady state that's naturally spaced out (one
    batch roughly every time 20 headlines accumulate, or every
    _MAX_WAIT_SECONDS during a quiet stretch); the one case this messages
    more often than that is catching up a real backlog (a cold start, a
    gap after a restart) -- several batches in quick succession there,
    which is itself useful to see rather than noise to suppress.
    """
    engine = engine or get_engine()
    count, oldest_age = _pending_unscored(engine)
    if count == 0:
        return 0
    if count < _BATCH_SIZE and (oldest_age is None or oldest_age < _MAX_WAIT_SECONDS):
        return 0
    scored = backfill_unscored_news(batch_size=_BATCH_SIZE)
    if scored:
        send_followup(f"📰 Scored {scored} headline(s).")
    return scored


def run_forever(poll_interval: float = _POLL_INTERVAL_SECONDS) -> None:
    """Blocks forever (until interrupted), checking the two triggers on each tick."""
    logger.info(
        "Starting continuous sentiment scoring (batch of %d, or oldest pending row waiting over %ds).",
        _BATCH_SIZE, _MAX_WAIT_SECONDS,
    )
    while True:
        try:
            try:
                with advisory_lock(_SENTIMENT_WORKER_LOCK_KEY) as got_lock:
                    if not got_lock:
                        logger.warning("Another sentiment-worker pass is already running — skipping this tick.")
                        scored = 0
                    else:
                        scored = run_once()
                        if scored:
                            logger.info("Scored %d headline(s) this pass.", scored)
            except Exception:
                logger.exception("Sentiment-worker tick failed — retrying next poll.")
                scored = 0
            # A full batch just went out -- more may already be queued
            # behind it (catching up after a gap, or a burst outrunning the
            # poll interval). Check again immediately instead of sleeping,
            # so recovery from a backlog doesn't cost a full poll_interval
            # per 20 rows; a partial batch, nothing at all, or a failed
            # tick means business as usual, so wait out the normal
            # interval. One sleep call site, covered by the one
            # KeyboardInterrupt handler below, so Ctrl+C during the sleep
            # itself also breaks the loop cleanly rather than crashing out
            # of it.
            if scored < _BATCH_SIZE:
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            break


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    configure_file_logging()  # logs survive the console closing
    run_forever()


if __name__ == "__main__":
    main()
