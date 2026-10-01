"""
Redeploys capital freed by a mid-week contradiction close, or already
sitting idle, via the same confidence bar and selection logic as the
weekly screen (execution/contradiction_monitor.py's own
_attempt_reactivation).

Split out from execution/contradiction_monitor.py's hourly pass
deliberately: the ensemble retrain this requires (models/screener.py's
run_screen, deliberately never cached -- see full_book_rebalance.py's
module docstring for why a stale cached pool was tried and removed)
routinely takes 40-50+ minutes with no way to bound or interrupt it
partway through. Running it inside the hourly contradiction-check risked
that check's own pass still being Active when the next hourly cron mark
came around -- and Railway's cron does not overlap or kill a still-running
execution, it silently SKIPS the next firing outright when the previous
one is still Active (see
docs.railway.com/cron-jobs#service-execution-requirements). That meant a
slow retrain could cause an entire hour of contradiction/stop-loss/
circuit-breaker monitoring to quietly not run at all -- confirmed live
from a run that went silent for the better part of an hour deep in this
same retrain, finishing only narrowly before the next hourly mark.

This job runs on its own, much less frequent schedule instead (see
infra/railway/README.md), so the hourly contradiction check stays fast and
reliable, and this retrain gets a wide enough window to actually finish.
Trade-off: capital freed by a contradiction close now waits for this job's
own next scheduled pass, not the same hourly cycle that closed it, before
being redeployed -- sitting briefly in cash is safe; a reliable hourly
safety check matters more.

Safety boundary, same as contradiction_monitor.py/trading_loop.py: only
ever calls get_broker() without confirm_live=True, so it can never fire a
live order on the MASTER account no matter what TRADING_MODE is set to.

Deploy this behind monitoring/dashboard/Dockerfile, never a plain Railpack
Python build -- see scripts/run_weekly_cycle.py's docstring for why
(LightGBM's libgomp.so.1 dependency, only present in that Dockerfile).

Usage:
    python -m execution.reactivation_monitor
"""
from __future__ import annotations

import logging

from data.ingest.db import get_engine
from execution.approval_gate import advisory_lock
from execution.broker import get_broker
from execution.contradiction_monitor import _attempt_reactivation
from execution.trading_loop import _flatten_and_alert, _run_breaker_check
from monitoring.alerts import configure_file_logging
from monitoring.equity import record_equity_snapshot

logger = logging.getLogger(__name__)

# Own key, distinct from contradiction_monitor.py's _CONTRADICTION_LOCK_KEY
# (a different resource -- that one guards the hourly check, this one
# guards this job's own, much slower and less frequent pass) and from
# approval_gate.APPROVAL_LOCK_KEY (Telegram's single-consumer getUpdates
# poll). This job's own schedule already gives the retrain a wide window,
# but a run that somehow still spills past its own next scheduled fire
# must not overlap with itself and double-submit orders against the same
# freed capital -- same advisory_lock pattern contradiction_monitor.py
# already uses for itself.
_REACTIVATION_LOCK_KEY = 903219


def run_reactivation(request_fn=None) -> None:
    """See module docstring for why this runs on its own, rather than inline from the hourly contradiction check."""
    with advisory_lock(_REACTIVATION_LOCK_KEY) as got_lock:
        if not got_lock:
            logger.warning(
                "Another reactivation pass is already running (advisory lock held) — "
                "skipping this run rather than double-processing. It will run again next scheduled pass."
            )
            return
        _run_reactivation(request_fn)


def _run_reactivation(request_fn=None) -> None:
    """The actual pass, run under run_reactivation's advisory lock — see its docstring."""
    broker = get_broker()  # never passes confirm_live=True — paper-only by construction
    engine = get_engine()

    if hasattr(broker, "client") and not broker.client.get_clock().is_open:
        logger.info("Market is closed — skipping this pass.")
        return

    # Same master-account circuit breakers execution/contradiction_monitor.py
    # runs every hour -- checked here too so this job never opens a fresh
    # position right into (or right after) a breach its hourly sibling
    # hasn't had a chance to react to yet. A trip here flattens exactly like
    # a weekly-cycle/hourly-monitor trip does and skips reactivation this pass.
    breaker_triggers = _run_breaker_check(broker, engine)
    if breaker_triggers:
        reasons = "; ".join(r.reason for r in breaker_triggers)
        _flatten_and_alert(broker, reasons)  # already snapshots equity post-flatten, see trading_loop._flatten_and_alert
        return

    # Best-effort, same as the hourly monitor's own snapshot -- a missed
    # snapshot shouldn't take down the actual reactivation attempt below.
    try:
        record_equity_snapshot(broker.get_portfolio_value(), mode=broker.mode)
    except Exception:
        logger.exception("Could not record this pass's equity snapshot — continuing with reactivation.")

    _attempt_reactivation(broker, engine, request_fn=request_fn, excluded_symbols=set())


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    configure_file_logging()  # logs survive the console closing
    run_reactivation()
    print("Reactivation pass done.")


if __name__ == "__main__":
    main()
