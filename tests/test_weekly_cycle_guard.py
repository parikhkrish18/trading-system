"""
The staleness guard in scripts/run_weekly_cycle.py: run_job swallows every
ingest failure by design, so the guard between ingestion and trading is the
only thing standing between "every vendor silently failed" and "the cycle
trades on last week's data anyway". These tests pin the three behaviors that
matter: stale data aborts before any broker call, fresh data proceeds, and
the abort alerts.
"""
from __future__ import annotations

from unittest.mock import Mock

import pandas as pd

from scripts.run_weekly_cycle import freshness_issues, run_guarded_trading_cycle


def _now():
    return pd.Timestamp.now(tz="UTC")


# --------------------------------------------------------------------------
# freshness_issues — the pure staleness verdict
# --------------------------------------------------------------------------


def test_fresh_data_raises_no_issues():
    latest = {"prices": _now(), "features": _now() - pd.Timedelta(days=1)}
    assert freshness_issues(latest, max_age_days=3) == []


def test_stale_price_row_is_flagged():
    latest = {"prices": _now() - pd.Timedelta(days=10), "features": _now()}
    issues = freshness_issues(latest, max_age_days=3)
    assert len(issues) == 1
    assert "prices" in issues[0]


def test_stale_feature_row_is_flagged():
    latest = {"prices": _now(), "features": _now() - pd.Timedelta(days=10)}
    issues = freshness_issues(latest, max_age_days=3)
    assert len(issues) == 1
    assert "features" in issues[0]


def test_empty_table_is_flagged_not_silently_fresh():
    """An empty vendor response never raised — MAX(ts) is NULL. NULL must read as stale, not as fine."""
    latest = {"prices": None, "features": _now()}
    issues = freshness_issues(latest, max_age_days=3)
    assert len(issues) == 1
    assert "prices" in issues[0]


def test_age_exactly_at_the_limit_passes():
    # check_staleness flags age > max_age_days, so Friday's close on Monday
    # (3 calendar days) still trades with the default of 3.
    latest = {"prices": _now() - pd.Timedelta(days=3), "features": _now()}
    assert freshness_issues(latest, max_age_days=3) == []


def test_tz_naive_timestamps_are_handled():
    latest = {"prices": pd.Timestamp("2020-01-01"), "features": _now()}
    issues = freshness_issues(latest, max_age_days=3)
    assert len(issues) == 1


# --------------------------------------------------------------------------
# run_guarded_trading_cycle — the abort path
# --------------------------------------------------------------------------


def test_stale_data_aborts_before_any_broker_call():
    run_cycle_fn = Mock(name="run_cycle")
    alert_fn = Mock(name="alert")
    result = run_guarded_trading_cycle(
        "v3", ["AAPL"], False,
        run_cycle_fn=run_cycle_fn,
        freshness_fn=lambda: ["prices: latest bar is 12 day(s) old (2026-07-30)"],
        alert_fn=alert_fn,
    )
    assert result is None
    run_cycle_fn.assert_not_called()


def test_the_abort_alerts():
    alert_fn = Mock(name="alert")
    run_guarded_trading_cycle(
        "v3", ["AAPL"], False,
        run_cycle_fn=Mock(),
        freshness_fn=lambda: ["features: latest bar is 9 day(s) old (2026-08-02)"],
        alert_fn=alert_fn,
    )
    alert_fn.assert_called_once()
    job_name, detail = alert_fn.call_args.args
    assert job_name == "data_freshness_guard"
    assert "9 day(s) old" in detail


def test_fresh_data_proceeds_to_the_cycle():
    run_cycle_fn = Mock(name="run_cycle", return_value="cycle-result")
    alert_fn = Mock(name="alert")
    result = run_guarded_trading_cycle(
        "v3", ["AAPL", "MSFT"], True,
        run_cycle_fn=run_cycle_fn,
        freshness_fn=lambda: [],
        alert_fn=alert_fn,
    )
    assert result == "cycle-result"
    run_cycle_fn.assert_called_once_with("v3", ["AAPL", "MSFT"], True)
    alert_fn.assert_not_called()


def test_a_guard_that_cannot_check_refuses_to_trade():
    """DB down at check time must mean abort, not 'assume fresh'."""
    def broken_check():
        raise ConnectionError("connection refused")

    run_cycle_fn = Mock(name="run_cycle")
    alert_fn = Mock(name="alert")
    result = run_guarded_trading_cycle(
        "v3", ["AAPL"], False,
        run_cycle_fn=run_cycle_fn,
        freshness_fn=broken_check,
        alert_fn=alert_fn,
    )
    assert result is None
    run_cycle_fn.assert_not_called()
    alert_fn.assert_called_once()
    assert "could not verify" in alert_fn.call_args.args[1]
