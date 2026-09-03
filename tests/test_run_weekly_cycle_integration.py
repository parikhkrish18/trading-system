"""
End-to-end wiring test for scripts/run_weekly_cycle.py::main() -- the actual
entrypoint Railway's cron service invokes. tests/test_weekly_cycle_guard.py
already covers run_guarded_trading_cycle's own logic in isolation; this file
is the layer above it: does main() call every job in the right order, with
the right arguments, does a mid-pipeline vendor failure still let later jobs
(and the trading cycle) run per run_job's isolation contract, and does an
empty universe abort the whole cycle before anything else happens.

Every job function is monkeypatched at the scripts.run_weekly_cycle module
level (main() looks them up as bare names at call time, so this is enough --
no need to patch the underlying data.ingest.* etc. modules directly) and
records its name into a shared list so call order is directly assertable.
"""
from __future__ import annotations

import datetime as dt

import pytest

from scripts import run_weekly_cycle as rwc


@pytest.fixture(autouse=True)
def _no_file_logging(monkeypatch):
    """main() calls configure_file_logging() first -- keep tests from touching a real log file."""
    monkeypatch.setattr(rwc, "configure_file_logging", lambda: None)


@pytest.fixture
def _calls():
    return []


def _recorder(calls, name, return_value=None, raises: Exception | None = None):
    def _fn(*args, **kwargs):
        calls.append((name, args, kwargs))
        if raises is not None:
            raise raises
        return return_value

    return _fn


def _wire_happy_path(monkeypatch, calls, symbols=("AAPL", "MSFT")):
    monkeypatch.setattr(rwc, "refresh_universe", _recorder(calls, "universe_refresh"))
    monkeypatch.setattr(rwc, "load_active_universe", lambda: list(symbols))
    monkeypatch.setattr(rwc, "ingest_prices", _recorder(calls, "price_ingest"))
    monkeypatch.setattr(rwc, "ingest_fundamentals", _recorder(calls, "fundamentals_ingest"))
    monkeypatch.setattr(rwc, "ingest_news", _recorder(calls, "news_ingest"))
    monkeypatch.setattr(rwc, "backfill_unscored_news", _recorder(calls, "sentiment_backfill"))
    monkeypatch.setattr(rwc, "refresh_macro_calendar", _recorder(calls, "macro_calendar_refresh"))
    monkeypatch.setattr(rwc, "build_and_store", _recorder(calls, "build_features"))
    monkeypatch.setattr(
        rwc, "run_guarded_trading_cycle", _recorder(calls, "trading_cycle", return_value="cycle-done")
    )
    monkeypatch.setattr(rwc, "alert_pipeline_failure", _recorder(calls, "alert"))


def _run_main(monkeypatch, argv_extra=()):
    monkeypatch.setattr(
        "sys.argv", ["run_weekly_cycle.py", "--feature-set-id", "v4", *argv_extra]
    )
    rwc.main()


# --------------------------------------------------------------------------
# Happy path: every job runs, in order, ending with the trading cycle.
# --------------------------------------------------------------------------


def test_full_cycle_runs_every_job_in_order(monkeypatch, _calls):
    _wire_happy_path(monkeypatch, _calls)

    _run_main(monkeypatch)

    job_order = [name for name, _args, _kwargs in _calls]
    assert job_order == [
        "universe_refresh",
        "price_ingest",
        "fundamentals_ingest",
        "news_ingest",
        "sentiment_backfill",
        "macro_calendar_refresh",
        "build_features",
        "trading_cycle",
    ]


def test_price_ingest_includes_the_regime_proxy_and_active_universe(monkeypatch, _calls):
    _wire_happy_path(monkeypatch, _calls, symbols=("AAPL", "MSFT"))

    _run_main(monkeypatch)

    price_ingest_call = next(c for c in _calls if c[0] == "price_ingest")
    symbols_arg = price_ingest_call[1][0]
    assert set(symbols_arg) == {"AAPL", "MSFT", "SPY"}  # _REGIME_PROXY = "SPY"


def test_default_backfill_is_a_seven_day_top_up_not_a_full_history_pull(monkeypatch, _calls):
    """--backfill-years defaults to 0, meaning 'just top up the last week', not a fresh full backfill."""
    _wire_happy_path(monkeypatch, _calls)

    _run_main(monkeypatch)

    price_ingest_call = next(c for c in _calls if c[0] == "price_ingest")
    start_date = price_ingest_call[1][1]
    today = dt.datetime.now(tz=dt.UTC).date()
    assert (today - start_date).days == 7


def test_dry_run_flag_reaches_the_trading_cycle(monkeypatch, _calls):
    _wire_happy_path(monkeypatch, _calls)

    _run_main(monkeypatch, argv_extra=["--dry-run"])

    trading_cycle_call = next(c for c in _calls if c[0] == "trading_cycle")
    assert trading_cycle_call[1] == ("v4", ["AAPL", "MSFT"], True)


def test_build_features_gets_the_requested_feature_set_id(monkeypatch, _calls):
    _wire_happy_path(monkeypatch, _calls)

    _run_main(monkeypatch)

    build_features_call = next(c for c in _calls if c[0] == "build_features")
    assert build_features_call[1] == (["AAPL", "MSFT"], "v4")


# --------------------------------------------------------------------------
# A dead vendor mid-pipeline must not take the rest of the week down --
# run_job's isolation contract, exercised end-to-end through main().
# --------------------------------------------------------------------------


def test_a_failed_ingest_job_does_not_stop_later_jobs_or_the_trading_cycle(monkeypatch, _calls):
    _wire_happy_path(monkeypatch, _calls)
    monkeypatch.setattr(
        rwc, "ingest_news", _recorder(_calls, "news_ingest", raises=ConnectionError("polygon is down"))
    )

    _run_main(monkeypatch)

    job_order = [name for name, _args, _kwargs in _calls]
    # news_ingest itself still "ran" (and raised), but everything after it
    # -- including the trading cycle -- must still have gone ahead.
    assert "news_ingest" in job_order
    assert "sentiment_backfill" in job_order
    assert "macro_calendar_refresh" in job_order
    assert "build_features" in job_order
    assert "trading_cycle" in job_order
    # And the failure must have been surfaced, not swallowed silently.
    alert_calls = [c for c in _calls if c[0] == "alert"]
    assert any(c[1][0] == "news_ingest" for c in alert_calls)


def test_a_failed_fundamentals_job_still_lets_price_ingest_and_trading_proceed(monkeypatch, _calls):
    _wire_happy_path(monkeypatch, _calls)
    monkeypatch.setattr(
        rwc,
        "ingest_fundamentals",
        _recorder(_calls, "fundamentals_ingest", raises=RuntimeError("rate limited")),
    )

    _run_main(monkeypatch)

    job_order = [name for name, _args, _kwargs in _calls]
    assert job_order.index("price_ingest") < job_order.index("fundamentals_ingest")
    assert "trading_cycle" in job_order


# --------------------------------------------------------------------------
# Empty universe: the one failure mode that must abort everything else,
# rather than being swallowed like a normal ingest job failure.
# --------------------------------------------------------------------------


def test_empty_universe_aborts_before_any_other_job_runs(monkeypatch, _calls):
    monkeypatch.setattr(rwc, "refresh_universe", _recorder(_calls, "universe_refresh"))
    monkeypatch.setattr(rwc, "load_active_universe", lambda: [])
    monkeypatch.setattr(rwc, "alert_pipeline_failure", _recorder(_calls, "alert"))
    # None of these should ever be called -- fail loudly if they are.
    for name in (
        "ingest_prices", "ingest_fundamentals", "ingest_news", "backfill_unscored_news",
        "refresh_macro_calendar", "build_and_store", "run_guarded_trading_cycle",
    ):
        monkeypatch.setattr(rwc, name, _recorder(_calls, name))

    _run_main(monkeypatch)

    job_order = [name for name, _args, _kwargs in _calls]
    assert job_order == ["universe_refresh", "alert"]
    alert_call = next(c for c in _calls if c[0] == "alert")
    assert alert_call[1][0] == "weekly_cycle"
    assert "universe" in alert_call[1][1].lower()
