"""
The first-run setup path. Its whole reason to exist is ordering and
history depth — each step is the precondition for the next, and the daily
ingest job's 7-day window silently produces an empty features table on a
fresh database.
"""
from __future__ import annotations

import datetime as dt

import pytest

from scripts import init_database


@pytest.fixture
def _recorded(monkeypatch) -> dict:
    """Stub every step, recording call order and arguments."""
    calls: dict = {"order": []}

    def _migrate():
        calls["order"].append("migrate")

    def _refresh():
        calls["order"].append("universe")
        return 503

    def _load():
        return ["AAPL", "MSFT"]

    def _prices(symbols, start, end, source):
        calls["order"].append("prices")
        calls["prices"] = {"symbols": symbols, "start": start, "end": end, "source": source}
        return 4200

    def _features(symbols, feature_set_id):
        calls["order"].append("features")
        calls["features"] = {"symbols": symbols, "feature_set_id": feature_set_id}
        return 900

    monkeypatch.setattr(init_database, "migrate", _migrate)
    monkeypatch.setattr(init_database, "refresh_universe", _refresh)
    monkeypatch.setattr(init_database, "load_active_universe", _load)
    monkeypatch.setattr(init_database, "ingest_prices", _prices)
    monkeypatch.setattr(init_database, "build_and_store", _features)
    return calls


def test_steps_run_in_the_only_order_that_works(_recorded):
    """
    Schema before universe before prices before features. Any other order
    fails on an empty database: no tables to write the universe into, no
    symbols to fetch prices for, no prices to compute features from.
    """
    init_database.initialise()

    assert _recorded["order"] == ["migrate", "universe", "prices", "features"]


def test_price_history_is_years_deep_not_days(_recorded):
    """
    The daily ingest job pulls 7 days, which is right for topping up and
    useless here: features need years behind them, so a week of prices
    yields an empty features table and a screener with nothing to rank.
    """
    init_database.initialise(years=4)

    span = _recorded["prices"]["end"] - _recorded["prices"]["start"]
    assert span > dt.timedelta(days=3 * 365)


def test_prices_come_from_yfinance(_recorded):
    """Free and unthrottled — minutes, where the rate-limited vendor is hours."""
    init_database.initialise()

    assert _recorded["prices"]["source"] == "yfinance"


def test_the_regime_proxy_is_ingested_alongside_the_universe(_recorded):
    """
    SPY isn't tradeable here, but without it the market-regime read
    silently defaults to CHOP on every cycle.
    """
    init_database.initialise()

    assert "SPY" in _recorded["prices"]["symbols"]


def test_the_regime_proxy_is_not_duplicated_when_already_present(monkeypatch, _recorded):
    monkeypatch.setattr(init_database, "load_active_universe", lambda: ["AAPL", "SPY"])

    init_database.initialise()

    assert _recorded["prices"]["symbols"].count("SPY") == 1


def test_features_are_built_for_the_universe_without_the_proxy(_recorded):
    """SPY is an input to the regime read, not a candidate to be ranked."""
    init_database.initialise()

    assert "SPY" not in _recorded["features"]["symbols"]


def test_feature_set_id_falls_back_to_configured_value(_recorded, monkeypatch):
    monkeypatch.setattr(init_database.settings, "feature_set_id", "v9")

    init_database.initialise()

    assert _recorded["features"]["feature_set_id"] == "v9"


def test_an_empty_universe_stops_before_ingesting_anything(monkeypatch, _recorded):
    """
    A scrape that returns nothing but does not raise would otherwise send
    an empty symbol list into the price fetcher and report success on a
    database with no data in it.
    """
    monkeypatch.setattr(init_database, "load_active_universe", list)

    with pytest.raises(RuntimeError, match="Universe is empty"):
        init_database.initialise()

    assert "prices" not in _recorded["order"]


def test_a_failing_step_stops_the_run_rather_than_carrying_on(monkeypatch, _recorded):
    """
    Unlike the daily/weekly jobs, failures here are NOT isolated: every
    step depends on the one before, so continuing past a failure only
    produces a second, more confusing error.
    """
    def _boom():
        raise RuntimeError("wikipedia changed its table layout")

    monkeypatch.setattr(init_database, "refresh_universe", _boom)

    with pytest.raises(RuntimeError, match="wikipedia"):
        init_database.initialise()

    assert _recorded["order"] == ["migrate"]


def test_the_dashboard_exposes_it_as_a_job():
    """The hosted deployment has no shell — the button is the only way in."""
    from monitoring.dashboard import server

    assert "init_db" in server._JOB_COMMANDS
    assert "scripts.init_database" in server._JOB_COMMANDS["init_db"]["command"]
