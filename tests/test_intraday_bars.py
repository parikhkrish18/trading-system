from __future__ import annotations

import pandas as pd

from config.settings import settings
from data.ingest import intraday_bars


class _FakeBarsResponse:
    def __init__(self, df: pd.DataFrame):
        self.df = df


def _empty_bars_df() -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples([], names=["symbol", "timestamp"])
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"], index=idx)


def _some_bars_df() -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples(
        [("AAPL", pd.Timestamp("2026-01-02 14:30", tz="UTC")), ("AAPL", pd.Timestamp("2026-01-02 14:45", tz="UTC"))],
        names=["symbol", "timestamp"],
    )
    return pd.DataFrame(
        {"open": [100.0, 101.0], "high": [101.0, 102.0], "low": [99.0, 100.0], "close": [100.5, 101.5], "volume": [1000.0, 1200.0]},
        index=idx,
    )


def test_returns_empty_for_no_symbols(monkeypatch):
    monkeypatch.setattr(settings, "alpaca_paper_api_key", "key")
    monkeypatch.setattr(settings, "alpaca_paper_secret_key", "secret")
    result = intraday_bars.fetch_recent_minute_bars([])
    assert result.empty
    assert list(result.columns) == ["symbol", "ts", "high", "low", "close", "volume"]


def test_returns_empty_when_credentials_are_unset_without_calling_the_client(monkeypatch):
    monkeypatch.setattr(settings, "alpaca_paper_api_key", "")
    monkeypatch.setattr(settings, "alpaca_paper_secret_key", "")
    called = []
    import alpaca.data.historical as alpaca_historical

    class _ShouldNotBeCalled:
        def __init__(self, *a, **k):
            called.append(1)

    monkeypatch.setattr(alpaca_historical, "StockHistoricalDataClient", _ShouldNotBeCalled)

    result = intraday_bars.fetch_recent_minute_bars(["AAPL"])

    assert result.empty
    assert called == []


def test_requests_split_and_dividend_adjusted_bars_at_the_configured_timeframe(monkeypatch):
    monkeypatch.setattr(settings, "alpaca_paper_api_key", "key")
    monkeypatch.setattr(settings, "alpaca_paper_secret_key", "secret")
    monkeypatch.setattr(settings, "volume_profile_bar_minutes", 15)
    monkeypatch.setattr(settings, "volume_profile_lookback_days", 20)

    import alpaca.data.historical as alpaca_historical
    from alpaca.data.enums import Adjustment
    from alpaca.data.timeframe import TimeFrameUnit

    captured = {}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def get_stock_bars(self, req):
            captured["adjustment"] = req.adjustment
            captured["timeframe"] = req.timeframe
            return _FakeBarsResponse(_empty_bars_df())

    monkeypatch.setattr(alpaca_historical, "StockHistoricalDataClient", _FakeClient)

    intraday_bars.fetch_recent_minute_bars(["AAPL"])

    assert captured["adjustment"] == Adjustment.ALL
    assert captured["timeframe"].amount_value == 15
    assert captured["timeframe"].unit == TimeFrameUnit.Minute


def test_returns_the_expected_columns_on_a_successful_fetch(monkeypatch):
    monkeypatch.setattr(settings, "alpaca_paper_api_key", "key")
    monkeypatch.setattr(settings, "alpaca_paper_secret_key", "secret")

    import alpaca.data.historical as alpaca_historical

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def get_stock_bars(self, req):
            return _FakeBarsResponse(_some_bars_df())

    monkeypatch.setattr(alpaca_historical, "StockHistoricalDataClient", _FakeClient)

    result = intraday_bars.fetch_recent_minute_bars(["AAPL"])

    assert list(result.columns) == ["symbol", "ts", "high", "low", "close", "volume"]
    assert len(result) == 2
    assert set(result["symbol"]) == {"AAPL"}


def test_returns_empty_and_never_raises_when_the_client_fails(monkeypatch):
    monkeypatch.setattr(settings, "alpaca_paper_api_key", "key")
    monkeypatch.setattr(settings, "alpaca_paper_secret_key", "secret")

    import alpaca.data.historical as alpaca_historical

    class _BoomClient:
        def __init__(self, *a, **k):
            pass

        def get_stock_bars(self, req):
            raise RuntimeError("boom")

    monkeypatch.setattr(alpaca_historical, "StockHistoricalDataClient", _BoomClient)

    result = intraday_bars.fetch_recent_minute_bars(["AAPL"])  # must not raise

    assert result.empty
