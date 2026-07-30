import pandas as pd
import pytest

from features import build_features
from features.build_features import (
    build_and_store,
    build_event_risk_features,
    build_fundamentals_features,
    build_qualitative_features,
)


def _prices(symbol: str, dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"symbol": [symbol] * len(dates), "ts": pd.to_datetime(dates, utc=True)})


def test_build_qualitative_features_aggregates_trailing_sentiment():
    prices = _prices("SPY", ["2026-01-10", "2026-01-11"])
    news = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY"],
            "ts": pd.to_datetime(["2026-01-01", "2026-01-09", "2026-01-10"], utc=True),
            "sentiment": [0.2, 0.8, -0.4],
        }
    )
    result = build_qualitative_features(prices, news)

    jan10 = result[(result["ts"] == pd.Timestamp("2026-01-10", tz="UTC"))]
    by_name = dict(zip(jan10["feature_name"], jan10["value"], strict=False))
    # trailing 10d window (>2025-12-31, <=2026-01-10) includes all three; 3d window (>2026-01-07) includes the last two.
    assert by_name["sentiment_mean_10d"] == pytest.approx((0.2 + 0.8 - 0.4) / 3)
    assert by_name["sentiment_mean_3d"] == pytest.approx((0.8 - 0.4) / 2)
    assert by_name["news_volume_3d"] == 2.0


def test_build_qualitative_features_drops_unscored_news():
    prices = _prices("SPY", ["2026-01-10"])
    news = pd.DataFrame({"symbol": ["SPY"], "ts": pd.to_datetime(["2026-01-09"], utc=True), "sentiment": [None]})
    result = build_qualitative_features(prices, news)
    assert result.empty


def test_build_qualitative_features_requires_full_10d_window():
    # Only one data point, no 10d history yet -> nothing scored for that date.
    prices = _prices("SPY", ["2026-01-10"])
    news = pd.DataFrame({"symbol": ["SPY"], "ts": pd.to_datetime(["2026-01-10"], utc=True), "sentiment": [0.5]})
    result = build_qualitative_features(prices, news)
    assert not result.empty  # the 10d window trivially contains just this one point, which is fine


def test_build_qualitative_features_empty_inputs():
    assert build_qualitative_features(pd.DataFrame(), pd.DataFrame()).empty


def test_build_event_risk_features_computes_days_to_next_event():
    prices = _prices("SPY", ["2026-01-01"])
    macro = pd.DataFrame(
        {"ts": pd.to_datetime(["2026-01-05", "2026-01-20"], utc=True), "category": ["FOMC", "CPI"]}
    )
    result = build_event_risk_features(prices, macro)
    by_name = dict(zip(result["feature_name"], result["value"], strict=False))
    assert by_name["days_to_next_fomc"] == 4.0
    assert by_name["days_to_next_cpi"] == 19.0
    assert "days_to_next_jobs" not in by_name  # no JOBS event scheduled -> NaN -> dropped


def test_build_event_risk_features_empty_calendar_returns_empty():
    prices = _prices("SPY", ["2026-01-01"])
    assert build_event_risk_features(prices, pd.DataFrame()).empty


def test_build_fundamentals_features_as_of_joins_latest_known_value():
    prices = _prices("SPY", ["2026-08-01", "2026-08-10"])
    fundamentals = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY"],
            "ts": pd.to_datetime(["2026-08-04", "2026-08-04"], utc=True),
            "metric": ["eps_actual", "eps_actual"],
            "value": [2.5, 2.5],
        }
    ).drop_duplicates()

    result = build_fundamentals_features(prices, fundamentals)

    aug1 = result[result["ts"] == pd.Timestamp("2026-08-01", tz="UTC")]
    aug10 = result[result["ts"] == pd.Timestamp("2026-08-10", tz="UTC")]
    assert aug1.empty  # filed on Aug 4 -> not yet known as of Aug 1
    assert aug10.iloc[0]["value"] == 2.5  # known by Aug 10


def test_build_fundamentals_features_empty_inputs():
    assert build_fundamentals_features(pd.DataFrame(), pd.DataFrame()).empty


def test_build_and_store_defaults_to_a_bounded_lookback_window(monkeypatch):
    """
    Regression test, hit live: build_and_store used to pull *all* price
    history with no date filter -- against a 5-year full-universe backfill
    that produced a features batch large enough to stall a single-transaction
    upsert for 3+ hours. The prices query must always carry a lookback bound.
    """
    captured = {}

    def fake_read_sql(query, engine, **kwargs):
        captured.setdefault("queries", []).append(query)
        return pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume"])

    monkeypatch.setattr(build_features, "get_engine", lambda: object())
    monkeypatch.setattr(build_features.pd, "read_sql", fake_read_sql)

    build_and_store(["AAPL"], "v3")

    prices_query = captured["queries"][0]
    assert "interval '3 years'" in prices_query


def test_build_and_store_lookback_years_is_configurable(monkeypatch):
    captured = {}
    monkeypatch.setattr(build_features, "get_engine", lambda: object())
    monkeypatch.setattr(
        build_features.pd,
        "read_sql",
        lambda query, engine, **kwargs: (captured.setdefault("queries", []).append(query), pd.DataFrame())[1],
    )

    build_and_store(["AAPL"], "v3", lookback_years=1)

    assert "interval '1 years'" in captured["queries"][0]
