"""
Coverage for features/qualitative/macro_sentiment.py -- the market/sector
sentiment features and the explicit interaction terms with a stock's own
sentiment_mean_3d.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.qualitative import macro_sentiment as ms


def _news(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """rows: [(symbol, ts_iso, sentiment), ...]"""
    symbols, tss, sentiments = zip(*rows, strict=True) if rows else ([], [], [])
    return pd.DataFrame({"symbol": list(symbols), "ts": pd.to_datetime(list(tss), utc=True), "sentiment": list(sentiments)})


def _prices(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """rows: [(symbol, ts_iso), ...]"""
    symbols, tss = zip(*rows, strict=True) if rows else ([], [])
    return pd.DataFrame({"symbol": list(symbols), "ts": pd.to_datetime(list(tss), utc=True)})


# ---------------------------------------------------------------------
# _recency_weighted_asof
# ---------------------------------------------------------------------


def test_recency_weighted_asof_half_life_math_is_exact():
    """Locks in the decay formula: a headline exactly one half-life old
    carries exactly half the weight of a fresh one, same convention as
    execution/contradiction_monitor.py::_recent_sentiment."""
    anchor = pd.Timestamp("2026-01-02T00:00:00Z")
    news_ts = pd.Series(
        [anchor, anchor - pd.Timedelta(hours=ms._DECAY_HALF_LIFE_HOURS)]
    )
    news_sentiment = pd.Series([1.0, 0.0])

    result = ms._recency_weighted_asof(news_ts, news_sentiment, pd.Series([anchor]))

    # weight(fresh)=1.0, weight(1 half-life old)=0.5 -> (1.0*1.0 + 0.0*0.5) / 1.5
    assert result[0] == pytest.approx(1.0 / 1.5, abs=1e-9)


def test_recency_weighted_asof_never_looks_forward():
    """A headline published AFTER the anchor must not contribute -- this is
    what makes the feature safe for point-in-time historical backfill."""
    anchor = pd.Timestamp("2026-01-01T00:00:00Z")
    news_ts = pd.Series([anchor - pd.Timedelta(hours=1), anchor + pd.Timedelta(hours=1)])
    news_sentiment = pd.Series([0.5, -0.9])  # the future one is extreme -- must not leak in

    result = ms._recency_weighted_asof(news_ts, news_sentiment, pd.Series([anchor]))

    assert result[0] == pytest.approx(0.5, abs=1e-9)


def test_recency_weighted_asof_excludes_anything_past_the_lookback_window():
    """Regression target for 'stress the decay': news outside the bounded
    window must read as NO signal (NaN), not a fossilized old value --
    unlike pandas .ewm(times=...), which was verified to just replay the
    last computed value forever with no new observations."""
    anchor = pd.Timestamp("2026-01-10T00:00:00Z")
    just_outside = anchor - pd.Timedelta(hours=ms._LOOKBACK_HOURS + 1)
    news_ts = pd.Series([just_outside])
    news_sentiment = pd.Series([0.9])

    result = ms._recency_weighted_asof(news_ts, news_sentiment, pd.Series([anchor]))

    assert np.isnan(result[0])


def test_recency_weighted_asof_a_fresh_headline_dominates_an_older_opposite_one():
    """A same-day headline should overwhelm a 3-day-old opposite-sign one,
    not be flatly averaged against it -- the actual point of 'stress the
    decay' relative to the per-stock sentiment_mean_3d/10d flat mean."""
    anchor = pd.Timestamp("2026-01-04T00:00:00Z")
    news_ts = pd.Series([anchor - pd.Timedelta(hours=1), anchor - pd.Timedelta(hours=72)])
    news_sentiment = pd.Series([0.9, -0.9])  # fresh: strongly positive; 3 days old: strongly negative

    result = ms._recency_weighted_asof(news_ts, news_sentiment, pd.Series([anchor]))

    assert result[0] > 0.8  # the 1-hour-old headline dominates despite the older one being equally extreme


# ---------------------------------------------------------------------
# build_macro_sentiment_features
# ---------------------------------------------------------------------


def test_build_macro_sentiment_features_broadcasts_market_sentiment_to_every_stock():
    prices = _prices([("AAPL", "2026-01-02"), ("MSFT", "2026-01-02")])
    macro_news = _news([("SPY", "2026-01-01T23:00:00Z", 0.6), ("QQQ", "2026-01-01T23:30:00Z", 0.4)])

    result = ms.build_macro_sentiment_features(prices, macro_news, sector_by_symbol={})

    mkt = result[result["feature_name"] == "macro_mkt_sentiment"]
    assert set(mkt["symbol"]) == {"AAPL", "MSFT"}
    assert mkt["value"].nunique() == 1  # identical value broadcast to every stock on this date


def test_build_macro_sentiment_features_matches_sector_via_gics_sector():
    prices = _prices([("AAPL", "2026-01-02"), ("JPM", "2026-01-02")])
    macro_news = _news(
        [
            ("XLK", "2026-01-01T23:00:00Z", 0.8),  # tech sector -- should reach AAPL only
            ("XLF", "2026-01-01T23:00:00Z", -0.8),  # financials -- should reach JPM only
        ]
    )
    sector_by_symbol = {"AAPL": "Information Technology", "JPM": "Financials"}

    result = ms.build_macro_sentiment_features(prices, macro_news, sector_by_symbol)

    sector = result[result["feature_name"] == "macro_sector_sentiment"].set_index("symbol")["value"]
    assert sector["AAPL"] == pytest.approx(0.8, abs=1e-6)
    assert sector["JPM"] == pytest.approx(-0.8, abs=1e-6)


def test_build_macro_sentiment_features_stock_with_unmapped_sector_gets_no_sector_value():
    prices = _prices([("NEWCO", "2026-01-02")])
    macro_news = _news([("XLK", "2026-01-01T23:00:00Z", 0.8)])

    result = ms.build_macro_sentiment_features(prices, macro_news, sector_by_symbol={})  # NEWCO missing

    assert result[result["feature_name"] == "macro_sector_sentiment"].empty
    # Market-wide feature still requires at least one market-proxy headline --
    # none present here (only a sector ETF), so nothing at all should exist.
    assert result.empty


def test_build_macro_sentiment_features_drops_mistagged_and_unscored_rows():
    prices = _prices([("AAPL", "2026-01-02")])
    macro_news = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY"],
            "ts": pd.to_datetime(["2026-01-01T20:00:00Z"] * 3, utc=True),
            "sentiment": [0.9, -0.9, None],
            "sentiment_relevant": [True, False, True],
        }
    )

    result = ms.build_macro_sentiment_features(prices, macro_news, sector_by_symbol={})

    mkt = result[result["feature_name"] == "macro_mkt_sentiment"]
    assert mkt.iloc[0]["value"] == pytest.approx(0.9, abs=1e-6)  # only the relevant, scored row counts


def test_build_macro_sentiment_features_empty_inputs():
    assert ms.build_macro_sentiment_features(pd.DataFrame(), pd.DataFrame(), {}).empty
    assert ms.build_macro_sentiment_features(_prices([("AAPL", "2026-01-02")]), pd.DataFrame(), {}).empty


# ---------------------------------------------------------------------
# build_macro_interaction_features
# ---------------------------------------------------------------------


def _own_sentiment(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    symbols, tss, values = zip(*rows, strict=True)
    return pd.DataFrame(
        {
            "symbol": list(symbols),
            "ts": pd.to_datetime(list(tss), utc=True),
            "feature_name": "sentiment_mean_3d",
            "value": list(values),
        }
    )


def test_macro_interaction_multiplies_matching_symbol_and_date():
    macro = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "ts": pd.to_datetime(["2026-01-02"], utc=True),
            "feature_name": "macro_mkt_sentiment",
            "value": [0.5],
        }
    )
    own = _own_sentiment([("AAPL", "2026-01-02", -0.4)])

    result = ms.build_macro_interaction_features(macro, own)

    row = result[result["feature_name"] == "macro_mkt_x_own_sentiment"].iloc[0]
    assert row["value"] == pytest.approx(0.5 * -0.4)


def test_macro_interaction_positive_when_signs_align_negative_when_they_fight():
    macro = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "ts": pd.to_datetime(["2026-01-02", "2026-01-02"], utc=True),
            "feature_name": ["macro_sector_sentiment", "macro_sector_sentiment"],
            "value": [-0.6, -0.6],  # bearish sector backdrop for both
        }
    )
    own = _own_sentiment([("A", "2026-01-02", -0.7), ("B", "2026-01-02", 0.7)])  # A: bad news too; B: good news

    result = ms.build_macro_interaction_features(macro, own)
    by_symbol = result[result["feature_name"] == "macro_sector_x_own_sentiment"].set_index("symbol")["value"]

    assert by_symbol["A"] > 0  # both bearish -- aligned, compounding
    assert by_symbol["B"] < 0  # stock bullish against a bearish sector -- fighting the tape


def test_macro_interaction_only_joins_matching_symbol_and_date():
    macro = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "ts": pd.to_datetime(["2026-01-02"], utc=True),
            "feature_name": "macro_mkt_sentiment",
            "value": [0.5],
        }
    )
    own = _own_sentiment([("MSFT", "2026-01-02", -0.4)])  # different symbol -- no match

    result = ms.build_macro_interaction_features(macro, own)

    assert result.empty


def test_macro_interaction_empty_inputs():
    assert ms.build_macro_interaction_features(pd.DataFrame(), pd.DataFrame()).empty


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------


def test_macro_proxy_symbols_is_the_union_of_market_and_sector_etfs():
    assert set(ms.MACRO_PROXY_SYMBOLS) == set(ms.MARKET_PROXY_SYMBOLS) | set(ms.SECTOR_ETF_BY_GICS_SECTOR.values())
    assert len(ms.SECTOR_ETF_BY_GICS_SECTOR) == 11  # the 11 standard GICS sectors, no more, no fewer
