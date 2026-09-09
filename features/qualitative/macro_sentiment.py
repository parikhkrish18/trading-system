"""
Phase 2+ macro/sector sentiment features -- the market-wide and
sector-wide counterpart to features/qualitative/sentiment.py's per-stock
scoring.

News vendors tag articles with far more than the specific company a story
is about -- a Fed/oil/earnings-season piece routinely also carries broad
index and sector ETF tags (SPY, QQQ, XLK, ...) alongside whatever
individual stocks it mentions (see data/ingest/news_stream.py's
article_to_rows and data/ingest/news.py's equivalent: one row per symbol
the vendor tagged it with). Those rows already land in news_events and
already get scored by features/qualitative/sentiment.py's normal batch
pass -- nothing about ingestion or scoring needs to change for the data
to exist. What's missing is (a) making sure the proxy tickers below are
explicitly subscribed/ingested rather than only caught incidentally when
a broad story also happens to name a subscribed stock (see
data/ingest/news_stream.py and scripts/run_weekly_cycle.py), and (b)
turning that per-proxy sentiment into features every stock can use.

Deliberately NOT a hand-coded "if market bearish, favor shorts" rule --
these are plain feature columns like every other one in build_features.py
(mom_ret_*, sentiment_mean_*, ...); the model decides on its own, from
training data, how much a bearish market/sector backdrop should matter
and how it should interact with a stock's own signals. See
build_macro_interaction_features for the one deliberate exception.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The 11 SPDR Select Sector ETFs, keyed by the exact GICS sector string
# data/ingest/universe.py::fetch_sp500_constituents scrapes into
# universe.gics_sector (Wikipedia's S&P 500 table column -- the standard
# GICS sector names).
SECTOR_ETF_BY_GICS_SECTOR = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

# Broad-market proxies -- not sector-specific, describe the whole tape.
MARKET_PROXY_SYMBOLS = ["SPY", "QQQ", "DIA", "IWM"]

# Every ticker news ingestion needs to explicitly subscribe to/pull for
# this feature to have reliable (not incidental) coverage -- see
# data/ingest/news_stream.py and scripts/run_weekly_cycle.py.
MACRO_PROXY_SYMBOLS = MARKET_PROXY_SYMBOLS + list(SECTOR_ETF_BY_GICS_SECTOR.values())

# Bounded window + a short half-life within it: "stress the decay" -- a
# same-day headline should utterly dominate one from a few days ago, not
# be flatly averaged against it the way sentiment_mean_3d/10d average the
# per-stock signal. At half_life=12h: 1 day old carries 25% weight, 2 days
# ~6%, 3 days ~1.6%, the 4-day window edge ~0.4% -- effectively zero, so
# the explicit bound below mostly just caps the query/computation size;
# the decay itself already renders anything past ~2-3 days negligible.
_DECAY_HALF_LIFE_HOURS = 12.0
_LOOKBACK_HOURS = 96.0


def _recency_weighted_asof(news_ts: pd.Series, news_sentiment: pd.Series, anchor_ts: pd.Series) -> np.ndarray:
    """
    For each timestamp in `anchor_ts`, the recency-weighted mean of every
    `news_sentiment` value whose `news_ts` falls in
    (anchor - _LOOKBACK_HOURS, anchor] -- point-in-time correct (only ever
    looks backward from each anchor), so this is safe for historical
    feature backfill, not just a live "now" read.

    Deliberately NOT pandas .ewm(times=...): that computes a running
    average that only updates ON a real observation -- an anchor with no
    new data in between just replays the last computed value unchanged,
    regardless of how much time has actually passed (verified directly
    before writing this). That is exactly backwards for "stress the
    decay" -- a reading should visibly fade toward "no signal" as it ages
    even with no new headlines, not persist at full strength indefinitely.
    This instead recomputes each anchor's weighted average directly from
    the bounded window, so a stale-but-still-in-window reading correctly
    carries less weight the further the anchor sits past it, and an
    anchor past the whole window gets NaN (no signal), not a fossilized
    old value.
    """
    news_ts_arr = news_ts.to_numpy(dtype="datetime64[ns]")
    news_sentiment_arr = news_sentiment.to_numpy(dtype=float)
    anchor_arr = anchor_ts.to_numpy(dtype="datetime64[ns]")

    result = np.full(len(anchor_arr), np.nan)
    for i, t in enumerate(anchor_arr):
        age_hours = (t - news_ts_arr) / np.timedelta64(1, "h")
        in_window = (age_hours >= 0) & (age_hours <= _LOOKBACK_HOURS)
        if not in_window.any():
            continue
        weights = 0.5 ** (age_hours[in_window] / _DECAY_HALF_LIFE_HOURS)
        result[i] = np.average(news_sentiment_arr[in_window], weights=weights)
    return result


def _proxy_sentiment_by_date(news: pd.DataFrame, proxy_symbols: list[str], dates: pd.Series) -> pd.Series:
    """
    Recency-weighted mean sentiment across every symbol in `proxy_symbols`
    pooled together (e.g. all 4 broad-market ETFs combined into one
    market-wide read, or a single sector's ETF alone), one value per date
    in `dates`. `news` must already be filtered to sentiment_relevant !=
    False rows with a non-null sentiment (see build_macro_sentiment_features).
    """
    pooled = news[news["symbol"].isin(proxy_symbols)]
    if pooled.empty:
        return pd.Series(np.nan, index=dates.index)
    values = _recency_weighted_asof(pooled["ts"], pooled["sentiment"], dates)
    return pd.Series(values, index=dates.index)


def build_macro_sentiment_features(
    prices: pd.DataFrame, macro_news: pd.DataFrame, sector_by_symbol: dict[str, str]
) -> pd.DataFrame:
    """
    prices: columns [symbol, ts, ...] -- only symbol/ts are used, one row
    per (stock, trading day) the feature needs a value for.
    macro_news: columns [symbol, ts, sentiment] (+ optional
    sentiment_relevant) from news_events, for MACRO_PROXY_SYMBOLS only --
    NOT the whole universe (see build_features.py::build_and_store's
    separate query for this).
    sector_by_symbol: {stock symbol: GICS sector string}, from the
    universe table. A stock missing from this map (a new/renamed
    constituent the sector scrape hasn't caught up to yet) simply gets no
    macro_sector_sentiment value for that date -- same as any other
    feature with incomplete inputs, not a reason to fail the whole batch.

    Returns [symbol, ts, feature_name, value] with two feature names:
    macro_mkt_sentiment (broadcast to every stock -- the same value across
    the whole universe on a given date) and macro_sector_sentiment
    (matched per stock via its own sector's ETF).
    """
    if prices.empty or macro_news.empty:
        return pd.DataFrame(columns=["symbol", "ts", "feature_name", "value"])

    news = macro_news.dropna(subset=["sentiment"])
    if "sentiment_relevant" in news.columns:
        news = news[news["sentiment_relevant"] != False]  # noqa: E712 — NaN-safe, see build_qualitative_features
    if news.empty:
        return pd.DataFrame(columns=["symbol", "ts", "feature_name", "value"])

    unique_dates = prices[["ts"]].drop_duplicates().reset_index(drop=True)
    frames: list[pd.DataFrame] = []

    market_values = _proxy_sentiment_by_date(news, MARKET_PROXY_SYMBOLS, unique_dates["ts"])
    market_by_date = unique_dates.assign(value=market_values.to_numpy())
    market_out = prices[["symbol", "ts"]].drop_duplicates().merge(market_by_date, on="ts", how="inner")
    market_out = market_out.dropna(subset=["value"])
    if not market_out.empty:
        market_out = market_out.assign(feature_name="macro_mkt_sentiment")
        frames.append(market_out[["symbol", "ts", "feature_name", "value"]])

    for sector, etf in SECTOR_ETF_BY_GICS_SECTOR.items():
        sector_symbols = [s for s, sec in sector_by_symbol.items() if sec == sector]
        if not sector_symbols:
            continue
        sector_values = _proxy_sentiment_by_date(news, [etf], unique_dates["ts"])
        sector_by_date = unique_dates.assign(value=sector_values.to_numpy())
        sector_prices = prices[prices["symbol"].isin(sector_symbols)][["symbol", "ts"]].drop_duplicates()
        sector_out = sector_prices.merge(sector_by_date, on="ts", how="inner").dropna(subset=["value"])
        if sector_out.empty:
            continue
        sector_out = sector_out.assign(feature_name="macro_sector_sentiment")
        frames.append(sector_out[["symbol", "ts", "feature_name", "value"]])

    if not frames:
        return pd.DataFrame(columns=["symbol", "ts", "feature_name", "value"])
    return pd.concat(frames, ignore_index=True)


def build_macro_interaction_features(macro_features: pd.DataFrame, own_sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    The explicit "weigh this more" mechanism: a raw magnitude on
    macro_mkt_sentiment/macro_sector_sentiment carries no special
    importance to a tree-based model on its own -- LightGBM splits on
    thresholds, not linear coefficients, so scaling a column up does not
    make the model rely on it more. The compound signal actually wanted --
    a stock's own bad news landing alongside a bad sector/market backdrop
    should count for more than either alone -- is instead handed to the
    model directly as its own column: macro_mkt_sentiment multiplied by
    the stock's own sentiment_mean_3d, and the same for the sector
    version. Matching signs (both negative, or both positive) make the
    product large and positive -- "aligned", either bearish or bullish;
    opposite signs make it negative -- "fighting the tape". A tree model
    can split directly on that product in one step, rather than having to
    approximate the interaction across several splits on the two raw
    features separately.

    macro_features: build_macro_sentiment_features' output.
    own_sentiment: [symbol, ts, feature_name, value] already containing
    "sentiment_mean_3d" rows (build_qualitative_features' output -- reused
    rather than recomputed here: one feature, one source of truth, for a
    stock's own short-term sentiment).
    """
    if macro_features.empty or own_sentiment.empty:
        return pd.DataFrame(columns=["symbol", "ts", "feature_name", "value"])

    own = own_sentiment.loc[own_sentiment["feature_name"] == "sentiment_mean_3d", ["symbol", "ts", "value"]]
    own = own.rename(columns={"value": "own_sentiment"})
    if own.empty:
        return pd.DataFrame(columns=["symbol", "ts", "feature_name", "value"])

    frames: list[pd.DataFrame] = []
    for macro_name, out_name in (
        ("macro_mkt_sentiment", "macro_mkt_x_own_sentiment"),
        ("macro_sector_sentiment", "macro_sector_x_own_sentiment"),
    ):
        macro = macro_features.loc[macro_features["feature_name"] == macro_name, ["symbol", "ts", "value"]]
        if macro.empty:
            continue
        joined = macro.merge(own, on=["symbol", "ts"], how="inner")
        if joined.empty:
            continue
        joined = joined.assign(value=joined["value"] * joined["own_sentiment"], feature_name=out_name)
        frames.append(joined[["symbol", "ts", "feature_name", "value"]])

    if not frames:
        return pd.DataFrame(columns=["symbol", "ts", "feature_name", "value"])
    return pd.concat(frames, ignore_index=True)
