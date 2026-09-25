"""
On-demand intraday (sub-day) bars for volume-profile confirmation
(features/quant/volume_profile.py, models/screener.py's
_solidified_channel_distances) -- never persisted to any table, unlike
data/ingest/prices.py's daily bars.

The `prices` table carries exactly one volume number per symbol per day,
with no information on how that volume distributed across the day's price
range -- not enough to compute a real volume profile (see the module
docstring discussion this was built from). Alpaca is the only vendor
already wired into this repo with genuine sub-day price/volume (its market
data API is the same feed regardless of paper/live trading mode, per
execution/broker_alpaca.py's own comment) -- this reuses that same
StockHistoricalDataClient/StockBarsRequest pattern data/ingest/prices.py's
_fetch_alpaca already uses for daily bars, just with a shorter timeframe.

Deliberately not stored: persisting bar_minutes-resolution history for
every screened symbol, indefinitely, would be a real and ongoing storage
cost for a signal that only ever needs a short trailing window recomputed
fresh each cycle -- the same "fresh, targeted read" reasoning
donchian_channel_levels already uses for its own daily high/low query.
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


def fetch_recent_minute_bars(
    symbols: list[str],
    lookback_days: int | None = None,
    bar_minutes: int | None = None,
) -> pd.DataFrame:
    """
    Intraday bars for `symbols` over the trailing `lookback_days`, at
    `bar_minutes`-minute resolution (both default to settings.
    volume_profile_lookback_days/volume_profile_bar_minutes).

    Returns columns [symbol, ts, high, low, close, volume] -- open is
    fetched but dropped, since volume-profile binning only needs a
    representative price per bar (see features/quant/volume_profile.py's
    typical-price calculation) and every other consumer here only reads
    these five. Empty DataFrame (never raises) when credentials are unset,
    the request fails, or nothing comes back -- best-effort, same
    fail-open convention as donchian_channel_levels/_correlation_matrix:
    a volume-profile confirmation signal that can't be computed this cycle
    must never block or corrupt the screen that wants it, only cost that
    cycle the confirmation check (see is_level_volume_confirmed's own
    fail-open-to-True default for the other half of this).
    """
    if not symbols:
        return pd.DataFrame(columns=["symbol", "ts", "high", "low", "close", "volume"])
    if not settings.alpaca_paper_api_key or not settings.alpaca_paper_secret_key:
        return pd.DataFrame(columns=["symbol", "ts", "high", "low", "close", "volume"])

    lookback_days = lookback_days if lookback_days is not None else settings.volume_profile_lookback_days
    bar_minutes = bar_minutes if bar_minutes is not None else settings.volume_profile_bar_minutes

    try:
        from alpaca.data.enums import Adjustment
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        end = dt.datetime.now(tz=dt.UTC)
        start = end - dt.timedelta(days=lookback_days)
        client = StockHistoricalDataClient(settings.alpaca_paper_api_key, settings.alpaca_paper_secret_key)
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame(bar_minutes, TimeFrameUnit.Minute),
            start=start,
            end=end,
            # Same reasoning as _fetch_alpaca's daily-bar request: adjusted
            # for splits/dividends, matching every other price series in
            # this repo, so a level derived here lines up with the
            # adjusted `close` everything else is computed against.
            adjustment=Adjustment.ALL,
        )
        bars = client.get_stock_bars(req).df.reset_index()
    except Exception:
        logger.exception("Could not fetch intraday bars for volume-profile confirmation — skipping this cycle.")
        return pd.DataFrame(columns=["symbol", "ts", "high", "low", "close", "volume"])

    if bars.empty:
        return pd.DataFrame(columns=["symbol", "ts", "high", "low", "close", "volume"])
    bars = bars.rename(columns={"timestamp": "ts"})
    return bars[["symbol", "ts", "high", "low", "close", "volume"]]
