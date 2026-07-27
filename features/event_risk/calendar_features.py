"""
Event-risk features: countdown to the next known risk event, plus how
volatile the symbol has historically been around similar past events.
"""
from __future__ import annotations

import pandas as pd


def days_to_next_event(as_of: pd.Timestamp, event_dates: pd.Series) -> float:
    """Calendar days until the next event on/after `as_of`. NaN if none scheduled."""
    upcoming = event_dates[event_dates >= as_of]
    if upcoming.empty:
        return float("nan")
    return (upcoming.min() - as_of).days


def days_to_next_macro_event(as_of: pd.Timestamp, macro_calendar: pd.DataFrame, category: str | None = None) -> float:
    """
    macro_calendar: dataframe with columns ['ts', 'category'] (from the
    macro_calendar table). If `category` is given, restricts to that type
    (e.g. only 'FOMC'); otherwise considers any scheduled macro event.
    """
    df = macro_calendar
    if category:
        df = df[df["category"] == category]
    return days_to_next_event(as_of, df["ts"])


def historical_event_vol(
    realized_vol_series: pd.Series,
    event_dates: pd.Series,
    window_days: int = 3,
) -> float:
    """
    Average realized volatility in the `window_days` around each past event
    date, as a baseline for "how much should we expect to move" heading into
    the next one. Returns NaN if there's no history to draw on yet.
    """
    samples = []
    for event_date in event_dates:
        window = realized_vol_series.loc[
            (realized_vol_series.index >= event_date - pd.Timedelta(days=window_days))
            & (realized_vol_series.index <= event_date + pd.Timedelta(days=window_days))
        ]
        if not window.empty:
            samples.append(window.mean())
    if not samples:
        return float("nan")
    return sum(samples) / len(samples)
