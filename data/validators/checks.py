"""
Sanity checks on incoming data, run before anything downstream consumes it.

The plan calls this "the #1 source of silent bugs later" — the goal here
isn't to be exhaustive, it's to catch the specific failure modes that
actually happen with market data feeds:
  - a symbol silently stops updating (staleness)
  - a vendor sends the same bar twice under different ingestion timestamps
    (duplicates on the business key, not the DB primary key)
  - a symbol has a gap in its expected calendar (missing trading days)
"""
from __future__ import annotations

import pandas as pd


def check_duplicates(df: pd.DataFrame, key_cols: list[str]) -> list[str]:
    """Flag rows that share a business key (e.g. symbol+ts) more than once."""
    issues = []
    dupe_mask = df.duplicated(subset=key_cols, keep=False)
    if dupe_mask.any():
        dupes = df.loc[dupe_mask, key_cols].drop_duplicates()
        for _, row in dupes.iterrows():
            issues.append(f"duplicate key: {dict(row)}")
    return issues


def check_gaps(df: pd.DataFrame, symbol_col: str = "symbol", ts_col: str = "ts", expect_daily: bool = True) -> list[str]:
    """
    For each symbol, flag missing business days between its min and max
    timestamp in this batch. Weekends are excluded via pandas' business-day
    calendar; this does NOT account for market holidays, so a small number
    of expected "gaps" around holidays is normal — treat clusters of gaps,
    not single ones, as the real signal.
    """
    issues = []
    if not expect_daily or df.empty:
        return issues

    for symbol, sub in df.groupby(symbol_col):
        ts = pd.to_datetime(sub[ts_col]).dt.tz_localize(None).dt.normalize().sort_values().unique()
        if len(ts) < 2:
            continue
        expected = pd.bdate_range(start=ts.min(), end=ts.max())
        missing = expected.difference(pd.DatetimeIndex(ts))
        if len(missing) > 0:
            issues.append(
                f"{symbol}: {len(missing)} missing business day(s) between "
                f"{ts.min().date()} and {ts.max().date()} (e.g. {missing[0].date()})"
            )
    return issues


def check_staleness(df: pd.DataFrame, symbol_col: str = "symbol", ts_col: str = "ts", max_age_days: int = 5) -> list[str]:
    """Flag symbols whose latest bar in this batch is older than max_age_days."""
    issues = []
    if df.empty:
        return issues

    now = pd.Timestamp.now(tz="UTC")
    latest = df.groupby(symbol_col)[ts_col].max()
    for symbol, ts in latest.items():
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        age = (now - ts).days
        if age > max_age_days:
            issues.append(f"{symbol}: latest bar is {age} day(s) old ({ts.date()})")
    return issues


def check_nonpositive_prices(df: pd.DataFrame, price_cols: tuple[str, ...] = ("open", "high", "low", "close")) -> list[str]:
    """
    Flag any bar with a zero, negative, or missing OHLC value.

    A stock's price is a physical impossibility below zero, but nothing
    upstream of this enforced that — a bad vendor row (a decimal-shift glitch,
    a placeholder 0.0, a corporate-action mixup) could reach `close` unchecked
    and, through rolling_return()'s simple pct_change, an already-impossible
    close (say 0 or negative against a positive prior close) shows up
    downstream as a return below -100% -- e.g. a reported "-118.7% in 5
    days," which cannot happen to a real security no matter how bad the week
    was. Existing checks here (duplicates/gaps/staleness/nulls) never caught
    this because none of them look at the price values themselves. Symbols
    are returned present-in-the-issue so the caller can identify and drop
    just the bad rows rather than the whole batch.
    """
    issues = []
    cols = [c for c in price_cols if c in df.columns]
    if not cols:
        return issues
    bad_mask = (df[cols] <= 0).any(axis=1) | df[cols].isna().any(axis=1)
    if bad_mask.any():
        bad = df.loc[bad_mask]
        symbol_col = "symbol" if "symbol" in df.columns else None
        ts_col = "ts" if "ts" in df.columns else None
        for _, row in bad.iterrows():
            where = f"{row[symbol_col]} on {row[ts_col]}" if symbol_col and ts_col else "a row"
            values = {c: row[c] for c in cols}
            issues.append(f"non-positive or missing price: {where} — {values}")
    return issues


def check_nulls(df: pd.DataFrame, required_cols: list[str]) -> list[str]:
    issues = []
    for col in required_cols:
        if col not in df.columns:
            issues.append(f"missing required column: {col}")
            continue
        n_null = df[col].isna().sum()
        if n_null:
            issues.append(f"{n_null} null value(s) in required column '{col}'")
    return issues


def run_all_validators(
    df: pd.DataFrame,
    key_cols: list[str],
    expect_daily: bool = False,
    required_cols: list[str] | None = None,
    max_age_days: int = 5,
) -> list[str]:
    """Run the full validator suite and return a flat list of human-readable issues."""
    issues: list[str] = []
    issues += check_duplicates(df, key_cols)
    issues += check_nonpositive_prices(df)
    if expect_daily:
        issues += check_gaps(df, expect_daily=True)
        issues += check_staleness(df, max_age_days=max_age_days)
    if required_cols:
        issues += check_nulls(df, required_cols)
    return issues
