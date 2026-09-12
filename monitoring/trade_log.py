"""
Pure computation for the dashboard's "Trade Log" panel -- the raw, one-row-
per-executed-decision study data: every real (paper/live) open, add,
reduce, and close, with its forecast, regime, confidence, model reasoning,
outcome (once matured), full feature vector, and nearby news headlines.

Deliberately different from monitoring/model_report_card.py's
"trades taken" accounting and monitoring/forward_test.py's graded-decision
accuracy: both of those filter to target_position != 0 (a close carries no
new directional call, so it doesn't count toward "capital deployed" or
"was the direction right"). This module's job is the opposite -- show
every executed action as its own row, closes included, since a close is
still something the system actually did and is worth being able to study.

mode='backfill' (replayed history) is still always excluded -- this is a
record of what the live system did, never a replay.
"""
from __future__ import annotations

import pandas as pd

from monitoring.forecast_accuracy import compute_forecast_accuracy

LIVE_DECISION_MODES = ("paper", "live")


def real_trade_rows(decisions: pd.DataFrame) -> pd.DataFrame:
    """
    Every real (paper/live) decision that actually executed something --
    filtered on executed_position being set, not target_position, so a
    close (executed_position == 0, flattening a prior position) counts as
    an executed trade in its own right rather than being dropped.
    """
    if decisions.empty:
        return decisions
    real = decisions[decisions["mode"].isin(LIVE_DECISION_MODES)]
    return real[real["executed_position"].notna()]


def trade_direction(target_position: float | None) -> str:
    """long/short from the model's directional call; "close" for a flatten (target_position == 0)."""
    if target_position is None or pd.isna(target_position):
        return "unknown"
    if target_position > 0:
        return "long"
    if target_position < 0:
        return "short"
    return "close"


def grade_outcomes(trades: pd.DataFrame, prices: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    """
    Left-joins realized_return/hit (monitoring.forecast_accuracy, same
    grading every other panel uses) back onto every row that has a
    forecast to grade -- rows without one (e.g. a close logged with no
    forecast) or not yet matured simply carry NaN outcome columns rather
    than being dropped, since this is a record of every trade, not just
    the graded ones.
    """
    outcome_cols = ["realized_return", "hit"]
    if trades.empty:
        return trades.assign(**{c: pd.Series(dtype="float64") for c in outcome_cols})

    gradeable = trades[["symbol", "ts", "forecast"]].dropna(subset=["forecast"])
    scored = compute_forecast_accuracy(gradeable, prices, horizon_bars=horizon_bars)
    return trades.merge(scored[["symbol", "ts", *outcome_cols]], on=["symbol", "ts"], how="left")


def pivot_feature_vector(features_long: pd.DataFrame) -> dict[str, float]:
    """features_long: columns feature_name, value for ONE (symbol, feature_set_id, ts). {} if none stored."""
    if features_long.empty:
        return {}
    return dict(zip(features_long["feature_name"], features_long["value"], strict=True))


def attach_feature_vectors(trades: pd.DataFrame, features_long: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized version of pivot_feature_vector for the CSV export -- every
    feature the model saw becomes its own "feat_<name>" column, merged on
    the (symbol, feature_set_id, ts) the decision was made under. A best-
    effort join: features aren't stored per-decision, only per (symbol,
    feature_set_id, ts) they were computed at, so a decision whose ts
    doesn't exactly match a features row gets NaN for every feat_ column.
    """
    if trades.empty or features_long.empty:
        return trades
    wide = features_long.pivot_table(
        index=["symbol", "feature_set_id", "ts"], columns="feature_name", values="value", aggfunc="first"
    ).reset_index()
    wide.columns = [c if c in ("symbol", "feature_set_id", "ts") else f"feat_{c}" for c in wide.columns]
    return trades.merge(wide, on=["symbol", "feature_set_id", "ts"], how="left")


def nearby_news(symbol_news: pd.DataFrame, decision_ts, window_days: int = 3) -> pd.DataFrame:
    """
    symbol_news: any columns, for ONE symbol, with a `ts` column. Rows
    within +/- window_days of decision_ts, newest first -- approximate
    "what was in the news around this decision", not an authoritative
    record of what the model actually weighed (there's no foreign key from
    decisions to news_events).
    """
    if symbol_news.empty:
        return symbol_news
    lo, hi = decision_ts - pd.Timedelta(days=window_days), decision_ts + pd.Timedelta(days=window_days)
    return symbol_news[(symbol_news["ts"] >= lo) & (symbol_news["ts"] <= hi)].sort_values("ts", ascending=False)


def nearby_headlines(symbol_news: pd.DataFrame, decision_ts, window_days: int = 3) -> list[str]:
    """Just the headlines from nearby_news, for the CSV export's single flattened column."""
    nearby = nearby_news(symbol_news, decision_ts, window_days)
    return [] if nearby.empty else nearby["headline"].tolist()


def attach_nearby_headlines(trades: pd.DataFrame, news: pd.DataFrame, window_days: int = 3) -> pd.DataFrame:
    """Vectorized version of nearby_headlines for the CSV export -- one semicolon-joined string column."""
    if trades.empty:
        return trades

    def _for_row(row: pd.Series) -> str:
        symbol_news = news[news["symbol"] == row["symbol"]] if not news.empty else news
        return "; ".join(nearby_headlines(symbol_news, row["ts"], window_days))

    result = trades.copy()
    result["nearby_headlines"] = result.apply(_for_row, axis=1)
    return result
