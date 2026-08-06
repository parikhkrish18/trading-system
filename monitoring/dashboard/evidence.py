"""
Turns the stored `decision_evidence` numbers into sentences a non-finance
reader can act on — the "Why this pick?" panel under each row of the
Latest picks table.

Split from monitoring/dashboard/app.py on the same principle as picks.py:
nothing here touches Streamlit or the database, so the wording can be
tested directly. models/evidence.py owns the model's numbers; this module
owns only how they read.

The wording rule throughout: name the thing in ordinary words, say what
the model saw, then say which way it pushed. A reader who knows nothing
about ADX or z-scores should still be able to follow "this stock has been
moving steadily upward, and that's most of the reason it was picked".
"""
from __future__ import annotations

import pandas as pd

from models.regime.trend_chop_classifier import CHOP, TREND

COL_FACTOR = "What the model looked at"
COL_OBSERVED = "What it saw"
COL_EFFECT = "Effect on this pick"

PUSHED_UP = "pushed UP"
PUSHED_DOWN = "pushed DOWN"

NO_NEWS_DATA = "no news data ingested yet"

# Plain-English name for every feature features/build_features.py can produce.
# Anything not listed falls back to a de-underscored version of its own name,
# so a new feature degrades to readable-ish rather than to a crash.
FEATURE_LABELS: dict[str, str] = {
    "mom_ret_5d": "Momentum over the last week",
    "mom_ret_20d": "Momentum over the last month",
    "adx_14": "Trend strength (ADX)",
    "vol_realized_20d": "How jumpy the price has been lately",
    "vol_atr_14": "Typical size of a daily move",
    "vol_of_vol": "How unstable that jumpiness itself is",
    "meanrev_zscore_20d": "Distance from its own one-month average price",
    "meanrev_bollinger_pctb": "Where it sits in its recent price range",
    "meanrev_rsi_14": "Overbought / oversold gauge (RSI)",
    "sentiment_mean_10d": "News sentiment over the last 10 days",
    "sentiment_mean_3d": "News sentiment over the last 3 days",
    "sentiment_momentum_3v10": "News sentiment: last 3 days vs the last 10",
    "news_volume_3d": "How much news there's been in the last 3 days",
    "days_to_next_fomc": "Days until the next Fed interest-rate decision",
    "days_to_next_cpi": "Days until the next inflation report",
    "days_to_next_jobs": "Days until the next jobs report",
}

# Features carrying news sentiment — the panel calls these out separately
# because "absent" means something specific for them (nothing ingested yet)
# rather than "the model didn't care".
SENTIMENT_FEATURES = ("sentiment_mean_10d", "sentiment_mean_3d", "sentiment_momentum_3v10", "news_volume_3d")


def feature_label(feature_name: str) -> str:
    """Human name for a feature; falls back to the raw name made readable."""
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]
    if feature_name.startswith("fund_") and feature_name.endswith("_latest"):
        metric = feature_name[len("fund_") : -len("_latest")].replace("_", " ")
        return f"Company financials: {metric}"
    return feature_name.replace("_", " ").capitalize()


def _describe_percent(value: float) -> str:
    return f"{value:+.1%}"


def _describe_adx(value: float) -> str:
    # 25 is the threshold models/regime/trend_chop_classifier.py splits on.
    state = "trending" if value >= 25 else "drifting sideways"
    return f"{value:.0f} ({state})"


def _describe_rsi(value: float) -> str:
    if value >= 70:
        state = "overbought — has run up hard"
    elif value <= 30:
        state = "oversold — has been beaten down"
    else:
        state = "neither overbought nor oversold"
    return f"{value:.0f} ({state})"


def _describe_sentiment(value: float) -> str:
    if value > 0.15:
        state = "positive"
    elif value < -0.15:
        state = "negative"
    else:
        state = "roughly neutral"
    return f"{value:+.2f} ({state})"


def _describe_zscore(value: float) -> str:
    if value >= 1:
        state = "well above its average"
    elif value <= -1:
        state = "well below its average"
    else:
        state = "close to its average"
    return f"{value:+.1f} ({state})"


def _describe_days(value: float) -> str:
    days = int(round(value))
    return "today" if days == 0 else f"{days} day(s) away"


def _describe_count(value: float) -> str:
    count = int(round(value))
    return "no stories" if count == 0 else f"{count} story/stories"


_VALUE_FORMATTERS = {
    "mom_ret_5d": _describe_percent,
    "mom_ret_20d": _describe_percent,
    "vol_realized_20d": _describe_percent,
    "adx_14": _describe_adx,
    "meanrev_rsi_14": _describe_rsi,
    "meanrev_zscore_20d": _describe_zscore,
    "sentiment_mean_10d": _describe_sentiment,
    "sentiment_mean_3d": _describe_sentiment,
    "sentiment_momentum_3v10": _describe_sentiment,
    "news_volume_3d": _describe_count,
    "days_to_next_fomc": _describe_days,
    "days_to_next_cpi": _describe_days,
    "days_to_next_jobs": _describe_days,
}


def describe_value(feature_name: str, value: float | None) -> str:
    """
    The feature's own value, in the units a reader expects for that feature —
    a percentage for a price change, a plain number for an index like ADX,
    a countdown for a calendar feature.
    """
    if value is None or pd.isna(value):
        return "not available"
    formatter = _VALUE_FORMATTERS.get(feature_name)
    return formatter(float(value)) if formatter else f"{float(value):,.2f}"


def describe_effect(contribution: float, strongest: float) -> str:
    """
    Which way this feature pushed the forecast, and how hard relative to the
    strongest factor behind the same pick. Deliberately relative: the raw
    contributions are in units of forecast return and are tiny in absolute
    terms (a few tenths of a percent), so "0.004" tells a reader nothing
    while "this was the biggest single reason" tells them everything.
    """
    if contribution == 0:
        return "no effect"
    direction = PUSHED_UP if contribution > 0 else PUSHED_DOWN
    share = abs(contribution) / strongest if strongest else 0.0
    if share >= 0.66:
        strength = "strongly"
    elif share >= 0.33:
        strength = "moderately"
    else:
        strength = "slightly"
    return f"{strength} {direction}"


def evidence_table(evidence: pd.DataFrame) -> pd.DataFrame:
    """
    One symbol's `decision_evidence` rows as the three-column panel table,
    strongest factor first.

    `evidence`: columns [feature_name, feature_value, contribution,
    contribution_rank] — the rows stored by models.screener.log_evidence.
    """
    columns = [COL_FACTOR, COL_OBSERVED, COL_EFFECT]
    if evidence.empty:
        return pd.DataFrame(columns=columns)

    ordered = evidence.sort_values("contribution_rank")
    strongest = float(ordered["contribution"].abs().max())

    return pd.DataFrame(
        {
            COL_FACTOR: [feature_label(str(name)) for name in ordered["feature_name"]],
            COL_OBSERVED: [
                describe_value(str(name), value)
                for name, value in zip(ordered["feature_name"], ordered["feature_value"])
            ],
            COL_EFFECT: [describe_effect(float(c), strongest) for c in ordered["contribution"]],
        }
    ).reset_index(drop=True)


def news_sentiment_note(evidence: pd.DataFrame) -> str:
    """
    What the panel says about news, which is a different question from what
    the model's top factors were. Sentiment features only exist once news has
    been ingested *and* scored (features/build_features.py drops unscored
    rows), so their absence is a data-pipeline fact worth stating outright —
    silently omitting the row would read as "news said nothing", which is not
    the same as "we have no news".
    """
    if evidence.empty or "feature_name" not in evidence.columns:
        return f"News sentiment: {NO_NEWS_DATA}."

    rows = evidence.loc[evidence["feature_name"].isin(SENTIMENT_FEATURES)]
    rows = rows.loc[rows["feature_value"].notna()] if "feature_value" in rows.columns else rows.iloc[0:0]
    if rows.empty:
        return (
            f"News sentiment: {NO_NEWS_DATA} — this pick is based on price behaviour alone. "
            "Add a Polygon and an Anthropic key to start scoring headlines."
        )

    strongest_row = rows.loc[rows["contribution"].abs().idxmax()]
    name = str(strongest_row["feature_name"])
    return (
        f"News sentiment: {describe_value(name, strongest_row['feature_value'])} "
        f"({feature_label(name).lower()}) — "
        f"{describe_effect(float(strongest_row['contribution']), float(rows['contribution'].abs().max()))}."
    )


def regime_note(regime: str | None) -> str:
    """Plain-English version of the per-symbol regime tag, and what it did to the size."""
    if regime == TREND:
        return "Trending — it's been moving steadily one way, so this pick is sized normally."
    if regime == CHOP:
        return "Choppy — it's been drifting sideways, so this pick is deliberately sized smaller."
    return "Market regime unknown — not enough price history to tell yet."


def confidence_note(direction_agreement: float | None) -> str:
    """
    direction_agreement is the share of the ensemble's models that agreed on
    which way the price would go — the screener's own confidence measure. It
    runs from 0.5 (a dead split) to 1.0 (unanimous), so it's phrased as
    agreement rather than as a probability of being right, which is not what
    it measures.
    """
    if direction_agreement is None or pd.isna(direction_agreement):
        return "Model agreement: not recorded for this pick."
    share = float(direction_agreement)
    if share >= 0.999:
        return "Model agreement: unanimous — every model in the ensemble agreed on the direction."
    return (
        f"Model agreement: {share:.0%} of the models agreed on the direction "
        "(they are trained on the same data with different random seeds; "
        "disagreement is the honest measure of how sure the system is)."
    )
